from datetime import datetime, timezone
from contextlib import asynccontextmanager
import os
import shutil
import uuid
import uuid
import warnings
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import requests
import textwrap
from fastapi import Form

# SQLModel and database imports
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from sqlmodel import Session, select, delete
from sqlalchemy.orm import selectinload # Important for fetching relationships efficiently
from database import get_session
from models import Conversation, Message, FileChunk
from schemas import ConversationRead, RenameRequest, ChatResponse

# --- LangChain & Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, 
    UnstructuredPDFLoader, 
    PDFPlumberLoader
)
from langchain_core.documents import Document

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploaded_docs"
VECTOR_DB_PATH = "vector_store/faiss_index"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# --- Global State ---
# We load the heavy embedding model ONCE at startup.
print("--- STARTUP: Loading Embedding Model... ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# We hold the Vector DB in memory. It will be updated when files are uploaded.
vector_store = None

def load_vector_store():
    """Attempts to load the existing FAISS index from disk."""
    global vector_store
    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        try:
            vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("--- STARTUP: Loaded existing FAISS index. ---")
        except Exception as e:
            print(f"--- STARTUP: Could not load existing index: {e} ---")
            vector_store = None
    else:
        print("--- STARTUP: No existing index found. Waiting for uploads. ---")
        vector_store = None

# Load immediately on import
load_vector_store()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions can go here (if needed)
    try:
        yield
    finally:
        # Shutdown cleanup: remove uploaded files and persisted vector DB
        global vector_store
        try:
            # Remove uploaded files
            if os.path.exists(UPLOAD_DIR):
                for name in os.listdir(UPLOAD_DIR):
                    path = os.path.join(UPLOAD_DIR, name)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove upload path {path}: {e}")

            # Remove vector DB files
            if os.path.exists(VECTOR_DB_PATH):
                for name in os.listdir(VECTOR_DB_PATH):
                    path = os.path.join(VECTOR_DB_PATH, name)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove vector DB path {path}: {e}")

            # Clear in-memory reference
            vector_store = None
            logger.info("Shutdown cleanup completed: uploaded files and vector DB cleared.")
        except Exception as e:
            logger.error(f"Shutdown cleanup failed: {e}")

app = FastAPI(title="RAG API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper: PDF Processing (From your ingest.py) ---
def process_pdf(file_path: str) -> List[Document]:
    """Robust PDF loader with multiple fallbacks."""
    docs = []
    
    # 1. PyMuPDF (fitz) - good for complex layouts
    try:
        import fitz  # PyMuPDF  # type: ignore
        docs = []
        pdf = fitz.open(file_path)
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            text = page.get_text("text")
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
        if docs:
            logger.info("Loaded PDF using PyMuPDF")
            return docs
    except Exception as e:
        logger.warning(f"PyMuPDF loader failed: {e}")

    # 2. UnstructuredPDFLoader
    try:
        loader = UnstructuredPDFLoader(file_path)
        docs = loader.load()
        if docs and any(getattr(d, 'page_content', '').strip() for d in docs):
            logger.info("Loaded PDF using UnstructuredPDFLoader")
            return docs
    except Exception as e:
        logger.warning(f"UnstructuredPDFLoader failed: {e}")

    # 3. PDFPlumberLoader
    try:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        if docs and any(getattr(d, 'page_content', '').strip() for d in docs):
            logger.info("Loaded PDF using PDFPlumberLoader")
            return docs
    except Exception as e:
        logger.warning(f"PDFPlumberLoader failed: {e}")
        
    # 4. PyPDF2 fallback (covers many text-based PDFs)
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        pypdf_docs = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.debug(f"PyPDF2 page extract failed for page {i}: {e}")
                text = ""
            if text and text.strip():
                pypdf_docs.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
        if pypdf_docs:
            logger.info("Loaded PDF using PyPDF2")
            return pypdf_docs
    except Exception as e:
        logger.warning(f"PyPDF2 fallback failed: {e}")
        
    # 5. OCR Fallback (Simplified for API context)
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(file_path)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        if ocr_text.strip():
            logger.info("Loaded PDF using OCR")
            return [Document(page_content=ocr_text, metadata={"source": file_path})]
    except ImportError:
        logger.warning("OCR dependencies missing.")
    except Exception as e:
        logger.error(f"OCR failed: {e}")

    return []

def update_vector_db(new_docs: List[Document]):
    """Chunks documents and updates the global FAISS index."""
    global vector_store
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(new_docs)
    
    if not chunks:
        return

    if vector_store is None:
        # Create new index
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        # Add to existing index
        vector_store.add_documents(chunks)
    
    # Save to disk for persistence
    vector_store.save_local(VECTOR_DB_PATH)
    logger.info(f"Vector store updated with {len(chunks)} new chunks.")

# --- API Models ---
class QueryRequest(BaseModel):
    question: str
    model_name: str = "gemma3:1b"
    conversation_id: Optional[uuid.UUID] = None

# --- Management Endpoints ---
@app.get("/chat/getAllConversations", response_model=List[ConversationRead])
def get_all_conversations(session: Session = Depends(get_session)):
    """
    Fetches all conversations with their messages, sorted by UpdatedAt Descending.
    """
    # 1. Build the Query
    # We use 'options(selectinload(...))' to efficiently fetch the 'messages' list 
    # for each conversation in a single optimized query (avoids N+1 problem).
    statement = (
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updatedAt.desc())
    )
    
    # 2. Execute
    results = session.exec(statement).all()
    
    # 3. Sort Messages within each Conversation (Python-side sorting)
    # While we could sort in SQL, Python sorting for nested lists is often simpler 
    # when using ORMs unless you use complex loading strategies.
    for convo in results:
        convo.messages.sort(key=lambda m: m.messageIndex)
        # Attach file names to each message where available
        for m in convo.messages:
            try:
                rows = session.exec(
                    select(FileChunk.fileName).where(
                        FileChunk.conversationId == convo.conversationId,
                        FileChunk.messageIndex == m.messageIndex
                    )
                ).all()
                filenames = []
                for r in rows:
                    if isinstance(r, (list, tuple)):
                        val = r[0] if r else None
                    else:
                        val = r
                    if val is not None:
                        filenames.append(val)
                # Deduplicate while preserving order; set via object.__setattr__ to avoid Pydantic validation
                file_list = list(dict.fromkeys(filenames)) if filenames else []
                object.__setattr__(m, 'fileNames', file_list)
            except Exception:
                m.fileNames = []
        
    return results

# --- MANAGEMENT ENDPOINTS (Migrated from C#) ---

@app.get("/chat/getConversation/{conversation_id}", response_model=ConversationRead)
def get_conversation(conversation_id: uuid.UUID, session: Session = Depends(get_session)):
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Sort messages by index (C# did this)
    conversation.messages.sort(key=lambda m: m.messageIndex)
    # Attach file names per message if present in FileChunks
    for m in conversation.messages:
        try:
            rows = session.exec(
                select(FileChunk.fileName).where(
                    FileChunk.conversationId == conversation.conversationId,
                    FileChunk.messageIndex == m.messageIndex
                )
            ).all()
            filenames = []
            for r in rows:
                if isinstance(r, (list, tuple)):
                    val = r[0] if r else None
                else:
                    val = r
                if val is not None:
                    filenames.append(val)
            file_list = list(dict.fromkeys(filenames)) if filenames else []
            object.__setattr__(m, 'fileNames', file_list)
        except Exception:
            m.fileNames = []

    return conversation

@app.delete("/chat/deleteById/{conversation_id}")
def delete_conversation(conversation_id: uuid.UUID, session: Session = Depends(get_session)):
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    session.delete(conversation)
    session.commit()
    return {"message": "Conversation deleted."}

@app.delete("/chat/deleteAll")
def delete_all_conversations(session: Session = Depends(get_session)):
    # Efficiently delete all rows
    session.exec(delete(Message))
    session.exec(delete(Conversation))
    session.commit()
    return {"message": "All conversations deleted."}

@app.put("/chat/renameConversation/{conversation_id}")
def rename_conversation(conversation_id: uuid.UUID, req: RenameRequest, session: Session = Depends(get_session)):
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.conversationName = req.newTitle
    conversation.updatedAt = datetime.now(timezone.utc)
    session.add(conversation)
    session.commit()
    return {"message": "Conversation renamed."}

@app.put("/chat/Archive/{conversation_id}")
def archive_conversation(conversation_id: uuid.UUID, session: Session = Depends(get_session)):
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Toggle logic: 1 -> 0, 0 -> 1
    conversation.isArchived = 1 if conversation.isArchived == 0 else 0
    conversation.updatedAt = datetime.now(timezone.utc)
    session.add(conversation)
    session.commit()
    return {"conversationId": conversation_id, "isArchived": conversation.isArchived}

@app.put("/chat/pin/{conversation_id}")
def pin_conversation(conversation_id: uuid.UUID, session: Session = Depends(get_session)):
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Toggle logic
    conversation.isPinned = 1 if conversation.isPinned == 0 else 0
    conversation.updatedAt = datetime.now(timezone.utc)
    session.add(conversation)
    session.commit()
    return {"conversationId": conversation_id, "isPinned": conversation.isPinned}

# AI-endpoints

# Update the Helper Function to return chunks instead of just updating global state
def process_and_chunk_pdf(file_path: str) -> List[Document]:
    """Loads PDF and returns chunks (does NOT update vector DB directly)."""
    raw_docs = process_pdf(file_path)
    if not raw_docs:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(raw_docs)

@app.post("/ingest", summary="Upload PDF files to Knowledge Base")
async def ingest_files(
    files: List[UploadFile] = File(...), 
    # Use Form() to extract data from multipart/form-data
    conversation_id: Optional[uuid.UUID] = Form(None), 
    message_index: int = Form(0),
    session: Session = Depends(get_session)
):
    global vector_store
    processed_count = 0
    total_new_chunks = 0
    
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        chunks = process_and_chunk_pdf(file_location)
        
        if chunks:
            if vector_store is None:
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                vector_store.add_documents(chunks)
            
            for i, chunk in enumerate(chunks):
                db_chunk = FileChunk(
                    fileName=file.filename,
                    chunkIndex=i,
                    content=chunk.page_content,
                    # Store the ID and Index passed from Angular
                    conversationId=conversation_id,
                    messageIndex=message_index 
                )
                session.add(db_chunk)
            
            session.commit()
            processed_count += 1
            total_new_chunks += len(chunks)
            
    if processed_count == 0:
        return {"message": "No valid text extracted."}
        
    return {"message": f"Ingested {processed_count} files.", "chunks_added": total_new_chunks}

@app.post("/ask", response_model=ChatResponse)
def ask_question(req: QueryRequest, session: Session = Depends(get_session)):
    """
    1. Retrieval (RAG)
    2. Generation (Ollama)
    3. Persistence (Save to SQL DB)
    """
    # --- 1. RAG LOGIC ---
    global vector_store
    context_text = ""
    
    # Only try RAG if we have a vector store
    if vector_store:
        try:
            docs = vector_store.similarity_search(req.question, k=2)
            if docs:
                joined_docs = "\n\n".join([d.page_content for d in docs])
                context_text = textwrap.shorten(joined_docs, width=2000, placeholder=" ...")
        except Exception as e:
            logger.error(f"RAG Retrieval failed: {e}")
            # We continue even if RAG fails, just without context

    print(f"--- RAG Context Retrieved: {context_text}")
    # --- 2. PREPARE PROMPT ---
    if context_text:
        prompt = (
            f"CONTEXT:\n{context_text}\n\n"
            f"Question: {req.question}\n"
            "INSTRUCTION: You are a helpful assistant."
            "Use the provided context to answer the question only if the Context is relevant to the Question."
            "If not relevant then ignore the context and answer based on your own knowledge."
        )
    else:
        prompt = (
            f"Question: {req.question}\n"
            "INSTRUCTION: Answer using your general knowledge."
        )

    # --- 3. CALL LLM (Ollama) ---
    payload = {
        "model": req.model_name,
        "prompt": prompt,
        "temperature": 0.3,
        "stream": False,
        "keep_alive": -1
    }
    
    bot_answer = ""
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        bot_answer = response.json().get("response", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # --- 4. DATABASE PERSISTENCE (Business Logic) ---
    
    # A. Determine Conversation ID
    conversation = None
    if req.conversation_id:
        conversation = session.get(Conversation, req.conversation_id)
    
    if not conversation:
        conversation = Conversation(
            conversationId=req.conversation_id,
            conversationName=req.question[:50] + "..." if len(req.question) > 50 else req.question,
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
    else:
        conversation.updatedAt = datetime.now(timezone.utc)
        session.add(conversation)
    
    # B. Calculate Message Index
    # We need the max index to know where to append.
    from sqlmodel import func
    current_max_index = session.exec(
        select(func.max(Message.messageIndex)).where(Message.conversationId == conversation.conversationId)
    ).one()
    
    next_index = (current_max_index if current_max_index is not None else -1) + 1

    # C. Save User Message
    user_msg = Message(
        conversationId=conversation.conversationId,
        sender="User",
        messageText=req.question,
        messageIndex=next_index
    )
    session.add(user_msg)

    # D. Save Bot Message
    bot_msg = Message(
        conversationId=conversation.conversationId,
        sender="Bot",
        messageText=bot_answer,
        messageIndex=next_index + 1
    )
    session.add(bot_msg)
    
    session.commit()

    # --- 5. RETURN RESPONSE ---
    return {
        "conversationId": conversation.conversationId,
        "messageIndex": next_index + 1,
        "message": bot_answer
    }

@app.post("/chat/loadContext/{conversation_id}")
def load_context_from_db(conversation_id: uuid.UUID, session: Session = Depends(get_session)):
    """
    Hydrates the Vector DB with chunks stored in SQL Server for a specific conversation.
    """
    global vector_store

    # 1. Fetch chunks from SQL DB
    chunks = session.exec(
        select(FileChunk).where(FileChunk.conversationId == conversation_id)
    ).all()

    if not chunks:
        return {"message": "No documents found for this conversation.", "count": 0}

    # 2. Convert to LangChain Documents
    documents = []
    for chunk in chunks:
        # We reconstruct the Document object
        doc = Document(
            page_content=chunk.content,
            metadata={
                "source": chunk.fileName,
                "chunk_index": chunk.chunkIndex
            }
        )
        documents.append(doc)

    # 3. Load into FAISS (Resetting store vs Adding to store)
    # Strategy: Since we want ONLY this conversation's context, we force a new store.
    
    logger.info(f"Hydrating vector DB with {len(documents)} chunks from DB...")
    
    # Create a fresh vector store for this session
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return {"message": "Context loaded successfully", "count": len(documents)}


@app.post("/chat/clearContext")
def clear_context():
    """
    Clears the in-memory Vector DB.
    """
    global vector_store
    vector_store = None
    logger.info("Vector DB context cleared.")
    return {"message": "Context cleared"}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)