from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime, timezone
import uuid

# Base class to handle shared config
class ConversationBase(SQLModel):
    conversationName: str = Field(max_length=200)
    isArchived: int = Field(default=0)
    isPinned: int = Field(default=0)

class Conversation(ConversationBase, table=True):
    __tablename__ = "Conversations"

    conversationId: Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, 
        primary_key=True, 
        nullable=False
    )
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationship to messages
    messages: List["Message"] = Relationship(back_populates="conversation", sa_relationship_kwargs={"cascade": "all, delete"})
    # Relationship to file chunks (for documents linked to this conversation)
    file_chunks: List["FileChunk"] = Relationship(back_populates="conversation", sa_relationship_kwargs={"cascade": "all, delete"})

class MessageBase(SQLModel):
    sender: str = Field(max_length=50)
    messageText: str
    messageIndex: int

class Message(MessageBase, table=True):
    __tablename__ = "Messages"

    messageId: Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, 
        primary_key=True, 
        nullable=False
    )
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Foreign key
    conversationId: uuid.UUID = Field(foreign_key="Conversations.conversationId") # Update FK reference too
    # Relationship back to conversation
    conversation: Optional[Conversation] = Relationship(back_populates="messages")

class FileChunk(SQLModel, table=True):
    __tablename__ = "FileChunks" 
    
    # We explicitly map Python fields to SQL Server Column Names
    id: Optional[int] = Field(default=None, primary_key=True) # Maps to 'Id' automatically usually, but let's be safe
    fileName: str = Field(sa_column_kwargs={"name": "FileName"})
    chunkIndex: int = Field(sa_column_kwargs={"name": "ChunkIndex"})
    content: str = Field(sa_column_kwargs={"name": "Content"})
    conversationId: Optional[uuid.UUID] = Field(default=None, sa_column_kwargs={"name": "ConversationId"}, foreign_key="Conversations.conversationId")
    messageIndex: Optional[int] = Field(default=None, sa_column_kwargs={"name": "MessageIndex"})
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column_kwargs={"name": "CreatedAt"})
    # Relationship back to conversation to enable ORM cascade deletes
    conversation: Optional[Conversation] = Relationship(back_populates="file_chunks")
    class Config:
        populate_by_name = True