from typing import List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

# 1. Message Schema (Child)
class MessageRead(BaseModel):
    sender: str
    messageText: str
    createdAt: datetime
    messageIndex: int

    class Config:
        orm_mode = True

# 2. Conversation Schema (Parent)
class ConversationRead(BaseModel):
    conversationId: UUID
    conversationName: str
    createdAt: datetime
    updatedAt: datetime
    isArchived: int
    isPinned: int
    # This field will carry the list of messages
    messages: List[MessageRead] = [] 

    class Config:
        # This ensures the database model maps correctly to this schema
        orm_mode = True
        # Keep compatibility hint for newer Pydantic: some setups expect from_attributes
        from_attributes = True

class RenameRequest(BaseModel):
    newTitle: str

class ChatResponse(BaseModel):
    conversationId: UUID
    messageIndex: int
    message: str