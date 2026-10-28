"""Request / response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1)
    user_level: str = "public"


class ChatResponse(BaseModel):
    answer: str
    latency: float
    level: str


class HistoryItem(BaseModel):
    role: str
    content: str
    timestamp: float
