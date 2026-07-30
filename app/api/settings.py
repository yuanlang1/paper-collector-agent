from typing import Any, Dict, List, Optional, Tuple
import json
import re
from sqlalchemy.orm import Session
from app.models.system_setting import SystemSetting

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["settings"])


class DataSourceConfig(BaseModel):
    enabled: bool
    api_key: str
    engine: Optional[str] = None


class AgentConfig(BaseModel):
    proactive_enabled: bool = True
    heartbeat_interval: int = 60  # seconds


class LLMConnectionConfig(BaseModel):
    api_key: str = Field(default="", description="OpenAI-compatible API Key")
    base_url: str = Field(default="https://api.openai.com/v1", description="API Base URL")