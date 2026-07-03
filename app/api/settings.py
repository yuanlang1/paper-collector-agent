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


def _get_setting(db: Session, key: str, default: Any = None) -> Any:
    """从数据库读取设置，如果不存在则返回默认值"""
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting and setting.value:
        try:
            return json.loads(setting.value)
        except json.JSONDecodeError:
            return setting.value
    return default

def get_custom_system_prompt(db: Session) -> str:
    """供其他模块调用：获取用户自定义的系统提示词"""
    return _get_setting(db, "agent_system_prompt", "") or ""