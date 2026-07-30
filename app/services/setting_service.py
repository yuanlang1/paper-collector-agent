import json
from typing import Any
from sqlalchemy.orm import Session
from app.models.system_setting import SystemSetting


def get_setting(db: Session, key: str, default: Any = None) -> Any:
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting and setting.value:
        try:
            return json.loads(setting.value)
        except json.JSONDecodeError:
            return setting.value
    return default


def get_custom_system_prompt(db: Session) -> str:
    return get_setting(db, "agent_system_prompt", "") or ""