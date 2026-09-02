"""
系统设置模型
用于持久化存储运行时配置（如 API Key、模型选择等）
"""
from sqlalchemy import BigInteger, Column, DateTime, String, Text, text
from app.database import Base

class SystemSetting(Base):
    """系统设置表"""
    __tablename__ = "config_setting"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    key = Column(String(500), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=True)
    creator_id = Column(BigInteger, nullable=False, server_default=text("0"))
    creation_time = Column(DateTime, nullable=True, server_default=text("CURRENT_TIMESTAMP"))
    modifier_id = Column(BigInteger, nullable=False, server_default=text("0"))
    modification_time = Column(
        DateTime,
        nullable=True,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )

    def __repr__(self):
        return f"<SystemSetting(key='{self.key}', value='{self.value[:20]}...')>"
