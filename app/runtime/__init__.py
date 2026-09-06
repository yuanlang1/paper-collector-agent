from app.runtime.session import Session
from app.runtime.agent_runtime import get_agent_runtime as factory

def get_agent_runtime():

    return factory()


__all__ = ["Session", "get_agent_runtime"]
