from app.runtime.session import Session


def get_agent_runtime():
    from app.runtime.agent_runtime import get_agent_runtime as factory

    return factory()


__all__ = ["Session", "get_agent_runtime"]
