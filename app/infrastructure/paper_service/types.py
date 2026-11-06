from enum import IntEnum


class TaskState(IntEnum):
    """Search status codes; paper-service tracks RAG status separately."""
    SEARCH_PENDING = 0
    SEARCH_RUNNING = 1
    SEARCH_COMPLETED = 2
    SEARCH_FAILED = 3
    CANCELLED = 4
