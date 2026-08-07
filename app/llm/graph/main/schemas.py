from typing import Literal, TypeAlias


RunStatus: TypeAlias = Literal[
    "running",
    "waiting_confirmation",
    "completed",
    "failed",
    "blocked",
]
