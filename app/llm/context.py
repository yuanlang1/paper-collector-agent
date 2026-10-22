from dataclasses import dataclass


@dataclass(kw_only=True)
class BaseAgentContext:
    thread_id: str
    run_id: str
    user_id: str
    auto_threshold: float = 0
    context_window_tokens: int | None = None
    keep_last_n_turns: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.auto_threshold <= 1:
            raise ValueError("auto_threshold 必须在 0 到 1 之间")
        if type(self.keep_last_n_turns) is not int or self.keep_last_n_turns < 1:
            raise ValueError("keep_last_n_turns 必须是正整数")
        if self.context_window_tokens is not None and (
            type(self.context_window_tokens) is not int or self.context_window_tokens < 1
        ):
            raise ValueError("context_window_tokens 必须是正整数")


@dataclass(kw_only=True)
class AgentRunContext(BaseAgentContext):
    auto_threshold: float = 0.8
