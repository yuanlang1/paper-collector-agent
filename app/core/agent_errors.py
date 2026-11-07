from openai import APIStatusError


class AgentOutputError(Exception):
    status = "failed"
    reason = "Agent 执行失败"

    def __init__(self) -> None:
        super().__init__(self.reason)


class InsufficientBalanceError(AgentOutputError):
    reason = "模型服务余额不足，请充值或更换可用的模型服务"


class ModelAPIError(AgentOutputError):
    reason = "模型服务请求失败，请稍后重试"


def to_agent_output_error(exc: Exception) -> AgentOutputError:
    if isinstance(exc, AgentOutputError):
        return exc

    if isinstance(exc, APIStatusError):
        if (
            exc.status_code == 402
            and "insufficient balance" in str(exc).lower()
        ):
            return InsufficientBalanceError()

        return ModelAPIError()

    return AgentOutputError()