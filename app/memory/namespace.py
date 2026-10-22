def profile_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", str(user_id), "profile")


def fact_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", str(user_id), "facts")


def episode_prefix(
    user_id: str,
    thread_id: str,
) -> tuple[str, ...]:
    return ("users", str(user_id), "episodes", str(thread_id))


def episode_namespace(
    user_id: str,
    thread_id: str,
    job_id: str,
) -> tuple[str, ...]:
    return (*episode_prefix(user_id, thread_id), str(job_id))
