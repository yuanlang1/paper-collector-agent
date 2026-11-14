BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE public.agent_thread_summary (
    user_id BIGINT NOT NULL,
    thread_id VARCHAR(128) NOT NULL REFERENCES public.agent_thread (thread_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    end_message_id TEXT NOT NULL,
    version BIGINT NOT NULL DEFAULT 1,
    id BIGSERIAL PRIMARY KEY,
    creator_id BIGINT NOT NULL DEFAULT 0,
    creation_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    modifier_id BIGINT NOT NULL DEFAULT 0,
    modification_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_thread_summary_owner UNIQUE (user_id, thread_id)
);

COMMIT;
