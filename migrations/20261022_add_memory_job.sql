BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE public.memory_job (
    job_id VARCHAR(128) NOT NULL UNIQUE,
    kind VARCHAR(32) NOT NULL,
    user_id BIGINT NOT NULL,
    thread_id VARCHAR(128) NOT NULL REFERENCES public.agent_thread (thread_id) ON DELETE CASCADE,
    run_id VARCHAR(128) NOT NULL REFERENCES public.agent_run (run_id) ON DELETE CASCADE,
    checkpoint_id VARCHAR(128) NOT NULL,
    end_id VARCHAR(128) NOT NULL,
    source_thread_update_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    lease_expires_at TIMESTAMP WITH TIME ZONE,
    claim_token VARCHAR(128),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    id BIGSERIAL PRIMARY KEY,
    creator_id BIGINT NOT NULL DEFAULT 0,
    creation_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    modifier_id BIGINT NOT NULL DEFAULT 0,
    modification_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_memory_job_kind CHECK (kind IN ('userprofile', 'fact', 'episode')),
    CONSTRAINT ck_memory_job_status CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),
    CONSTRAINT ck_memory_job_attempts CHECK (attempts >= 0),
    CONSTRAINT ck_memory_job_episode_source CHECK (
        (kind = 'episode' AND source_thread_update_time IS NOT NULL) OR
        (kind <> 'episode' AND source_thread_update_time IS NULL)
    ),
    CONSTRAINT ck_memory_job_lease CHECK (
        (status = 'running' AND claim_token IS NOT NULL AND lease_expires_at IS NOT NULL) OR
        (status <> 'running' AND claim_token IS NULL AND lease_expires_at IS NULL)
    )
);

CREATE INDEX ix_memory_job_dispatch ON public.memory_job (status, available_at, id);
CREATE INDEX ix_memory_job_thread_kind ON public.memory_job (user_id, thread_id, kind, id);

CREATE UNIQUE INDEX uq_memory_job_run_kind ON public.memory_job (run_id, kind)
    WHERE kind IN ('userprofile', 'fact');
CREATE UNIQUE INDEX uq_memory_job_episode_source
    ON public.memory_job (thread_id, source_thread_update_time) WHERE kind = 'episode';
CREATE UNIQUE INDEX uq_memory_job_active_episode ON public.memory_job (thread_id)
    WHERE kind = 'episode' AND status IN ('pending', 'running');

COMMIT;
