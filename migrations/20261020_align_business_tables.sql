BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE
    public.agent_session, public.agent_message, public.agent_summary,
    public.agent_thread, public.agent_run, public.llm_setting,
    public.config_setting, public.memory_facts, public.memory_episodes,
    public.memory_consolidation_cursors
IN ACCESS EXCLUSIVE MODE;

-- This migration targets the inspected database whose business tables are empty.
DO $$
DECLARE
    table_name text;
    has_rows boolean;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'agent_session', 'agent_message', 'agent_summary', 'agent_thread',
        'agent_run', 'llm_setting', 'config_setting', 'memory_facts',
        'memory_episodes', 'memory_consolidation_cursors'
    ] LOOP
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM public.%I)', table_name)
            INTO has_rows;
        IF has_rows THEN
            RAISE EXCEPTION 'Table % is no longer empty; migration cancelled', table_name;
        END IF;
    END LOOP;
END $$;

ALTER TABLE public.agent_run RENAME COLUMN input TO metadata;

ALTER TABLE public.llm_setting ADD COLUMN user_id BIGINT NOT NULL;
CREATE INDEX ix_llm_setting_user_id ON public.llm_setting (user_id);

ALTER TABLE public.config_setting
    ALTER COLUMN creation_time TYPE TIMESTAMP WITH TIME ZONE
        USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT now(),
    ALTER COLUMN creation_time SET NOT NULL,
    ALTER COLUMN modification_time TYPE TIMESTAMP WITH TIME ZONE
        USING modification_time AT TIME ZONE 'UTC',
    ALTER COLUMN modification_time SET DEFAULT now(),
    ALTER COLUMN modification_time SET NOT NULL;

ALTER TABLE public.memory_facts RENAME COLUMN created_at TO creation_time;
ALTER TABLE public.memory_facts RENAME COLUMN updated_at TO modification_time;
ALTER TABLE public.memory_facts
    ALTER COLUMN id TYPE BIGINT,
    ADD COLUMN creator_id BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN modifier_id BIGINT NOT NULL DEFAULT 0;
ALTER SEQUENCE public.memory_facts_id_seq AS BIGINT;

ALTER TABLE public.memory_episodes RENAME COLUMN created_at TO creation_time;
ALTER TABLE public.memory_episodes RENAME COLUMN updated_at TO modification_time;
ALTER TABLE public.memory_episodes
    ALTER COLUMN id TYPE BIGINT,
    ALTER COLUMN source_assistant_message_id TYPE BIGINT,
    ADD COLUMN creator_id BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN modifier_id BIGINT NOT NULL DEFAULT 0;
ALTER SEQUENCE public.memory_episodes_id_seq AS BIGINT;

ALTER TABLE public.memory_consolidation_cursors
    RENAME COLUMN updated_at TO modification_time;
ALTER TABLE public.memory_consolidation_cursors
    DROP CONSTRAINT memory_consolidation_cursors_pkey,
    ADD COLUMN id BIGSERIAL PRIMARY KEY,
    ADD COLUMN creator_id BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN creation_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    ADD COLUMN modifier_id BIGINT NOT NULL DEFAULT 0,
    ALTER COLUMN last_assistant_message_id TYPE BIGINT,
    ADD CONSTRAINT uq_memory_cursors_user_conversation
        UNIQUE (user_id, conversation_id);

DROP TABLE public.agent_message;
DROP TABLE public.agent_summary;
DROP TABLE public.agent_session;

COMMIT;
