BEGIN;

ALTER TABLE llm_setting ADD COLUMN user_id BIGINT NOT NULL;
CREATE INDEX ix_llm_setting_user_id ON llm_setting (user_id);

COMMIT;
