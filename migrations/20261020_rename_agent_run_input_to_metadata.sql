BEGIN;

ALTER TABLE agent_run RENAME COLUMN input TO metadata;
UPDATE agent_run SET metadata = metadata -> 'metadata';

COMMIT;
