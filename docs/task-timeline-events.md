# Task timeline SSE events

`timeline_step` drives the front-end task list. It is distinct from
`subagent_progress`: the latter is a post-node state snapshot, while the former
is emitted when a visible task starts, completes, or fails.

```json
{
  "event": "timeline_step",
  "data": {
    "workflow": "paper_search",
    "delegation_id": "action_xxx",
    "child_thread_id": "conv_xxx:paper_search_agent:action_xxx",
    "step_id": "paper_search:multi_source_search:2",
    "step_key": "multi_source_search",
    "label": "多来源检索（第 2 轮）",
    "state": "started",
    "iteration": 2,
    "error": null
  }
}
```

The client groups task lists by `delegation_id` (falling back to
`child_thread_id`), appends a task only for `started`, then updates that item
by `step_id` for `completed` or `failed`. It must not pre-create future tasks.

For `paper_search`, repeated search, review, and supplemental-planning tasks
use `supplemental_search_round`. For `task_review`, repeated claim, evidence,
section, assembly, and reflection tasks use `reflection_round`. Re-entering a
task therefore creates a different `step_id` and preserves the execution
history.
