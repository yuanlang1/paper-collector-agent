# Agent SSE 事件

前端通过 `POST /api/agent/chat/stream` 和
`POST /api/agent/chat/resume/stream` 消费 SSE。由于接口使用 POST，前端应使用
支持 POST 的 SSE 客户端或基于 `fetch()` 的 `ReadableStream` 解析响应。

每条响应使用标准 SSE 事件名，`data` 是统一 JSON 信封：

```text
event: subagent_progress
data: {"event_id":"run_xxx:12","sequence":12,"event":"subagent_progress",...}
```

```json
{
  "event_id": "run_xxx:12",
  "sequence": 12,
  "event": "subagent_progress",
  "conversation_id": "conv-123",
  "run_id": "run_xxx",
  "timestamp": "2026-09-09T08:00:00+00:00",
  "data": {
    "source": "subagent",
    "delegation_id": "call-42",
    "workflow": "task_indexing",
    "progress": 47,
    "message": "知识库索引：47/100 篇（47%）"
  }
}
```

`action_id` 和 `delegation_id` 是前端关联工具、子图和时间线的稳定键。SSE 不暴露
LangGraph 的 `namespace`、`checkpoint_namespace` 或派生的子图线程标识。

## 事件

| 事件 | 用途 |
| --- | --- |
| `run_started`、`run_completed`、`run_failed` | 整个请求的生命周期。 |
| `message`、`reasoning_delta` | 助手回复和推理增量。 |
| `action_started`、`action_result` | 所有工具与子图共享的动作生命周期。 |
| `tool_*` | 工具执行事件。 |
| `subagent_*` | 子图的启动、真实业务进度、完成或失败。 |
| `timeline_step` | 子图中可见阶段的开始、完成或失败。 |
| `confirmation_required` | 需要用户确认后才能继续。 |

`timeline_step` 描述当前子图执行到哪一步；`subagent_progress` 只在存在真实业务进度时
发送。当前任务索引子图会转发 RAG worker 的进度；论文检索和文献综述主要发送
`timeline_step`。

前端以 `delegation_id` 建立子图卡片，以 `step_id` 更新其时间线。最终状态以
`action_result.status` 为准。子图完成或失败事件中的嵌套 `data` 会持久化为
`meta.card.subagents[].result`，用于会话刷新后的最终诊断展示。
