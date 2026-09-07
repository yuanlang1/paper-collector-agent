# Paper search 前端可视化接入

前端通过 `POST /api/agent/chat/stream` 消费 SSE。task-service 只保存
`SEARCH_PENDING/SEARCH_RUNNING/SEARCH_COMPLETED/SEARCH_FAILED/`
`SEARCH_PARTIAL_COMPLETED` 等粗粒度状态；论文检索的
实时节点、阶段、计数和警告由 `subagent_progress` 事件提供。

## 发起检索

```http
POST /api/agent/chat/stream
Content-Type: application/json
Accept: text/event-stream

{
  "message": "检索 RAG evaluation 相关论文",
  "conversation_id": "conv-123",
  "subagent": {
    "name": "paper_search_agent",
    "constraints": {
      "year_from": 2023,
      "year_to": 2026,
      "sources": ["arXiv", "DBLP"]
    }
  }
}
```

由于接口使用 POST，浏览器原生 `EventSource` 不能直接使用。前端可以使用
支持 POST 的 SSE 客户端，或者基于 `fetch()` 的 `ReadableStream` 解析 SSE。

## 可视化事件

```text
run_started
decision
confirmation_required
subagent_progress
action_result
message
run_completed | run_failed
```

每条 SSE 的 `data` 都包含统一信封：

```json
{
  "event_id": "run_xxx:12",
  "sequence": 12,
  "event": "subagent_progress",
  "conversation_id": "conv-123",
  "run_id": "run_xxx",
  "timestamp": "2026-07-29T08:00:00+00:00",
  "data": {
    "workflow": "paper_search"
  }
}
```

`subagent_progress` 是论文检索时间线的主要数据源，其信封中的 `data` 为：

```json
{
  "workflow": "paper_search",
  "subagent": "paper_search_agent",
  "delegation_id": "action_xxx",
  "task_id": 42,
  "node": "search_arxiv",
  "phase": "search",
  "phase_label": "多来源检索",
  "phase_index": 3,
  "phase_count": 8,
  "progress_percent": 38,
  "terminal": false,
  "stage": "normalizing",
  "status": "running",
  "progress": {
    "task_created": 1,
    "discovered": 25
  },
  "source_stats": {},
  "supplemental_search_round": 0,
  "warnings": [],
  "error": null,
  "task_status_update_error": null,
  "pdf_cleanup_error": null,
  "degraded": false,
  "remote_task_state": "SEARCH_RUNNING"
}
```

同一个 `checkpoint_namespace` 的事件已经是后端累积后的子图快照。前端可以直接
使用最新事件替换对应检索卡片，不需要自行拼接各节点的增量字段。

## 推荐的前端状态

```ts
type PaperSearchView = {
  delegationId: string
  taskId: number | null
  phase: string
  phaseLabel: string
  phaseIndex: number
  phaseCount: number
  progressPercent: number
  node: string
  stage: string | null
  status: string | null
  progress: Record<string, number>
  sourceStats: Record<string, unknown>
  supplementalSearchRound: number
  warnings: string[]
  error: string | null
  taskStatusUpdateError: string | null
  pdfCleanupError: string | null
  degraded: boolean
  remoteTaskState: string | null
  terminal: boolean
}
```

以 `delegation_id` 作为检索卡片主键；它来自根图 `dispatch` 输出的
`active_tool_call.id`。主图通过通用 `subagent` 节点执行已注册子图，后端会把
该调用 ID 绑定到实际 checkpoint namespace；因此 `timeline_step`、
`subagent_progress` 与 `action_result` 可关联到同一个调用。

论文检索的 `workflow`、阶段和节点映射由论文检索子图的运行时注册项提供。新增
其他子图时，应在其自身注册项定义对应阶段，不应在主图或 SSE 分发代码新增名称分支。

## 页面呈现建议

固定展示八个阶段：

1. 准备检索
2. 生成检索计划
3. 多来源检索
4. 清洗与质量审核
5. 论文信息补充
6. 论文推荐
7. 保存与清理
8. 同步任务状态

收到 `task_id` 后展示远程任务编号。收到 `task_status_update_error` 时，应显示
“检索结果已生成，但远程任务状态同步失败”，不能把它展示为完全成功。
`terminal` 仅在 `finalize_result` 节点为 `true`；`persist` 返回终态结果时仍需
等待 PDF 清理和远程任务状态同步。

`run_completed` 表示主图已完成，不代表远程 task-service 一定同步成功；最终仍应
检查最近一次 `subagent_progress.task_status_update_error` 和
`action_result.status`。

## TypeScript 消费示例

以下示例使用支持 POST 的 SSE 客户端。关键点是读取统一信封的 `data` 字段：

```ts
import { fetchEventSource } from "@microsoft/fetch-event-source"

const searches = new Map<string, PaperSearchView>()

await fetchEventSource("/api/agent/chat/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  },
  body: JSON.stringify({
    message: "检索 RAG evaluation 相关论文",
    conversation_id: "conv-123",
    subagent: {
      name: "paper_search_agent",
      constraints: {},
    },
  }),
  onmessage(message) {
    const envelope = JSON.parse(message.data)

    if (envelope.event === "subagent_progress") {
      const progress = envelope.data
      const key =
        progress.delegation_id ?? progress.child_thread_id

      searches.set(key, {
        delegationId: progress.delegation_id,
        taskId: progress.task_id,
        phase: progress.phase,
        phaseLabel: progress.phase_label,
        phaseIndex: progress.phase_index,
        phaseCount: progress.phase_count,
        progressPercent: progress.progress_percent,
        node: progress.node,
        stage: progress.stage,
        status: progress.status,
        progress: progress.progress,
        sourceStats: progress.source_stats,
        supplementalSearchRound:
          progress.supplemental_search_round,
        warnings: progress.warnings,
        error: progress.error,
        taskStatusUpdateError:
          progress.task_status_update_error,
        pdfCleanupError: progress.pdf_cleanup_error,
        degraded: progress.degraded,
        remoteTaskState: progress.remote_task_state,
        terminal: progress.terminal,
      })
    }
  },
})
```
