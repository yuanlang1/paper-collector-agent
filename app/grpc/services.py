from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import grpc
from google.protobuf.wrappers_pb2 import Int64Value

from app.database import SessionLocal
from app.history import chat_log, stream_events
from app.llm.streaming.utils import to_jsonable
from app.protos.agent.v1 import agent_pb2, agent_pb2_grpc
from app.runtime.agent_runtime import AgentRuntime
from app.services import internal_settings_service as settings_service


def _require_text(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _json_object(value: str, name: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _profile_view(profile: settings_service.LlmProfileData) -> agent_pb2.LlmProfileView:
    return agent_pb2.LlmProfileView(
        id=profile.id,
        name=profile.name,
        provider=profile.provider,
        base_url=profile.base_url,
        model=profile.model,
        key_set=profile.key_set,
        key_last4=profile.key_last4,
        enabled=profile.enabled,
        is_default=profile.is_default,
        is_small_model=profile.is_small_model,
        version=profile.version,
        created_at=profile.created_at.isoformat(),
        updated_at=profile.updated_at.isoformat(),
    )


def _limits_view(values: dict[str, dict[str, int]]) -> agent_pb2.SourceLimitsView:
    limits = values["limits"]
    page_sizes = values["page_sizes"]
    maximums = values["maximum_limits"]
    return agent_pb2.SourceLimitsView(
        arxiv=limits["arxiv"],
        dblp=limits["dblp"],
        google_scholar=limits["google_scholar"],
        arxiv_page_size=page_sizes["arxiv"],
        dblp_page_size=page_sizes["dblp"],
        google_scholar_page_size=page_sizes["google_scholar"],
        arxiv_maximum=maximums["arxiv"],
        dblp_maximum=maximums["dblp"],
        google_scholar_maximum=maximums["google_scholar"],
    )


class AgentGrpcService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def StartRun(self, request, context):
        try:
            user_id = _require_text(request.user_id, "user_id")
            conversation_id = _require_text(request.conversation_id, "conversation_id")
            run_id = _require_text(request.run_id, "run_id")
            message = _require_text(request.message, "message")
            profile_id = request.llm_profile_id.value if request.HasField("llm_profile_id") else None
            if profile_id is not None:
                _positive(profile_id, "llm_profile_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        db = SessionLocal()
        try:
            await self.runtime.chat_stream(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                llm_profile_id=profile_id,
                db=db,
            )
        except chat_log.RunAlreadyExistsError as exc:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except (LookupError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            await context.abort(grpc.StatusCode.INTERNAL, "Unable to start run")
        finally:
            db.close()
        return agent_pb2.RunReference(conversation_id=conversation_id, run_id=run_id, status="started")

    async def ResumeRun(self, request, context):
        try:
            user_id = _require_text(request.user_id, "user_id")
            conversation_id = _require_text(request.conversation_id, "conversation_id")
            run_id = _require_text(request.run_id, "run_id")
            action_id = _require_text(request.action_id, "action_id")
            decision = _require_text(request.decision, "decision")
            if decision not in {"approved", "rejected"}:
                raise ValueError("decision must be approved or rejected")
            payload: dict[str, Any] = {"decision": decision}
            for field, name in ((request.query_understanding_json, "query_understanding"), (request.search_tag_json, "search_tag")):
                value = _json_object(field, name)
                if value is not None:
                    payload[name] = value
            if request.comment:
                payload["comment"] = request.comment
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        scope = await self.runtime.history_db.run(chat_log.get_run_scope, run_id)
        if scope is None or scope.user_id != user_id or scope.conversation_id != conversation_id:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        db = SessionLocal()
        try:
            subscription = await self.runtime.resume_chat_stream(
                user_id=user_id,
                conversation_id=conversation_id,
                resume_payload=payload,
                requested_run_id=run_id,
                requested_action_id=action_id,
                db=db,
            )
        except chat_log.PendingActionConflictError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except LookupError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception:
            await context.abort(grpc.StatusCode.INTERNAL, "Unable to resume run")
        finally:
            db.close()
        return agent_pb2.RunReference(conversation_id=conversation_id, run_id=subscription.run_id, status="resumed")

    async def GetRun(self, request, context):
        try:
            user_id = _require_text(request.user_id, "user_id")
            conversation_id = _require_text(request.conversation_id, "conversation_id")
            run_id = _require_text(request.run_id, "run_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        scope = await self.runtime.history_db.run(chat_log.get_run_scope, run_id)
        if scope is None or scope.user_id != user_id or scope.conversation_id != conversation_id:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        status = await self.runtime.history_db.run(chat_log.get_run_status, user_id=user_id, run_id=run_id) or "running"
        latest_event = await self.runtime.history_db.run(
            stream_events.latest,
            user_id=user_id,
            run_id=run_id,
        )
        terminal = (
            latest_event.data
            if latest_event and latest_event.event_name in {"run_completed", "run_failed", "confirmation_required"}
            else {}
        )
        reply = str(terminal.get("reply") or "")
        return agent_pb2.RunView(
            conversation_id=conversation_id,
            run_id=run_id,
            status=status,
            latest_sequence=latest_event.sequence if latest_event else 0,
            reply=reply,
            result_json=json.dumps(to_jsonable(terminal), ensure_ascii=False),
        )

    async def SubscribeRunEvents(self, request, context) -> AsyncIterator[agent_pb2.RunEvent]:
        try:
            user_id = _require_text(request.user_id, "user_id")
            conversation_id = _require_text(request.conversation_id, "conversation_id")
            run_id = _require_text(request.run_id, "run_id")
            after_sequence = request.after_sequence.value if request.HasField("after_sequence") else None
            if after_sequence is not None and after_sequence < 0:
                raise ValueError("after_sequence must not be negative")
            subscription = await self.runtime.subscribe_stream(
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                last_event_sequence=after_sequence,
            )
        except LookupError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        async for event in subscription.events:
            yield agent_pb2.RunEvent(
                sequence=event.sequence,
                event_name=event.event_name,
                data_json=json.dumps(to_jsonable(event.data), ensure_ascii=False),
            )


class ConversationGrpcService(agent_pb2_grpc.ConversationServiceServicer):
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def ListConversations(self, request, context):
        try:
            user_id = _require_text(request.user_id, "user_id")
            limit = request.limit or 30
            if not 1 <= limit <= 100:
                raise ValueError("limit must be between 1 and 100")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        items = await self.runtime.history_db.run(chat_log.list_conversations, user_id=user_id, limit=limit)
        return agent_pb2.ListConversationsResponse(items=[agent_pb2.ConversationView(**item) for item in items])

    async def ListMessages(self, request, context):
        try:
            user_id = _require_text(request.user_id, "user_id")
            conversation_id = _require_text(request.conversation_id, "conversation_id")
            limit = request.limit or 100
            if not 1 <= limit <= 100:
                raise ValueError("limit must be between 1 and 100")
            before_id = request.before_id.value if request.HasField("before_id") else None
            if before_id is not None:
                _positive(before_id, "before_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        exists = await self.runtime.history_db.run(
            chat_log.conversation_exists,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not exists:
            await context.abort(grpc.StatusCode.NOT_FOUND, "conversation not found")
        items, next_before = await self.runtime.history_db.run(chat_log.list_messages, user_id=user_id, conversation_id=conversation_id, limit=limit, before_id=before_id)
        response = agent_pb2.ListMessagesResponse(conversation_id=conversation_id)
        for item in items:
            response.items.append(agent_pb2.ChatMessageView(
                id=item["id"], conversation_id=item["conversation_id"], run_id=item["run_id"], role=item["role"],
                content_json=json.dumps(item["content"], ensure_ascii=False), status=item["status"], source=item["source"],
                meta_json=json.dumps(item["meta"], ensure_ascii=False), created_at=item["created_at"],
            ))
        if next_before is not None:
            response.next_before_id.CopyFrom(Int64Value(value=next_before))
        return response

    async def DeleteConversation(self, request, context):
        try:
            user_id = _require_text(request.user_id, "user_id")
            conversation_id = _require_text(request.conversation_id, "conversation_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        exists = await self.runtime.history_db.run(
            chat_log.conversation_exists,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not exists:
            await context.abort(grpc.StatusCode.NOT_FOUND, "conversation not found")
        db = SessionLocal()
        try:
            await self.runtime.delete_conversation(user_id=user_id, conversation_id=conversation_id, db=db)
        except Exception:
            await context.abort(grpc.StatusCode.INTERNAL, "Unable to delete conversation")
        finally:
            db.close()
        return agent_pb2.DeleteConversationResponse(conversation_id=conversation_id, deleted=True)


class SettingsGrpcService(agent_pb2_grpc.SettingsServiceServicer):
    @staticmethod
    def _updates(request) -> dict[str, object]:
        updates: dict[str, object] = {}
        for field in ("name", "provider", "base_url", "model", "api_key", "enabled", "is_default", "is_small_model"):
            if request.HasField(field):
                updates[field] = getattr(request, field).value
        return updates

    async def GetSourceLimits(self, request, context):
        db = SessionLocal()
        try:
            return _limits_view(settings_service.get_limits(db))
        finally:
            db.close()

    async def UpdateSourceLimits(self, request, context):
        try:
            limits = {"arxiv": _positive(request.arxiv, "arxiv"), "dblp": _positive(request.dblp, "dblp"), "google_scholar": _positive(request.google_scholar, "google_scholar")}
            db = SessionLocal()
            try:
                return _limits_view(settings_service.update_limits(db, limits))
            finally:
                db.close()
        except settings_service.SettingsValidationError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    async def ListLlmProfiles(self, request, context):
        db = SessionLocal()
        try:
            return agent_pb2.ListLlmProfilesResponse(items=[_profile_view(item) for item in settings_service.list_profiles(db)])
        finally:
            db.close()

    async def CreateLlmProfile(self, request, context):
        db = SessionLocal()
        try:
            profile = settings_service.create_profile(
                db,
                name=request.name,
                provider=request.provider,
                base_url=request.base_url,
                model=request.model,
                api_key=request.api_key,
                enabled=request.enabled.value if request.HasField("enabled") else True,
                is_default=request.is_default.value if request.HasField("is_default") else False,
                is_small_model=request.is_small_model.value if request.HasField("is_small_model") else False,
            )
            return _profile_view(profile)
        except settings_service.SettingsConflictError as exc:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except settings_service.SettingsValidationError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        finally:
            db.close()

    async def UpdateLlmProfile(self, request, context):
        try:
            profile_id = _positive(request.profile_id, "profile_id")
            updates = self._updates(request)
            if not updates:
                raise ValueError("at least one update is required")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        db = SessionLocal()
        try:
            return _profile_view(settings_service.update_profile(db, profile_id, updates))
        except LookupError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "LLM profile was not found")
        except settings_service.SettingsConflictError as exc:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except settings_service.SettingsValidationError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        finally:
            db.close()

    async def SetDefaultLlmProfile(self, request, context):
        try:
            profile_id = _positive(request.profile_id, "profile_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        db = SessionLocal()
        try:
            return _profile_view(settings_service.set_default_profile(db, profile_id))
        except LookupError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "LLM profile was not found")
        except settings_service.SettingsValidationError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        finally:
            db.close()

    async def DeleteLlmProfile(self, request, context):
        try:
            profile_id = _positive(request.profile_id, "profile_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        db = SessionLocal()
        try:
            settings_service.delete_profile(db, profile_id)
        except LookupError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "LLM profile was not found")
        except settings_service.SettingsValidationError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        finally:
            db.close()
        return agent_pb2.DeleteLlmProfileResponse(id=profile_id, deleted=True)

    async def TestLlmProfile(self, request, context):
        try:
            profile_id = _positive(request.profile_id, "profile_id")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        db = SessionLocal()
        try:
            ok, model_count, message = await settings_service.test_profile(db, profile_id)
        except LookupError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "LLM profile was not found")
        finally:
            db.close()
        response = agent_pb2.TestLlmProfileResponse(ok=ok, message=message or "")
        if model_count is not None:
            response.model_count.value = model_count
        return response


def add_services(server: grpc.aio.Server, runtime: AgentRuntime) -> None:
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentGrpcService(runtime), server)
    agent_pb2_grpc.add_ConversationServiceServicer_to_server(ConversationGrpcService(runtime), server)
    agent_pb2_grpc.add_SettingsServiceServicer_to_server(SettingsGrpcService(), server)
