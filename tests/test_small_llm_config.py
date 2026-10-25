import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import text

from app.common.exceptions import NotFoundException
from app.db.repositories import llm_setting, system_setting
from app.services import internal_settings_service, llm_setting_service
from tests.test_postgres_models import PostgresDatabaseTestCase


class SmallLlmRepositoryTests(PostgresDatabaseTestCase):
    async def test_small_model_selection_is_owned_flagged_and_latest(self):
        async with self.factory.begin() as db:
            for user_id, model, small in (
                (11, "small-old", True),
                (11, "small-new", True),
                (11, "regular-newer", False),
                (22, "other-user-newest", True),
            ):
                await llm_setting.create(
                    db, user_id=user_id, provider="openai",
                    base_url="https://example.test", model=model,
                    api_key="test-key", is_small_model=small,
                )
            selected = await llm_setting.get_small_model(db, user_id=11)
            self.assertIsInstance(selected, llm_setting.LlmSettingRow)
            self.assertEqual(selected.model, "small-new")
            rows = await llm_setting.list_settings(db, user_id=11, is_small_model=True)
            self.assertEqual([row.model for row in rows], ["small-new", "small-old"])
            self.assertIsNone(await llm_setting.get_small_model(db, user_id=33))

    async def test_regular_model_is_not_a_small_model_fallback(self):
        async with self.factory.begin() as db:
            await llm_setting.create(
                db, user_id=11, provider="openai", base_url="https://example.test",
                model="regular", api_key="test-key",
            )
            self.assertIsNone(await llm_setting.get_small_model(db, user_id=11))

    async def test_native_update_delete_preserve_owner_and_noop_timestamp(self):
        fields = {
            "provider": "openai", "base_url": "https://example.test",
            "model": "small", "api_key": "test-key", "is_small_model": True,
        }
        async with self.factory.begin() as db:
            setting = await llm_setting.create(db, user_id=11, **fields)
            setting_id = setting.id
            original_time = await db.scalar(text("""
                UPDATE llm_setting
                SET modification_time = '2000-01-01T00:00:00Z'
                WHERE id = :id
                RETURNING modification_time
            """), {"id": setting_id})

        async with self.factory.begin() as db:
            unchanged = await llm_setting.update(
                db, setting_id, user_id=11, **fields,
            )
            self.assertEqual(unchanged.modification_time, original_time)
            with self.assertRaisesRegex(ValueError, "not found"):
                await llm_setting.update(db, setting_id, user_id=22, **fields)
            self.assertFalse(await llm_setting.delete(db, setting_id, user_id=22))
            changed = await llm_setting.update(
                db, setting_id, user_id=11, **{**fields, "model": "changed"},
            )
            self.assertEqual(changed.model, "changed")
            self.assertGreater(changed.modification_time, original_time)
            self.assertTrue(await llm_setting.delete(db, setting_id, user_id=11))
            self.assertFalse(await llm_setting.delete(db, setting_id, user_id=11))

    async def test_system_upsert_returns_value_and_preserves_noop_timestamp(self):
        async with self.factory.begin() as db:
            self.assertIsNone(await system_setting.get(db, "key"))
            await system_setting.set_value(db, "key", "first")
            original_time = await db.scalar(text("""
                UPDATE config_setting
                SET modification_time = '2000-01-01T00:00:00Z'
                WHERE key = 'key'
                RETURNING modification_time
            """))

        async with self.factory.begin() as db:
            await system_setting.set_value(db, "key", "first")
            self.assertEqual(await system_setting.get(db, "key"), "first")
            unchanged = await db.scalar(text(
                "SELECT modification_time FROM config_setting WHERE key = 'key'"
            ))
            self.assertEqual(unchanged, original_time)
            await system_setting.set_value(db, "key", "second")
            self.assertEqual(await system_setting.get(db, "key"), "second")
            changed = await db.scalar(text(
                "SELECT modification_time FROM config_setting WHERE key = 'key'"
            ))
            self.assertGreater(changed, original_time)


class SmallLlmConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_existing_config_inside_short_transaction(self):
        db = MagicMock()
        setting = SimpleNamespace(
            provider="openai", base_url="https://example.test",
            model="small", api_key="test-key",
        )
        with patch.object(llm_setting, "get_small_model", new=AsyncMock(return_value=setting)) as get:
            config = await llm_setting_service.get_small_llm_config(db, user_id=11)
        get.assert_awaited_once_with(db, user_id=11)
        self.assertEqual(config, llm_setting_service.LlmConfig(
            "openai", "https://example.test", "small", "test-key",
        ))
        db.begin.return_value.__aenter__.assert_awaited_once()
        db.begin.return_value.__aexit__.assert_awaited_once()

    async def test_limits_update_uses_saved_value_without_readback(self):
        db = MagicMock()
        limits = {"arxiv": 2, "dblp": 3, "google_scholar": 4}
        with patch.object(
            internal_settings_service, "save_source_limits",
            new=AsyncMock(return_value=limits),
        ) as save, patch.object(
            internal_settings_service, "get_source_limits", new=AsyncMock(),
        ) as read:
            result = await internal_settings_service.update_limits(db, limits)
        self.assertEqual(result["limits"], limits)
        save.assert_awaited_once_with(db, limits)
        read.assert_not_awaited()

    async def test_missing_small_model_raises_not_found(self):
        db = MagicMock()
        with patch.object(llm_setting, "get_small_model", new=AsyncMock(return_value=None)):
            with self.assertRaisesRegex(NotFoundException, "小模型配置不存在"):
                await llm_setting_service.get_small_llm_config(db, user_id=11)
        db.begin.return_value.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
