import unittest

from app.llm.graph.workflows.paper_search.nodes.generate_queries import (
    BuildSourceQueryPlanNode,
    SOURCE_QUERY_PLAN_SYSTEM_PROMPT,
    SourceQueryPlansOutput,
)


class _StructuredModel:
    async def ainvoke(self, _messages):
        return SourceQueryPlansOutput.model_validate(
            {
                "plans": [
                    {
                        "source": "arXiv",
                        "display_query": "RAG evaluation benchmark",
                        "reasoning": "根据主题和 benchmark 关键词生成。",
                        "arguments": {
                            "query": "RAG evaluation benchmark",
                            "search_type": "topic",
                        },
                    }
                ]
            }
        )


class BuildSourceQueryPlanNodeTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_describes_json_output(self):
        self.assertIn("最小 JSON 输出示例", SOURCE_QUERY_PLAN_SYSTEM_PROMPT)
        self.assertIn("plans[].arguments", SOURCE_QUERY_PLAN_SYSTEM_PROMPT)

    async def test_accepts_json_output(self):
        node = BuildSourceQueryPlanNode(
            model=_StructuredModel(),
            pagination_settings={
                "arxiv_max_pages": 2,
                "arxiv_page_size": 10,
                "arxiv_total_limit": 20,
                "dblp_max_pages": 2,
                "dblp_page_size": 10,
                "dblp_total_limit": 20,
                "google_max_pages": 2,
                "google_page_size": 10,
                "google_total_limit": 20,
            },
        )

        result = await node(
            {
                "query_understanding": {
                    "topic": "RAG evaluation",
                    "intent": "benchmark",
                    "keywords": ["RAG", "evaluation"],
                    "reasoning": "用户需要 RAG 评测论文。",
                },
                "search_tag": {
                    "sourceTag": ["arXiv"],
                },
            }
        )

        self.assertEqual(result["stage"], "searching")
        self.assertEqual(result["source_query_plans"][0]["source"], "arXiv")
        self.assertEqual(
            result["source_query_plans"][0]["arguments"]["max_results"],
            10,
        )


if __name__ == "__main__":
    unittest.main()
