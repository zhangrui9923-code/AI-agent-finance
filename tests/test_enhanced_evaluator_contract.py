import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.enhanced_evaluation.enhanced_evaluator import (  # noqa: E402
    ComprehensiveEvaluator,
    DimensionScore,
    EvaluationReport,
    MetricScore,
    ScoreLevel,
)


class EnhancedEvaluatorContractTest(unittest.TestCase):
    def test_comprehensive_evaluator_exposes_llm_switch(self):
        evaluator = ComprehensiveEvaluator(use_llm=True)

        self.assertTrue(evaluator.rag_evaluator.use_llm)

    def test_report_serializes_all_five_dimensions(self):
        metric = MetricScore(
            name="Contract Metric",
            value=0.8,
            level=ScoreLevel.GOOD,
        )
        dimension = DimensionScore(
            dimension="Contract Dimension",
            metrics=[metric],
            weighted_score=0.8,
            level=ScoreLevel.GOOD,
        )
        report = EvaluationReport(
            evaluation_id="eval_contract",
            timestamp="2026-05-14T00:00:00",
            query="测试",
            answer="测试答案",
            rag_quality=dimension,
            agent_quality=dimension,
            output_quality=dimension,
            system_performance=dimension,
            user_satisfaction=dimension,
        )

        serialized = report.to_dict()

        self.assertIn("rag_quality", serialized)
        self.assertIn("agent_quality", serialized)
        self.assertIn("output_quality", serialized)
        self.assertIn("system_performance", serialized)
        self.assertIn("user_satisfaction", serialized)

    def test_module_doc_no_longer_marks_dimensions_as_unimplemented(self):
        source = (ROOT / "src/enhanced_evaluation/enhanced_evaluator.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("System Performance 和 User Satisfaction 维度为规划中", source)
        self.assertNotIn("暂未实现", source)


if __name__ == "__main__":
    unittest.main()
