from absl.testing import absltest

from snowflake.cortex import _candidate, _optimization_result


def _candidate_with(
    run_name: str, val_score: float | None, relative_cost: float | None, test_score: float | None = None
) -> _candidate.Candidate:
    return _candidate.Candidate(
        model=f"model-{run_name}",
        body=f"body-{run_name}",
        run_name=run_name,
        val_score=val_score,
        test_score=test_score,
        relative_cost=relative_cost,
    )


class OptimizationResultTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.frontier = [
            _candidate_with("ITER_1", val_score=0.8, relative_cost=0.5),
            _candidate_with("ITER_2", val_score=0.9, relative_cost=1.0),
            _candidate_with("ITER_3", val_score=0.7, relative_cost=0.2),
        ]
        self.rejected = _candidate_with("ITER_4", val_score=None, relative_cost=None)
        self.result = _optimization_result.OptimizationResult(
            self.frontier, baseline=self.frontier[0], candidates=[*self.frontier, self.rejected]
        )

    def test_exposes_frontier_baseline_and_candidates(self) -> None:
        self.assertEqual(self.result.frontier, tuple(self.frontier))
        self.assertIs(self.result.baseline, self.frontier[0])
        self.assertEqual(self.result.candidates, (*self.frontier, self.rejected))

    def test_candidates_default_to_empty(self) -> None:
        self.assertEqual(_optimization_result.OptimizationResult(self.frontier).candidates, ())

    def test_copies_frontier_into_immutable_tuple(self) -> None:
        self.frontier.clear()

        self.assertLen(self.result.frontier, 3)
        with self.assertRaises(AttributeError):
            self.result.frontier.append(self.result.frontier[0])  # type: ignore[attr-defined]

    def test_rejects_empty_frontier(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least one frontier candidate"):
            _optimization_result.OptimizationResult([])

    def test_rejects_frontier_or_baseline_without_val_score_or_cost(self) -> None:
        for frontier, baseline in (
            ([self.rejected], None),
            ([_candidate_with("ITER_5", val_score=0.5, relative_cost=None)], None),
            (self.frontier, self.rejected),
        ):
            with self.subTest(frontier=frontier, baseline=baseline):
                with self.assertRaisesRegex(ValueError, "require a val_score and a relative_cost"):
                    _optimization_result.OptimizationResult(frontier, baseline=baseline)

    def test_selectors_use_val_score_without_test_scores(self) -> None:
        self.assertEqual(self.result.selection_metric, "val_score")
        self.assertEqual(self.result.best_quality().run_name, "ITER_2")
        self.assertEqual(self.result.cheapest().run_name, "ITER_3")
        self.assertEqual(self.result.best_under_cost(0.6).run_name, "ITER_1")
        self.assertEqual(self.result.cheapest_above_quality(0.75).run_name, "ITER_1")

    def test_selectors_use_test_score_when_frontier_and_baseline_all_have_one(self) -> None:
        frontier = [
            _candidate_with("ITER_1", val_score=0.9, relative_cost=0.5, test_score=0.6),
            _candidate_with("ITER_2", val_score=0.7, relative_cost=1.0, test_score=0.8),
        ]
        baseline = _candidate_with("SEED", val_score=0.5, relative_cost=1.0, test_score=0.4)
        result = _optimization_result.OptimizationResult(frontier, baseline=baseline)

        self.assertEqual(result.selection_metric, "test_score")
        self.assertEqual(result.best_quality().run_name, "ITER_2")
        self.assertEqual(result.cheapest_above_quality(0.7).run_name, "ITER_2")

    def test_selectors_fall_back_to_val_score_when_baseline_lacks_test_score(self) -> None:
        frontier = [
            _candidate_with("ITER_1", val_score=0.9, relative_cost=0.5, test_score=0.6),
            _candidate_with("ITER_2", val_score=0.7, relative_cost=1.0, test_score=0.8),
        ]
        baseline = _candidate_with("SEED", val_score=0.5, relative_cost=1.0)
        result = _optimization_result.OptimizationResult(frontier, baseline=baseline)

        self.assertEqual(result.selection_metric, "val_score")
        self.assertEqual(result.best_quality().run_name, "ITER_1")

    def test_selectors_reject_empty_constraints(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative_cost <= 0.1"):
            self.result.best_under_cost(0.1)
        with self.assertRaisesRegex(ValueError, "val_score >= 0.95"):
            self.result.cheapest_above_quality(0.95)


if __name__ == "__main__":
    absltest.main()
