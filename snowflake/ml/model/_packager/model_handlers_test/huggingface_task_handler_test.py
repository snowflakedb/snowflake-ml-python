from absl.testing import absltest

from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler


class FilterNoneKwargsTest(absltest.TestCase):
    def test_empty_dict(self) -> None:
        self.assertEqual(_task_handler._filter_none_kwargs({}), {})

    def test_all_none(self) -> None:
        self.assertEqual(_task_handler._filter_none_kwargs({"a": None, "b": None}), {})

    def test_no_none(self) -> None:
        self.assertEqual(_task_handler._filter_none_kwargs({"a": 1, "b": "x"}), {"a": 1, "b": "x"})

    def test_mixed_scalars(self) -> None:
        self.assertEqual(
            _task_handler._filter_none_kwargs({"temperature": 0.7, "top_k": None, "do_sample": True}),
            {"temperature": 0.7, "do_sample": True},
        )

    def test_nested_dict_all_none(self) -> None:
        self.assertEqual(
            _task_handler._filter_none_kwargs({"watermarking_config": {"bias": None, "greenlist_ratio": None}}),
            {},
        )

    def test_nested_dict_partial_none(self) -> None:
        self.assertEqual(
            _task_handler._filter_none_kwargs({"watermarking_config": {"bias": 2.0, "greenlist_ratio": None}}),
            {"watermarking_config": {"bias": 2.0}},
        )

    def test_nested_dict_no_none(self) -> None:
        self.assertEqual(
            _task_handler._filter_none_kwargs({"watermarking_config": {"bias": 2.0, "greenlist_ratio": 0.5}}),
            {"watermarking_config": {"bias": 2.0, "greenlist_ratio": 0.5}},
        )

    def test_deeply_nested(self) -> None:
        self.assertEqual(
            _task_handler._filter_none_kwargs({"outer": {"middle": {"inner": None}}}),
            {},
        )
        self.assertEqual(
            _task_handler._filter_none_kwargs({"outer": {"middle": {"inner": 42}, "other": None}}),
            {"outer": {"middle": {"inner": 42}}},
        )

    def test_mixed_scalars_and_dicts(self) -> None:
        self.assertEqual(
            _task_handler._filter_none_kwargs(
                {
                    "temperature": 0.7,
                    "top_k": None,
                    "watermarking_config": {"bias": None, "greenlist_ratio": None},
                }
            ),
            {"temperature": 0.7},
        )


if __name__ == "__main__":
    absltest.main()
