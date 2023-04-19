from absl.testing import absltest

from snowflake.ml.model import schema


class SchemaTest(absltest.TestCase):
    def test_1(self) -> None:
        s = schema.Schema(
            inputs=[
                schema.ColSpec(dtype=schema.DataType.FLOAT, name="c1"),
                schema.ColGroupSpec(
                    name="cg1",
                    specs=[
                        schema.ColSpec(
                            dtype=schema.DataType.FLOAT,
                            name="cc1",
                        ),
                        schema.ColSpec(
                            dtype=schema.DataType.FLOAT,
                            name="cc2",
                        ),
                    ],
                ),
                schema.ColSpec(dtype=schema.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[schema.ColSpec(name="output", dtype=schema.DataType.FLOAT)],
        )
        target = {
            "inputs": [
                {"type": "FLOAT", "name": "c1"},
                {
                    "column_group": {
                        "name": "cg1",
                        "specs": [{"type": "FLOAT", "name": "cc1"}, {"type": "FLOAT", "name": "cc2"}],
                    }
                },
                {"type": "FLOAT", "name": "c2", "shape": (-1,)},
            ],
            "outputs": [{"type": "FLOAT", "name": "output"}],
        }
        self.assertEqual(s.to_dict(), target)

    def test_2(self) -> None:
        s = schema.Schema(
            inputs=[
                schema.ColSpec(dtype=schema.DataType.FLOAT, name="c1"),
                schema.ColGroupSpec(
                    name="cg1",
                    specs=[
                        schema.ColSpec(
                            dtype=schema.DataType.FLOAT,
                            name="cc1",
                        ),
                        schema.ColSpec(
                            dtype=schema.DataType.FLOAT,
                            name="cc2",
                        ),
                    ],
                ),
                schema.ColSpec(dtype=schema.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[schema.ColSpec(name="output", dtype=schema.DataType.FLOAT)],
        )
        self.assertEqual(s, schema.Schema.from_dict(s.to_dict()))


if __name__ == "__main__":
    absltest.main()
