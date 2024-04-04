from absl.testing import absltest

from snowflake.ml._internal.human_readable_id import hrid_generator, hrid_generator_base
from snowflake.ml._internal.utils import sql_identifier


class HRIDTest(hrid_generator_base.HRIDBase):
    def __id_generator__(self) -> int:
        return 1

    __hrid_structure__ = ("alphabet", "number")
    __hrid_words__ = dict(
        number=tuple(str(x) for x in range(1, 3)),
        alphabet=("a", "b"),
    )


class HRIDGeneratorBaseTest(absltest.TestCase):
    def setUp(self) -> None:
        self.hg = HRIDTest()

    def test_id_to_hrid(self) -> None:
        self.assertEqual(self.hg.id_to_hrid(0), "a_1")
        self.assertEqual(self.hg.id_to_hrid(1), "a_2")
        self.assertEqual(self.hg.id_to_hrid(2), "b_1")
        self.assertEqual(self.hg.id_to_hrid(3), "b_2")

    def test_hrid_to_id(self) -> None:
        self.assertEqual(self.hg.hrid_to_id("a_1"), 0)
        self.assertEqual(self.hg.hrid_to_id("a_2"), 1)
        self.assertEqual(self.hg.hrid_to_id("b_1"), 2)
        self.assertEqual(self.hg.hrid_to_id("b_2"), 3)

    def test_generate(self) -> None:
        self.assertTupleEqual(self.hg.generate(), (1, "a_2"))
        # Assert re-entry
        self.assertTupleEqual(self.hg.generate(), (1, "a_2"))


class HRIDGenerator16Test(absltest.TestCase):
    def test_generator(self) -> None:
        hg = hrid_generator.HRID16()
        id1 = sql_identifier.SqlIdentifier(hg.generate()[1])
        self.assertNotEqual(id1, hg.generate()[1])


if __name__ == "__main__":
    absltest.main()
