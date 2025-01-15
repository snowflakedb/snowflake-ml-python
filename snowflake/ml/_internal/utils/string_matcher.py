import re
from typing import Dict, List

import sqlparse
from absl.logging import logging

from snowflake.ml._internal.utils.formatting import unwrap

logger = logging.getLogger(__name__)


class StringMatcherIgnoreWhitespace:
    """Matcher that removes all whitespace from strings before comparison."""

    def __init__(self, expected: str) -> None:
        self._space_remover = re.compile(r"\s+")
        self._punc_fixer = re.compile(r"\s+([.,!?:;])")
        self._expected = expected

    def _normalize(self, s: str) -> str:
        # Remove leading and trailing whitespace
        s = s.strip()

        # Replace multiple spaces with a single space
        s = re.sub(r"\s+", " ", s)

        # Remove space before punctuation
        s = re.sub(r"\s+([.,!?:;])", r"\1", s)

        return s

    def __eq__(self, other: object) -> bool:
        return self._normalize(str(other)) == self._normalize(self._expected)

    def __repr__(self) -> str:
        return self._expected


class StringMatcherSql:
    """Matcher that parses SQL from the given string and compares the parsed result."""

    def __init__(self, expected: str) -> None:
        self._expected_tokens = self._normalize(expected)

    def _normalize(self, sql: str) -> List[sqlparse.sql.Token]:
        """Normalize SQL query: strip comments, uppercase keywords and unquoted identifiers."""
        normalized_parsed_query = sqlparse.parse(
            sqlparse.format(sql.strip(), keyword_case="upper", identifier_case="upper", strip_comments=True)
        )
        assert len(normalized_parsed_query) == 1, unwrap(
            f"""Multi-statement SQL matching not yet supported. The given SQL string has {len(normalized_parsed_query)}
           statements."""
        )
        return [t for t in normalized_parsed_query[0].flatten()]

    def _format_sql_tokens(self, tokens: List[sqlparse.sql.Token], diff: Dict[int, str]) -> str:
        """Format SQL tokens into a string while highlighting differences."""
        output = []
        for ti in range(len(tokens)):
            t = tokens[ti]
            if t.is_whitespace:
                continue

            if ti in diff:
                hint = diff[ti]
                output.append(f"----> {t.value} <----{hint}")
            else:
                output.append(t.value)

        return " ".join(output)

    def __eq__(self, other: object) -> bool:
        actual_tokens = self._normalize(str(other))

        # Walk through expected and actual query tokens and compare. Whitespace is skipped. Differences are recorded.

        eti = 0
        ati = 0
        expected_mismatched_tokens = {}
        actual_mismatched_tokens = {}
        while eti < len(self._expected_tokens) and ati < len(actual_tokens):
            et = self._expected_tokens[eti]
            if et.is_whitespace:
                eti += 1
                continue
            at = actual_tokens[ati]
            if at.is_whitespace:
                ati += 1
                continue

            if et.ttype != at.ttype:
                expected_mismatched_tokens[eti] = f" [expected type: {et.ttype} actual type: {at.ttype}]"
                actual_mismatched_tokens[ati] = f" [expected type: {et.ttype} actual type: {at.ttype}]"
            elif et.value != at.value:
                expected_mismatched_tokens[eti] = f" [actual value: {at.value}]"
                actual_mismatched_tokens[ati] = f" [expected value: {et.value}]"
            eti += 1
            ati += 1

        # One or both of the token sequences have ended. Complete both sequences and mark all differences.

        while eti < len(self._expected_tokens):
            expected_mismatched_tokens[eti] = " [token not in actual query]"
            eti += 1

        while ati < len(actual_tokens):
            actual_mismatched_tokens[ati] = " [token not in expected query]"
            ati += 1

        # If mismatches have been recorded, output differences and return False.

        if len(expected_mismatched_tokens) + len(actual_mismatched_tokens) > 0:
            logger.warn(
                f"""
---- SQL string mismatch:
actual length {len(actual_tokens)} mismatched {len(actual_mismatched_tokens)}
expected length {len(self._expected_tokens)} mismatched {len(expected_mismatched_tokens)}

==== ACTUAL  : {self._format_sql_tokens(actual_tokens, actual_mismatched_tokens)}

==== EXPECTED: {self._format_sql_tokens(self._expected_tokens, expected_mismatched_tokens)}
"""
            )
            return False

        return True

    def __repr__(self) -> str:
        return self._format_sql_tokens(self._expected_tokens, {})
