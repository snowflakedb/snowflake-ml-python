import argparse
import json
from enum import Enum
from os import makedirs, path
from typing import Dict, List, Union


class _FSMState(Enum):
    # Define the states of the finite state machine
    START = 0
    SF = 1
    LH = 2
    LF = 3
    EOF = 4


class LcovCoverageParser:
    """
    Read from code coverage output in Lcov format, parse by Finite State Machine, and output as json file

    Input: lcov format file path, output directory
    - Information needed from Lcov:  SF (source file path), LF (line found), LH (line hit), end_of_record

    Output:
    1. summary_coverage.json
    - Format: {
            "all_line_hit": ...,
            "all_line_found": ...,
            "number_of_files": ...,
        }
    2. breakdown_coverage
    - Format: List of {"package": ..., "filename": ..., "line_found": ..., "line_hit": ...}
    - Explanation of keys: Package (file directory), Filename, LF (line found), LH (line hit)
    """

    def __init__(self, input_file_path: str, output_directory: str) -> None:
        self._input_file_path = input_file_path
        self._result_dir = output_directory

    def _read_and_parse(self) -> List[Dict[str, Union[str, int]]]:
        # Define initial state
        state = _FSMState.START

        # Create variables to store the parsed data
        breakdown_coverage: List[Dict[str, Union[str, int]]] = []
        package, filename, line_found, line_hit = "", "", 0, 0

        # Iterate over the LCOV format data line by line
        with open(self._input_file_path) as f:
            for line in f:
                line_arr = line.strip().split(":")
                line_category, line_info = line_arr[0], line_arr[1:]
                if state == _FSMState.START and line_category == "SF":
                    state = _FSMState.SF
                    current_file_info = line_info[0].split("/")
                    filename = current_file_info[-1]
                    package = "/".join(current_file_info[:-1])
                elif state == _FSMState.SF and line_category == "LH":
                    state = _FSMState.LH
                    line_hit = int(line_info[0])
                elif state == _FSMState.LH and line_category == "LF":
                    state = _FSMState.LF
                    line_found = int(line_info[0])
                elif state == _FSMState.LF and line_category == "end_of_record":
                    state = _FSMState.START
                    breakdown_coverage.append(
                        {"package": package, "filename": filename, "line_found": line_found, "line_hit": line_hit}
                    )
                elif line_category in ("TN", "FN", "FNF", "FNH", "BRDA", "BRF", "DA", "BRH"):
                    continue
                else:
                    # handles unexpected errors
                    raise ValueError(f"current state {state}, but category is {line_category}")

        if state != _FSMState.START:
            raise ValueError("Reached end of file, but haven't encountered keyword end_of_record!")

        return breakdown_coverage

    def _write_to_json(self, breakdown_coverage: List[Dict[str, Union[str, int]]]) -> None:
        makedirs(self._result_dir, exist_ok=True)
        with open(path.join(self._result_dir, "breakdown_coverage.json"), "w") as outfile:
            json.dump(breakdown_coverage, outfile)

    def process(self) -> None:
        breakdown_coverage = self._read_and_parse()
        self._write_to_json(breakdown_coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse lcov format to json file", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-file-path",
        type=str,
        help="lcov absolute file path",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        help="output directory for json files",
    )
    args = parser.parse_args()

    parse_coverage = LcovCoverageParser(args.input_file_path, args.output_directory)
    parse_coverage.process()
