#!/bin/bash

# Usage
# exclude_tests.sh [-b <bazel_path>] [-f <output_path>]
#
# Flags
# -b: specify path to bazel
# -f: specify output file path
#
# Action
#   - exclude integration tests whose dependency is not part of the wheel package.
#     The missing dependency cuold happen when a new operator is being developed, but not yet released.

set -o pipefail
set -u

echo "Running "$0

bazel="bazel"
output_path="/tmp/files_to_exclude"

while getopts "b:f:" opt; do
  case "${opt}" in
    b)
        bazel=${OPTARG}
        ;;
    f)
        output_path=${OPTARG}
        ;;
    :)
        echo "Option -[bf] requires an argument."
        exit 1
        ;;
    ?)
        echo "Invalid option."
        echo "Usage: $0 [-b <bazel_path>] [-f <output_path>]"
        exit 1
        ;;
  esac
done

# Compute missing dependencies by subtracting deps included in wheel from deps required by tests.
# We only care about dependencies in //snowflake/ml since that's our dev directory.
${bazel} query "kind('py_library rule', deps(tests/...) except deps(snowflake/ml:wheel))" \
  | grep -w "//snowflake/ml" > /tmp/missing_deps

# Reverse search on testing files depending on missing deps and exclude those.
files_to_exclude=$(${bazel} query  \
  "kind('source file', deps(kind('py_test rule', rdeps(tests/..., set($(</tmp/missing_deps)))), 1))" \
  --output location \
  | grep -o "source file.*" \
  | awk -F// '{print $2}' \
  | sed -e 's/:/\//g')

>| ${output_path}
for f in ${files_to_exclude}
do
  echo "Excluding file: "${f}
  echo ${f} >> ${output_path}
done

echo "Done running "$0
