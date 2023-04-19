# DESCRIPTION: Shell script to run code coverage for snowml repository, and then upload to the database
# AUTHOR     : Xinyi Jiang
# CONTACT    : xinyi.jiang@snowflake.com
# CLI TOOLS  : bazel, lcov (genhtml)
# NOTES      :
# - Run bazel coverage report, output as lcov format
# - Output code coverage report in html format, under ./html_coverage_report/ folder.
# - Upload to database

# Treat unset variables as an error when substituting.
set -u
set -o pipefail

bazel clean
_START_TIME=$(date +%s)
bazel coverage --cache_test_results=no \
               --combined_report=lcov \
               --test_tag_filters=-autogen \
               //...
_END_TIME=$(date +%s)
bazel_exit_code=$?
genhtml --prefix "$(pwd)" --output html_coverage_report "$(bazel info output_path)/_coverage/_coverage_report.dat"

# Assuming current directory is snowml root/ci
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Identify output file as snowml root/ci/tools/results
# - If not doing so, bazel would put file into ${function}.runfiles/ci/tools/results cache directory
OUTPUT_FILE="${CURRENT_DIR}/tools/results"
PARAMS_FILE="${HOME}/.snowsql/coverage_config"
INPUT_FILE="$(bazel info output_path)/_coverage/_coverage_report.dat"
BREAKDOWN_JSON="${OUTPUT_FILE}/breakdown_coverage.json"

echo "[INFO] running parsing coverage"
bazel run //ci/tools:parse_coverage -- -i $INPUT_FILE -o $OUTPUT_FILE

echo "[INFO] Uploading coverage results to database"
REVISION_NUM=$(git rev-parse HEAD)
ELAPSED_TIME=$(($_END_TIME-$_START_TIME))
bazel run //ci/tools:upload_result -- -b $BREAKDOWN_JSON -c $PARAMS_FILE -e $ELAPSED_TIME -r $REVISION_NUM -n ""

# Bazel exit code
#   0: Success;
#   4: Build Successful but no tests found
# See https://bazel.build/run/scripts#exit-codes
if (( $bazel_exit_code != 0 && $bazel_exit_code != 4)) ; then
  exit $bazel_exit_code
fi
