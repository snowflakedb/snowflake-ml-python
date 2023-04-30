# DESCRIPTION: Shell script to run code coverage for snowml repository
# AUTHOR     : Xinyi Jiang
# CONTACT    : xinyi.jiang@snowflake.com
# CLI TOOLS  : bazel, lcov (genhtml)
# NOTES      :
# - Run bazel coverage report, output as lcov format
# - Output code coverage report in html format, under ./html_coverage_report/ folder.

# Treat unset variables as an error when substituting.
bazel clean

set -u
bazel coverage --cache_test_results=no \
               --combined_report=lcov \
               --test_tag_filters=-autogen \
               //...
bazel_exit_code=$?
genhtml --prefix "$(pwd)" --output html_coverage_report "$(bazel info output_path)/_coverage/_coverage_report.dat"
# Bazel exit code
#   0: Success;
#   4: Build Successful but no tests found
# See https://bazel.build/run/scripts#exit-codes
if (( $bazel_exit_code != 0 && $bazel_exit_code != 4)) ; then
  exit $bazel_exit_code
fi
