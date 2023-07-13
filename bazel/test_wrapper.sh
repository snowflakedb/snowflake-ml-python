#!/usr/bin/env bash

# Bazel test wrapper

# Get the bazel arg, which is a bazel generated python entrance file.
# Example: snowflake/ml/model/_model_test
# Part of its content (auto-generated) where the main_rel_path gets picked.
#
#  # The main Python source file.
#  # The magic string percent-main-percent is replaced with the runfiles-relative
#  # filename of the main file of the Python binary in BazelPythonSemantics.java.
#  main_rel_path = 'SnowML/snowflake/ml/model/_model_test.py'
#

ENTRY_FILE="$1"

# Follow how bazel generated entry file works
RUNFILES_DIR=$(dirname $(pwd))
# Get the actual main file by searching in bazel generated file.
MAIN_REL_PATH=$(cat ${ENTRY_FILE} | grep -o "  main_rel_path = '[^']*" | sed "s/  main_rel_path = '//g")

# If not a python test then this is empty
if [[ "$MAIN_REL_PATH" ]]; then
  # Pattern for the main block
  TEST_STR="if[[:space:]]+__name__[[:space:]]+==[[:space:]]+[\'\"]__main__[\'\"]:[[:space:]]*"

  # Check if main block exist
  if ! grep -q -x -E "$TEST_STR" "${RUNFILES_DIR}/${MAIN_REL_PATH}" ; then
    echo "Missing \`if __name__ == \"__main__\":\` block in test entry file ${ENTRY_FILE}, your tests won't be run."
    # Follow pytest which uses exit code 5 to label no tests can be found and run.
    exit 5
  fi
fi

# Execute the actual target
$@
