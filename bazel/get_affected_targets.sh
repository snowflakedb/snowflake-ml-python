#!/bin/bash

# Usage
# get_affected_targets.sh [-b <bazel_path>] [-f <output_path>] [-r <revision>] [-w <workspace path>]
#
# Flags
# -b: specify path to bazel
# -f: specify output file path
# -r: specify the revision to use, default the current
# -w: specify the workspace_path, default $(cwd)
#
# Notes:
#   This script relies on bazel-diff, which is installed in WORKSPACE via L6-12 of WORKSPACE file.
#
# Action
#   - Get affected targets list in our repo to the output_path file

set -o pipefail
set -u
PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-f <output_path>] [-r <revision>] [-w <workspace>]"
    exit "${exit_code}"
}

echo "Running ${PROG}"

bazel="bazel"
current_revision=$(git symbolic-ref --short -q HEAD \
  || git describe --tags --exact-match 2> /dev/null \
  || git rev-parse --short HEAD)
pr_revision=$(git rev-parse HEAD)
output_path="/tmp/affected_targets/targets"
workspace_path=$(pwd)



while getopts "b:f:r:w:h" opt; do
    case "${opt}" in
    b)
        bazel=${OPTARG}
        ;;
    f)
        output_path=${OPTARG}
        ;;
    r)
        pr_revision=${OPTARG}
        ;;
    w)
        workspace_path=${OPTARG}
        ;;
    h)
        help 0
        ;;
    :)
        help 1
        ;;
    ?)
        help 1
        ;;
    esac
done

working_dir=$(mktemp -d "/tmp/tmp_XXXXX")
trap 'rm -rf "${working_dir}"' EXIT

starting_hashes_json="${working_dir}/starting_hashes.json"
final_hashes_json="${working_dir}/final_hashes.json"
impacted_targets_path="${working_dir}/impacted_targets.txt"
bazel_diff="${working_dir}/bazel_diff"

"${bazel}" run :bazel-diff --script_path="${bazel_diff}"

git -C "${workspace_path}" checkout "${pr_revision}" --quiet

echo "Generating Hashes for Revision '${pr_revision}'"

"${bazel_diff}" generate-hashes -w "$workspace_path" -b "${bazel}" "${starting_hashes_json}"

MERGE_BASE_MAIN=$(git merge-base "${pr_revision}" main)
git -C "${workspace_path}" checkout "${MERGE_BASE_MAIN}" --quiet

echo "Generating Hashes for merge base ${MERGE_BASE_MAIN}"

$bazel_diff generate-hashes -w "${workspace_path}" -b "${bazel}" "${final_hashes_json}"

echo "Determining Impacted Targets and output to ${output_path}"
$bazel_diff get-impacted-targets -sh "${starting_hashes_json}" -fh "${final_hashes_json}" -o "${impacted_targets_path}"

filter_query_rules_file="${working_dir}/filter_query_rules"

# -- Begin of Query Rules Heredoc --
cat > "${filter_query_rules_file}" << EndOfMessage
let raw_targets = set($(<"${impacted_targets_path}")) in
    \$raw_targets - kind('source file', \$raw_targets) - filter('//external[:/].*', \$raw_targets)
EndOfMessage
# -- End of Query Rules Heredoc --

"${bazel}" query --query_file="${filter_query_rules_file}" >"${output_path}"

git -C "${workspace_path}" checkout "${current_revision}" --quiet
