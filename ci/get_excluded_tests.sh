#!/bin/bash

# Usage
# exclude_tests.sh [-b <bazel_path>] [-f <output_path>] [- merge_gate|continuous_run|release]
#
# Flags
# -b: specify path to bazel
# -f: specify output file path
# -m: specify the mode from the following options
#       merge_gate: exclude local_only + integration tests whose dependency is not part of the affected targets
#                   compare to the the merge base to main of current revision.
#       continuous_run (default): exclude integration tests whose dependency is not part of the wheel package.
#               The missing dependency could happen when a new operator is being developed,
#               but not yet released. (Alias: release)
#       quarantined: exclude all tests that are not quarantined
#

set -o pipefail
set -u

PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-f <output_path>] [-m merge_gate|continuous_run|release|quarantined]"
    exit "${exit_code}"
}

echo "Running ${PROG}"

bazel="bazel"
output_path="/tmp/files_to_exclude"
mode="continuous_run"
SF_ENV="prod3"

while getopts "b:f:m:e:h" opt; do
    case "${opt}" in
    b)
        bazel=${OPTARG}
        ;;
    f)
        output_path=${OPTARG}
        ;;
    m)
        mode=${OPTARG}
        if ! [[ $mode = "merge_gate" || $mode = "continuous_run" || $mode = "release" || $mode = "quarantined" ]]; then
            help 1
        fi
        if [[ $mode = "release" ]]; then
            mode="continuous_run"
        fi
        ;;
    e)
        SF_ENV=${OPTARG}
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


# Compute missing dependencies by subtracting deps included in wheel from deps required by tests.
# We only care about dependencies in //snowflake since that's our dev directory.
# Reverse search on testing files depending on missing deps and exclude those.

unused_test_rule_file=${working_dir}/unused_test_rule

# -- Begin of Query Rules Heredoc --
cat >"${unused_test_rule_file}" <<EndOfMessage
let missing_deps = filter('//snowflake[:/].*', kind('py_library rule', deps(tests/...) except deps(:wheel))) in
    labels(srcs, kind('py_test rule', rdeps(//tests/..., \$missing_deps, 1) + set($(<"ci/targets/quarantine/${SF_ENV}.txt"))))
EndOfMessage
# -- End of Query Rules Heredoc --

unused_test_targets=$("${bazel}" query --query_file="${unused_test_rule_file}")


if [[ $mode = "merge_gate" ]]; then
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    unaffected_test_rule_file=${working_dir}/unaffected_test_rule

    # -- Begin of Query Rules Heredoc --
    cat >"${unaffected_test_rule_file}" <<EndOfMessage
    let affected_targets = set($(<"${affected_targets_file}")) - set($(<"ci/targets/slow.txt")) - set($(<"ci/targets/quarantine/${SF_ENV}.txt")) in
        let unaffected_targets = //tests/... - deps(\$affected_targets) - rdeps(//tests/..., \$affected_targets) in
            labels(srcs, \$unaffected_targets)
EndOfMessage
    # -- End of Query Rules Heredoc --

    unaffected_test_targets=$("${bazel}" query --query_file="${unaffected_test_rule_file}")
fi

targets_to_exclude_file="${working_dir}/targets_to_exclude_file"

case "${mode}" in
continuous_run)
    echo "${unused_test_targets}" >"${targets_to_exclude_file}"
    ;;
merge_gate)
    # Concat and deduplicate.
    targets_to_exclude=$(printf "%s\n%s\n" "${unused_test_targets}" "${unaffected_test_targets}" | awk '!a[$0]++')
    echo "${targets_to_exclude}" >"${targets_to_exclude_file}"
    ;;
quarantined)
    quarantined_test_rule_file=${working_dir}/quarantined_test_rule

# -- Begin of Query Rules Heredoc --
    cat >"${quarantined_test_rule_file}" <<EndOfMessage
    labels(srcs, kind('py_test rule', //tests/... - set($(<"ci/targets/quarantine/${SF_ENV}.txt"))))
EndOfMessage
# -- End of Query Rules Heredoc --

    quarantined_test_targets=$("${bazel}" query --query_file="${quarantined_test_rule_file}")
    echo "${quarantined_test_targets}" >"${targets_to_exclude_file}"
    ;;
*)
    help 1
    ;;
esac

excluded_test_source_rule_file=${working_dir}/excluded_test_source_rule

# -- Begin of Query Rules Heredoc --
cat >"${excluded_test_source_rule_file}" <<EndOfMessage
let skip_continuous_run_targets = set($(<"ci/targets/local_only.txt")) in
    let targets_to_exclude = set($(<"${targets_to_exclude_file}")) + labels(srcs, \$skip_continuous_run_targets) in
            kind('source file', \$targets_to_exclude)
EndOfMessage
    # -- End of Query Rules Heredoc --

${bazel} query --query_file="${excluded_test_source_rule_file}" \
    --output location |
    grep -o "source file.*" |
    awk -F// '{print $2}' |
    sed -e 's/:/\//g' >"${output_path}"

echo "Tests getting excluded:"

cat "${output_path}"

echo "Done running ${PROG}"
