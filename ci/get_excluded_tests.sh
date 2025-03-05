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
#               but not yet released.
#       quarantined: exclude all tests that are not quarantined
# -e: specify the environment (default: prod3)
# -g: specify the group (default: "core") Test group could be found in bazel/platforms/optional_dependency_groups.bzl.
#     `core` group is the default group that includes all tests that does not have a group specified.
#     `all` group includes all tests.

set -o pipefail
set -u

PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-f <output_path>] [-m merge_gate|continuous_run|quarantined] [-e <env>] [-g <group>]"
    exit "${exit_code}"
}

echo "Running ${PROG}"

bazel="bazel"
output_path="/tmp/files_to_exclude"
mode="continuous_run"
SF_ENV="prod3"
group="core"

while getopts "b:f:m:e:g:h" opt; do
    case "${opt}" in
    b)
        bazel=${OPTARG}
        ;;
    f)
        output_path=${OPTARG}
        ;;
    m)
        mode=${OPTARG}
        if ! [[ $mode = "merge_gate" || $mode = "continuous_run" || $mode = "quarantined" ]]; then
            help 1
        fi
        ;;
    e)
        SF_ENV=${OPTARG}
        ;;
    g)
        group=${OPTARG}
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

all_test_targets=$("${bazel}" query 'kind("py_test rule", //tests/...)')
all_test_targets_file="${working_dir}/all_test_targets"
echo "${all_test_targets}" >"${all_test_targets_file}"

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
    targets_to_exclude=$(printf "%s\n%s\n" "${unused_test_targets}" "${unaffected_test_targets}" | sort | uniq)
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

incompatible_targets=""

if [[ $group != "all" ]]; then
    incompatible_targets_file="${working_dir}/incompatible_targets"
    incompatible_targets=$("${bazel}" cquery --config="${group}" \
        --output=starlark --starlark:file=bazel/platforms/filter_incompatible_targets.cquery \
        'set('"${all_test_targets}"')' | \
        awk NF)
    if [[ $group != "core" ]]; then
        "${bazel}" cquery --config="core" \
            --output=starlark --starlark:file=bazel/platforms/filter_incompatible_targets.cquery \
            'set('"${all_test_targets}"')' | \
            awk NF>"${working_dir}/core_incompatible_targets"

        core_compatible_targets=$(comm -23 <(sort "${all_test_targets_file}") <(sort "${working_dir}/core_incompatible_targets"))
        incompatible_targets=$(printf "%s\n%s\n" "${incompatible_targets}" "${core_compatible_targets}" | sort | uniq)
    fi
    "${bazel}" query "labels(srcs, set(${incompatible_targets}))" >"${incompatible_targets_file}"
    mv "${targets_to_exclude_file}" "${targets_to_exclude_file}.tmp"
    sort -u "${targets_to_exclude_file}.tmp" "${incompatible_targets_file}" | uniq -u >"${targets_to_exclude_file}"
fi

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

# This is for modeling model tests that are automatically generated and not part of the build.
if [[ -n "${incompatible_targets}" ]]; then
    echo "${incompatible_targets}" | sed 's|^//||' | sed 's|:|/|g' | sed 's|$|.py|' >>"${output_path}"
fi

echo "Tests getting excluded:"

cat "${output_path}"

echo "Done running ${PROG}"
