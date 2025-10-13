#!/bin/bash

# Usage
# exclude_tests.sh [-b <bazel_path>] [-f <output_path>] [- merge_gate|continuous_run|release] [-a <feature_areas>]
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
# -a: specify comma-separated list of feature areas to INCLUDE (exclude all others) (e.g., "core,modeling,data")

set -o pipefail
set -u

PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-f <output_path>] [-m merge_gate|continuous_run|quarantined] [-e <env>] [-g <group>] [-a <feature_areas>]"
    exit "${exit_code}"
}

echo "Running ${PROG}"

bazel="bazel"
output_path="/tmp/files_to_exclude"
mode="continuous_run"
SF_ENV="prod3"
group="core"
feature_areas=""

while getopts "b:f:m:e:g:a:h" opt; do
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
    a)
        feature_areas=${OPTARG}
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
    let affected_targets = set($(<"${affected_targets_file}")) - set($(<"ci/targets/exclude_from_merge_gate.txt")) - set($(<"ci/targets/quarantine/${SF_ENV}.txt")) in
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

# Handle feature area exclusions if specified
if [[ -n "${feature_areas}" ]]; then
    feature_area_query_file="${working_dir}/feature_area_query"

    # Convert comma-separated feature areas to bazel query format
    IFS=',' read -ra AREAS <<< "${feature_areas}"
    include_conditions=""

    for area in "${AREAS[@]}"; do
        # Trim whitespace
        area=$(echo "${area}" | xargs)
        if [[ -z "${include_conditions}" ]]; then
            include_conditions="attr(tags, \"feature:${area}\", //tests/...)"
        else
            include_conditions="${include_conditions} + attr(tags, \"feature:${area}\", //tests/...)"
        fi
    done

    # Create bazel query to find py_test targets NOT in the specified feature areas
    # This excludes everything that doesn't have the specified feature tags
    cat >"${feature_area_query_file}" <<EndOfMessage
labels(srcs, kind('py_test rule', //tests/... - (${include_conditions})))
EndOfMessage

    feature_area_test_targets=$("${bazel}" query --query_file="${feature_area_query_file}")

    # Add feature area exclusions to existing exclusions
    if [[ -n "${feature_area_test_targets}" ]]; then
        echo "${feature_area_test_targets}" >>"${targets_to_exclude_file}"
        echo "Feature area filtering: added $(echo "${feature_area_test_targets}" | wc -l) non-${feature_areas} test exclusions"
    fi

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

# Special handling for modeling tests: exclude all modeling feature area tests if feature areas specified and "modeling" not included
if [[ -n "${feature_areas}" && ",${feature_areas}," != *",modeling,"* ]]; then
    modeling_query_file="${working_dir}/modeling_query"
    cat >"${modeling_query_file}" <<EndOfMessage
labels(srcs, attr(tags, "feature:modeling", //tests/...))
EndOfMessage
    modeling_tests=$("${bazel}" query --query_file="${modeling_query_file}")
    if [[ -n "${modeling_tests}" ]]; then
        # Transform generate_test_* files to *_test.py files (the actual generated test files)
        echo "${modeling_tests}" | sed 's|^//||' | sed 's|:|/|g' | sed 's|generate_test_\([^.]*\)$|\1_test.py|g' >>"${output_path}"
    fi
fi

# This is for modeling model tests that are automatically generated and not part of the build.
if [[ -n "${incompatible_targets}" ]]; then
    echo "${incompatible_targets}" | sed 's|^//||' | sed 's|:|/|g' | sed 's|$|.py|' >>"${output_path}"
fi

# Force-add any labels in the quarantine list that bazel query didn’t pick up
if [[ -f "ci/targets/quarantine/${SF_ENV}.txt" ]]; then
grep ':' "ci/targets/quarantine/${SF_ENV}.txt" | \
    sed -e 's|^//||' \
        -e 's|:|/|' \
        -e 's|$|.py|' \
    >> "${output_path}"
fi
echo "Tests getting excluded:"

# Sort and deduplicate the exclusion file
sort -u "${output_path}" -o "${output_path}"
cat "${output_path}"

echo "Done running ${PROG}"
