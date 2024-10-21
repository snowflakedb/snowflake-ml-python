#!/bin/bash
# DESCRIPTION: Utility Shell script to run bazel action for snowml repository
#
# RunBazelAction.sh <test|coverage> [-b <bazel_path>] [-m merge_gate|continuous_run|quarantined|local_unittest|local_all] [-t <target>] [-c <path_to_coverage_report>]
#
# Args:
#   action: bazel action, choose from test and coverage
#
# Flags
#   -b: specify path to bazel.
#   -m: specify the mode from the following options
#       merge_gate: run affected tests only.
#       continuous_run (default): run all tests. (For nightly run. Alias: release)
#       quarantined: Run quarantined tests.
#       local_unit: run all unit tests affected by target defined by -t
#       local_all: run all tests including integration tests affected by target defined by -t
#   -t: specify the target for local_unit and local_all mode
#   -c: specify the path to the coverage report dat file.
#   -e: specify the environment, used to determine.
#

set -o pipefail
set -u
set -e

bazel="bazel"
mode="continuous_run"
target=""
SF_ENV="prod3"
WITH_SPCS_IMAGE=false
PROG=$0

action=$1 && shift

help() {
    local exit_code=$1
    echo "Usage: ${PROG} <test|coverage> [-b <bazel_path>] [-m merge_gate|continuous_run|quarantined|local_unittest|local_all] [-e <snowflake_env>] [--with-spcs-image]"
    exit "${exit_code}"
}

if [[ "${action}" != "test" && "${action}" != "coverage" ]]; then
    help 1
fi

while (($#)); do
    case $1 in
    -m | --mode)
        shift
        if [[ $1 = "merge_gate" || $1 = "continuous_run" || $1 = "quarantined" || $1 = "release" || $1 = "local_unittest" || $1 = "local_all" ]]; then
            mode=$1
            if [[ $mode = "release" ]]; then
                mode="continuous_run"
            fi
        else
            help 1
        fi
        ;;
    -b | --bazel_path)
        shift
        bazel=$1
        ;;
    -t | --target)
        shift
        if [[ "${mode}" = "local_unittest" || "${mode}" = "local_all" ]]; then
            target=$1
        else
            help 1
        fi
        ;;
    -c | --coverage_report)
        shift
        coverage_report_file=$1
        ;;
    -e | --snowflake_env)
        shift
        SF_ENV=$1
        ;;
    --with-spcs-image)
        WITH_SPCS_IMAGE=true
        ;;
    -h | --help)
        help 0
        ;;
    *)
        help 1
        ;;
    esac
    shift
done

if [[ ("${mode}" = "local_unittest" || "${mode}" = "local_all") ]]; then
    if [[ -z "${target}" ]]; then
        help 1
    fi
else
    "${bazel}" clean
fi

action_env=()

if [[ "${WITH_SPCS_IMAGE}" = true ]]; then
    source model_container_services_deployment/ci/build_and_push_images.sh
    action_env=("--action_env=BUILDER_IMAGE_PATH=${BUILDER_IMAGE_PATH}" "--action_env=BASE_CPU_IMAGE_PATH=${BASE_CPU_IMAGE_PATH}" "--action_env=BASE_GPU_IMAGE_PATH=${BASE_GPU_IMAGE_PATH}")
fi

working_dir=$(mktemp -d "/tmp/tmp_XXXXX")
trap 'rm -rf "${working_dir}"' EXIT

tag_filter="--test_tag_filters="
cache_test_results="--cache_test_results=no"

case "${mode}" in
merge_gate)
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    query_expr='kind(".*_test rule", rdeps(//... - set('"$(<"ci/targets/quarantine/${SF_ENV}.txt")"') - set('"$(<"ci/targets/slow.txt")"') - set('"$(<"ci/targets/local_only.txt")"'), set('"$(<"${affected_targets_file}")"')))'
    ;;
continuous_run)
    query_expr='kind(".*_test rule", //... - set('"$(<"ci/targets/quarantine/${SF_ENV}.txt")"') - set('"$(<"ci/targets/local_only.txt")"'))'
    ;;
quarantined)
    query_expr='kind(".*_test rule", set('"$(<"ci/targets/quarantine/${SF_ENV}.txt")"') - set('"$(<"ci/targets/local_only.txt")"'))'
    ;;
local_unittest)
    cache_test_results="--cache_test_results=yes"

    query_expr='kind(".*_test rule", rdeps(//... - //tests/..., '"${target}"'))'
    ;;
local_all)
    cache_test_results="--cache_test_results=yes"

    query_expr='kind(".*_test rule", rdeps(//..., '"${target}"'))'
    ;;
*)
    help 1
    ;;
esac

# Query all targets
all_test_targets_query_file=${working_dir}/all_test_targets_query
printf "%s" "${query_expr}" >"${all_test_targets_query_file}"
all_test_targets_file=${working_dir}/all_test_targets
"${bazel}" query --query_file="${all_test_targets_query_file}" >"${all_test_targets_file}"

if [[ ! -s "${all_test_targets_file}" && "${mode}" = "merge_gate" ]]; then
    exit 0
fi

# Filter out targets need to run with extended env
extended_test_targets_query_expr="set($(<"${all_test_targets_file}"))"
extended_test_targets_query_file=${working_dir}/all_test_targets_query
printf "%s" "${extended_test_targets_query_expr}" >"${extended_test_targets_query_file}"
extended_test_targets_file=${working_dir}/extended_test_targets
"${bazel}" cquery --query_file="${extended_test_targets_query_file}" --output=starlark --starlark:file=bazel/platforms/filter_incompatible_targets.cquery | awk NF >"${extended_test_targets_file}"

# Subtract to get targets to run in sf_only env
sf_only_test_targets_file=${working_dir}/sf_only_test_targets
comm -2 -3 <(sort "${all_test_targets_file}") <(sort "${extended_test_targets_file}") >"${sf_only_test_targets_file}"

set +e
if [[ "${action}" = "test" ]]; then
    "${bazel}" test \
        "${cache_test_results}" \
        --test_output=errors \
        --flaky_test_attempts=2 \
        ${action_env[@]+"${action_env[@]}"} \
        "${tag_filter}" \
        --target_pattern_file "${sf_only_test_targets_file}"
    sf_only_bazel_exit_code=$?

    # Test with extended env
    "${bazel}" test \
        --config=extended \
        "${cache_test_results}" \
        --test_output=errors \
        --flaky_test_attempts=2 \
        ${action_env[@]+"${action_env[@]}"} \
        "${tag_filter}" \
        --target_pattern_file "${extended_test_targets_file}"
    extended_bazel_exit_code=$?
elif [[ "${action}" = "coverage" ]]; then
    "${bazel}" coverage \
        "${cache_test_results}" \
        --combined_report=lcov \
        ${action_env[@]+"${action_env[@]}"} \
        "${tag_filter}" \
        --experimental_collect_code_coverage_for_generated_files \
        --target_pattern_file "${sf_only_test_targets_file}"
    sf_only_bazel_exit_code=$?

    sf_only_coverage_report_file=${working_dir}/sf_only_coverage_report.dat
    cp "$(${bazel} info output_path)/_coverage/_coverage_report.dat" "${sf_only_coverage_report_file}"

    # Test with extended env
    "${bazel}" coverage \
        --config=extended \
        "${cache_test_results}" \
        --combined_report=lcov \
        ${action_env[@]+"${action_env[@]}"} \
        "${tag_filter}" \
        --experimental_collect_code_coverage_for_generated_files \
        --target_pattern_file "${extended_test_targets_file}"
    extended_bazel_exit_code=$?

    extended_coverage_report_file=${working_dir}/extended_coverage_report.dat
    cp "$(${bazel} info output_path)/_coverage/_coverage_report.dat" "${extended_coverage_report_file}"

    if [ -z "${coverage_report_file+x}" ]; then
        coverage_report_file=${working_dir}/coverage_report.dat
    fi

    lcov -a "${sf_only_coverage_report_file}" -a "${extended_coverage_report_file}" -o "${coverage_report_file}"

    if [[ "${mode}" = "local_unittest" || "${mode}" = "local_all" ]]; then
        cp -f "${coverage_report_file}" ".coverage.dat"
    fi

    genhtml --prefix "$(pwd)" --output html_coverage_report "${coverage_report_file}"
fi

# Bazel exit code
#   0: Success;
#   4: Build Successful but no tests found
# See https://bazel.build/run/scripts#exit-codes
# We allow exit code be 4 only when the targets is empty file.
if ! grep -q '[^[:space:]]' "${sf_only_test_targets_file}" && [[ ${sf_only_bazel_exit_code} -eq 4 ]]; then
    sf_only_bazel_exit_code=0
fi

if ! grep -q '[^[:space:]]' "${extended_test_targets_file}" && [[ ${extended_bazel_exit_code} -eq 4 ]]; then
    extended_bazel_exit_code=0
fi

if [[ ${sf_only_bazel_exit_code} -eq 0 && ${extended_bazel_exit_code} -eq 0 ]]; then
    exit 0
else
    exit 1
fi
