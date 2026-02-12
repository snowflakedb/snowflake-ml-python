#!/bin/bash
# DESCRIPTION: Utility Shell script to run bazel action for snowml repository
#
# RunBazelAction.sh <test|coverage> [-b <bazel_path>] [-m merge_gate|continuous_run|quarantined|local_unittest|local_all] [-t <target>] [-c <path_to_coverage_report>] [-p <python_version>] [--tags <tags>] [--with-spcs-image]
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
#   -p: specify the Python version (e.g., 3.9, 3.10, 3.11, 3.12, 3.13, 3.14). Default: uses bazel default (3.10)
#   --tags: specify bazel test tag filters (e.g., "feature:jobs,feature:data")
#   --with-spcs-image: use spcs image for testing.
#

set -o pipefail
set -u
set -e

bazel="bazel"
mode="continuous_run"
target=""
SF_ENV="prod3"
WITH_SPCS_IMAGE=false
TAG_FILTERS=""
PYTHON_VERSION=""
PROG=$0

action=$1 && shift

help() {
    local exit_code=$1
    echo "Usage: ${PROG} <test|coverage> [-b <bazel_path>] [-m merge_gate|continuous_run|quarantined|local_unittest|local_all|perf] [-e <snowflake_env>] [-p <python_version>] [--tags <tags>] [--with-spcs-image]"
    echo ""
    echo "Options:"
    echo "  -p <version>        Specify Python version (e.g., 3.9, 3.10, 3.11, 3.12, 3.13, 3.14)"
    echo "  --tags <tags>       Specify bazel tag filters (comma-separated)"
    echo "  --with-spcs-image   Use spcs image for testing."
    echo ""
    echo "Examples:"
    echo "  ${PROG} test --tags 'feature:jobs'"
    echo "  ${PROG} test --tags 'feature:jobs,feature:data'"
    echo "  ${PROG} test -p 3.13"
    echo "  ${PROG} test -m continuous_run -p 3.14"
    exit "${exit_code}"
}

if [[ "${action}" != "test" && "${action}" != "coverage" ]]; then
    help 1
fi

while (($#)); do
    case $1 in
    -m | --mode)
        shift
        if [[ $1 = "merge_gate" || $1 = "continuous_run" || $1 = "quarantined" || $1 = "local_unittest" || $1 = "local_all" || $1 = "perf" ]]; then
            mode=$1
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
    -p | --python-version)
        shift
        PYTHON_VERSION=$1
        ;;
    --tags)
        shift
        TAG_FILTERS="$1"
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
    export RUN_GRYPE=false
    source model_container_services_deployment/ci/build_and_push_images.sh
    action_env=(
        "--action_env=BUILDER_IMAGE_PATH=${BUILDER_IMAGE_PATH}"
        "--action_env=BASE_CPU_IMAGE_PATH=${BASE_CPU_IMAGE_PATH}"
        "--action_env=BASE_GPU_IMAGE_PATH=${BASE_GPU_IMAGE_PATH}"
        "--action_env=BASE_BATCH_CPU_IMAGE_PATH=${BASE_BATCH_CPU_IMAGE_PATH}"
        "--action_env=BASE_BATCH_GPU_IMAGE_PATH=${BASE_BATCH_GPU_IMAGE_PATH}"
        "--action_env=RAY_ORCHESTRATOR_PATH=${RAY_ORCHESTRATOR_PATH}"
        "--action_env=MODEL_LOGGER_PATH=${MODEL_LOGGER_PATH}"
        "--action_env=PROXY_IMAGE_PATH=${PROXY_IMAGE_PATH}"
        "--action_env=VLLM_IMAGE_PATH=${VLLM_IMAGE_PATH}"
        "--action_env=INFERENCE_IMAGE_BUILDER_PATH=${INFERENCE_IMAGE_BUILDER_PATH}"
    )
fi

working_dir=$(mktemp -d "/tmp/tmp_XXXXX")
trap 'rm -rf "${working_dir}"' EXIT

# Check if version.py is the only Python file modified - if so, skip all tests
if [[ "${mode}" = "merge_gate" ]]; then
    changed_files=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1 2>/dev/null || echo "")
    if echo "${changed_files}" | grep -q "snowflake/ml/version.py"; then
        python_files_count=$(echo "${changed_files}" | grep -c '\.py$' || true)
        if [[ ${python_files_count} -eq 1 ]]; then
            echo "Detected changes to version.py as the only Python file - skipping all tests"
            exit 0
        fi
    fi
fi

# Set up tag filtering
if [[ -n "${TAG_FILTERS:-}" ]]; then
    tag_filter="--test_tag_filters=${TAG_FILTERS}"
    echo "Running with tag filters: ${TAG_FILTERS}"
else
    tag_filter="--test_tag_filters="
fi

# Set up Python version config
python_config=""
if [[ -n "${PYTHON_VERSION}" ]]; then
    # Validate Python version is one of the supported versions
    case "${PYTHON_VERSION}" in
        3.9|3.10|3.11|3.12|3.13|3.14)
            python_config="--config=py${PYTHON_VERSION}"
            echo "Running with Python version: ${PYTHON_VERSION}"
            ;;
        *)
            echo "Error: Unsupported Python version '${PYTHON_VERSION}'. Supported versions: 3.9, 3.10, 3.11, 3.12, 3.13, 3.14"
            exit 1
            ;;
    esac
fi

cache_test_results="--cache_test_results=no"

case "${mode}" in
merge_gate)
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    query_expr='kind(".*_test rule", rdeps(//... - set('"$(<"ci/targets/quarantine/${SF_ENV}.txt")"') - set('"$(<"ci/targets/exclude_from_merge_gate.txt")"') - set('"$(<"ci/targets/local_only.txt")"'), set('"$(<"${affected_targets_file}")"')))'
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
perf)
    cache_test_results="--cache_test_results=no"
    query_expr='kind(".*_test rule", //tests/perf/...)'

    if [[ -n "${USER_LDAP:-}" ]]; then
        action_env+=("--action_env=USER_LDAP=${USER_LDAP}")
    fi

    # BUILD_URL is set by CI systems (e.g., Jenkins) to point to the build's web UI
    if [[ -n "${BUILD_URL:-}" ]]; then
        action_env+=("--action_env=BUILD_URL=${BUILD_URL}")
    fi
    git_commit=$(git rev-parse HEAD)
    if [[ -n "${git_commit:-}" ]]; then
        action_env+=("--action_env=GIT_COMMIT=${git_commit}")
    fi

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

# Read groups from optional_dependency_groups.bzl
groups=()
while IFS= read -r line; do
    groups+=("$line")
done < <(python3 -c '
import ast
with open("bazel/platforms/optional_dependency_groups.bzl", "r") as f:
    tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if type(node) == ast.Assign and node.targets[0].id == "OPTIONAL_DEPENDENCY_GROUPS":
            groups = ast.literal_eval(node.value)
            for group in groups.keys():
                print(group)
')

if [ ${#groups[@]} -eq 0 ]; then
    echo "Error: No groups found in optional_dependency_groups.bzl"
    exit 1
fi

# Identify non-python test targets that should always run with core
non_python_test_targets_file="${working_dir}/non_python_test_targets"
"${bazel}" query 'kind(".*_test rule", set('"$(<"${all_test_targets_file}")"')) - kind("py_test rule", set('"$(<"${all_test_targets_file}")"'))' >"${non_python_test_targets_file}"

# Create files for each group's targets
# Create arrays for each group's files and exit codes
group_test_targets_files=()
group_bazel_exit_codes=()
group_coverage_report_files=()

# Filter targets for each group
for i in "${!groups[@]}"; do
    group="${groups[$i]}"
    group_test_targets_files[$i]="${working_dir}/${group}_test_targets"

    # Filter out targets that are incompatible with the group
    "${bazel}" cquery --config="${group}" \
        --output=starlark --starlark:file=bazel/platforms/filter_incompatible_targets.cquery \
        'set('"$(<"${all_test_targets_file}")"')' | \
        awk NF >"${working_dir}/filtered_targets"

    # Compare two sorted files and output lines that are unique to the first file
    comm -2 -3 <(sort "${all_test_targets_file}") \
        <(sort "${working_dir}/filtered_targets") >"${group_test_targets_files[$i]}"
done

# Find targets compatible with core group
core_targets_file="${working_dir}/core_targets"
"${bazel}" cquery --config=core  \
    --output=starlark --starlark:file=bazel/platforms/filter_incompatible_targets.cquery \
    'set('"$(<"${all_test_targets_file}")"')' | \
    awk NF >"${working_dir}/filtered_targets"

comm -2 -3 <(sort "${all_test_targets_file}") \
    <(sort "${working_dir}/filtered_targets") >"${core_targets_file}"

# Always include non-python test targets in core group
if [[ -s "${non_python_test_targets_file}" ]]; then
    cat "${non_python_test_targets_file}" >> "${core_targets_file}"
    sort -u "${core_targets_file}" -o "${core_targets_file}"
fi

# Remove core-compatible targets from other groups
for i in "${!groups[@]}"; do
    group="${groups[$i]}"
    # Create temporary file for filtered targets
    filtered_file="${working_dir}/${group}_filtered"
    # Keep only targets that are not in core_targets or non-python targets
    comm -2 -3 <(sort "${group_test_targets_files[$i]}") \
        <(sort "${core_targets_file}") >"${filtered_file}"
    # Remove non-python test targets from other groups since they should only run with core
    if [[ -s "${non_python_test_targets_file}" ]]; then
        comm -2 -3 <(sort "${filtered_file}") \
            <(sort "${non_python_test_targets_file}") >"${filtered_file}.tmp"
        mv "${filtered_file}.tmp" "${filtered_file}"
    fi
    # Replace original file with filtered results
    mv "${filtered_file}" "${group_test_targets_files[$i]}"
done

groups+=("core")
group_test_targets_files+=("${core_targets_file}")

for i in "${!groups[@]}"; do
    group="${groups[$i]}"
    echo "Running tests for group: ${group}"
    echo "----------------------------------"
    cat "${group_test_targets_files[$i]}"
    echo "----------------------------------"
done

set +e
if [[ "${action}" = "test" ]]; then
    # Run tests for each group
    for i in "${!groups[@]}"; do
        group="${groups[$i]}"
        # Set default test output verbosity (can be overridden via BAZEL_TEST_OUTPUT)
        TEST_OUTPUT_FLAG="--test_output=${BAZEL_TEST_OUTPUT:-errors}"
        "${bazel}" test \
            --jobs=6 \
            --local_test_jobs=6 \
            --config="${group}" \
            ${python_config} \
            "${cache_test_results}" \
            ${TEST_OUTPUT_FLAG} \
            ${action_env[@]+"${action_env[@]}"} \
            "${tag_filter}" \
            --target_pattern_file "${group_test_targets_files[$i]}"
        group_bazel_exit_codes[$i]=$?
    done

elif [[ "${action}" = "coverage" ]]; then
    # Run coverage for each group
    for i in "${!groups[@]}"; do
        group="${groups[$i]}"
        group_coverage_report_files[$i]="${working_dir}/${group}_coverage_report.dat"

        "${bazel}" coverage \
            --config="${group}" \
            ${python_config} \
            "${cache_test_results}" \
            --combined_report=lcov \
            ${action_env[@]+"${action_env[@]}"} \
            "${tag_filter}" \
            --experimental_collect_code_coverage_for_generated_files \
            --target_pattern_file "${group_test_targets_files[$i]}"
        group_bazel_exit_codes[$i]=$?

        cp "$(${bazel} info output_path)/_coverage/_coverage_report.dat" "${group_coverage_report_files[$i]}"
    done

    # Combine all coverage reports
    if [ -z "${coverage_report_file+x}" ]; then
        coverage_report_file=${working_dir}/coverage_report.dat
    fi

    # Combine the first two files
    lcov -a "${group_coverage_report_files[0]}" \
         -a "${group_coverage_report_files[1]}" \
         -o "${coverage_report_file}"

    # Add remaining files if they exist
    for ((i=2; i<${#groups[@]}; i++)); do
        lcov -a "${coverage_report_file}" \
             -a "${group_coverage_report_files[$i]}" \
             -o "${coverage_report_file}"
    done

    if [[ "${mode}" = "local_unittest" || "${mode}" = "local_all" ]]; then
        cp -f "${coverage_report_file}" ".coverage.dat"
    fi

    genhtml --prefix "$(pwd)" --output html_coverage_report "${coverage_report_file}"
fi

# Check exit codes for all groups
exit_code=0
for i in "${!groups[@]}"; do
    # Allow exit code 4 if the target file is empty
    if ! grep -q '[^[:space:]]' "${group_test_targets_files[$i]}" && \
       [[ ${group_bazel_exit_codes[$i]} -eq 4 ]]; then
        group_bazel_exit_codes[$i]=0
    fi

    # If any group fails, mark the overall execution as failed
    if [[ ${group_bazel_exit_codes[$i]} -ne 0 ]]; then
        exit_code=1
    fi
done

exit $exit_code
