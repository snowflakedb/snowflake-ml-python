#!/usr/bin/env bash

{VERBOSE_BASH}
set -u
set -o pipefail

main() {
  local report_file
  local status
  local mypy

  report_file="{OUTPUT}"
  mypy="{MYPY_BIN}"

  export MYPYPATH="$(pwd):{ADDITIONAL_MYPYPATH}"

  $mypy {VERBOSE_OPT} --bazel {PACKAGE_ROOTS} --config-file {MYPY_INI} --cache-map {CACHE_MAP_TRIPLES} -- {SRCS} > "${report_file}" 2>&1
  status=$?

  if [[ $status -ne 0 ]]; then
    printf "\033[0;31m======== BEGIN MYPY ERROR ========\033[0m\n"
    cat "${report_file}" # Show MyPy's error to end-user via Bazel's console logging
    printf "\033[0;31m======== END   MYPY ERROR ========\033[0m\n"
    exit 1
  fi
}

main "$@"
