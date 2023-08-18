#!/usr/bin/env sh

# Usage
# changelog_version_check.sh <version> <changelog_path>
#
# Action
#   - Check if the section corresponding to the provided version exists in CHANGELOG.

version=$1
changelog_path=$2

version_escaped=$(echo "${version}" | sed 's/[^^]/[&]/g; s/\^/\\^/g' )

grep -E "##\s+${version_escaped}" "${changelog_path}" || \
(echo "CHNAGELOG.md was not updated, please update by adding new section for the new version." && exit 1)
