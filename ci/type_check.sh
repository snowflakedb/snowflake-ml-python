#!/bin/bash

# Just an alias to avoid break Jenkins
PROG=$0

SCRIPT=$(readlink -f "$PROG")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

"${SCRIPTPATH}/type_check/type_check.sh" "$@"
