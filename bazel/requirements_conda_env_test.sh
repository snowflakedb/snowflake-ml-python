#!/bin/sh
# Test to make sure //requirements.txt is the same as bazel/requirements.txt.

if [ $# -ne 2 ]; then
  echo "must provide the two requirements.txt files as arguments."
  exit 1
fi

# -y: multi-column output
# -W: max column width (contents will be clipped).
diff -y  -W 160 $1 $2 ||  \
(echo -e "\nrequirements.txt should be updated." \
         "Please see the instructions on top of the file to re-generate it." && \
 exit 1)
