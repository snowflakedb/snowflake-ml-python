#!/bin/sh
# Test to make sure //conda-env.yml is the same as bazel/conda-env.yml.

if [ $# -ne 2 ]; then
  echo "must provide the two .yml files as arguments."
  exit 1
fi

# -y: multi-column output
# -W: max column width (contents will be clipped).
diff -y -W 160 $1 $2 ||  \
(echo -e "\nconda-env.yml should be updated." \
         "Please see the instructions on top of the file to re-generate it." && \
 exit 1)
