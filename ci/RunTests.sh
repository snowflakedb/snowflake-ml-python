# DESCRIPTION: Shell script to run all tests for snowml repository
# AUTHOR     : Tyler Hoyt
# CONTACT    : tyler.hoyt@snowflake.com
# CLI TOOLS  : bazel

bazel clean

bazel test --cache_test_results=no \
           --test_output=errors \
           --test_tag_filters=-autogen \
           //...
