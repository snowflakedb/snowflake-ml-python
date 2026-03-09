package(default_visibility = ["//visibility:public"])

genrule(
    name = "pr",
    srcs = glob([
        "pr/**/*.go",
        "pr/go.mod",
        "pr/go.sum",
    ]),
    outs = ["pr_bin"],
    cmd = """
        set -e
        export HOME=$$(echo ~)
        export GOMODCACHE=$$HOME/go/pkg/mod
        export GOCACHE=$$HOME/.cache/go-build
        EXEC_ROOT=$$(pwd)
        REPO_ROOT=$$(dirname $$(dirname $$(realpath $(location pr/go.mod))))
        WORK_DIR=$$(mktemp -d)
        trap "rm -rf $$WORK_DIR" EXIT
        cp -r $$REPO_ROOT/pr/* $$WORK_DIR/
        cd $$WORK_DIR
        go mod download
        go build -o $$EXEC_ROOT/$@ ./cmd/pr
    """,
    executable = True,
    local = True,
    tags = ["no-sandbox"],
)

genrule(
    name = "jira",
    srcs = glob([
        "jira/**/*.go",
        "jira/go.mod",
    ]),
    outs = ["jira_bin"],
    cmd = """
        set -e
        export HOME=$$(echo ~)
        export GOMODCACHE=$$HOME/go/pkg/mod
        export GOCACHE=$$HOME/.cache/go-build
        EXEC_ROOT=$$(pwd)
        REPO_ROOT=$$(dirname $$(dirname $$(realpath $(location jira/go.mod))))
        WORK_DIR=$$(mktemp -d)
        trap "rm -rf $$WORK_DIR" EXIT
        cp -r $$REPO_ROOT/jira/* $$WORK_DIR/
        cd $$WORK_DIR
        go build -o $$EXEC_ROOT/$@ ./cmd/jira
    """,
    executable = True,
    local = True,
    tags = ["no-sandbox"],
)
