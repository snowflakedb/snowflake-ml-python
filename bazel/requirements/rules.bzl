load("@aspect_bazel_lib//lib:yq.bzl", "yq")
load("@bazel_skylib//rules:diff_test.bzl", "diff_test")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("//bazel:py_rules.bzl", "py_genrule")

_AUTOGEN_HEADERS = """# DO NOT EDIT!
# Generate by running '{generation_cmd}'
"""

_SCHEMA_FILE = "//bazel/requirements:requirements.schema.json"
_PYPROJECT_FILE = "//bazel/requirements/templates:pyproject.toml"

_GENERATE_TOOL = "//bazel/requirements:parse_and_generate_requirements"

_GENERATE_COMMAND = "$(location {}) $(location {{src_requirement_file}}) --schema $(location {}) {{options}} > $@".format(_GENERATE_TOOL, _SCHEMA_FILE)

# "---" is a document start marker, which is legit but optional (https://yaml.org/spec/1.1/#c-document-start). This
# is needed for conda meta.yaml to bypass some bug from conda side.
_YAML_START_DOCUMENT_MARKER = "---"

def generate_pyproject_file(
        name,
        src_requirement_file):
    _cmd = "$(location {}) $(location {{src_requirement_file}} ) --schema $(location {}) --pyproject-template $(location {}) --mode version_requirements --format toml > $@".format(_GENERATE_TOOL, _SCHEMA_FILE, _PYPROJECT_FILE)
    py_genrule(
        name = "{name}_pyproject".format(name = name),
        srcs = [
            src_requirement_file,
            _SCHEMA_FILE,
            _PYPROJECT_FILE,
        ],
        outs = ["pyproject.toml"],
        cmd = _cmd.format(src_requirement_file = src_requirement_file),
        tools = [_GENERATE_TOOL],
    )

def generate_requirement_file(
        name,
        generated_file,
        target,
        cmd,
        src_requirement_file,
        generation_cmd):
    py_genrule(
        name = "gen_{name}_body".format(name = name),
        srcs = [
            src_requirement_file,
            _SCHEMA_FILE,
        ],
        outs = ["{generated}.body".format(generated = generated_file)],
        cmd = _GENERATE_COMMAND.format(src_requirement_file = src_requirement_file, options = cmd),
        tools = [_GENERATE_TOOL],
    )
    if "yml" in target:
        cmd = "(echo \"" + _YAML_START_DOCUMENT_MARKER + "\" ; echo -e \"" + _AUTOGEN_HEADERS.format(generation_cmd = generation_cmd) + "\" ; cat $(location :{generated}.body) ) > $@".format(
            generated = generated_file,
        )
    else:
        cmd = "(echo -e \"" + _AUTOGEN_HEADERS.format(generation_cmd = generation_cmd) + "\" ; cat $(location :{generated}.body) ) > $@".format(
            generated = generated_file,
        )
    native.genrule(
        name = "gen_{name}".format(name = name),
        srcs = [
            "{generated}.body".format(generated = generated_file),
        ],
        outs = [generated_file],
        cmd = cmd,
    )
    diff_test(
        name = "check_{name}".format(name = name),
        failure_message = "Please run: {generation_cmd}".format(generation_cmd = generation_cmd),
        file1 = ":{generated}".format(generated = generated_file),
        file2 = target,
    )

def generate_requirement_file_yaml(
        name,
        template_file,
        generated_file,
        target,
        cmd,
        src_requirement_file,
        generation_cmd):
    py_genrule(
        name = "gen_{name}_body".format(name = name),
        srcs = [
            src_requirement_file,
            _SCHEMA_FILE,
        ],
        outs = ["{generated_file}.body.yaml".format(generated_file = generated_file)],
        cmd = _GENERATE_COMMAND.format(src_requirement_file = src_requirement_file, options = cmd),
        tools = [_GENERATE_TOOL],
    )

    yq(
        name = "gen_{name}_body_format".format(name = name),
        srcs = [
            "{generated_file}.body.yaml".format(generated_file = generated_file),
            template_file,
        ],
        outs = ["{generated_file}.body.formatted.yaml".format(generated_file = generated_file)],
        expression = ". as $item ireduce ({}; . *+ $item ) | sort_keys(..)",
    )

    native.genrule(
        name = "gen_{name}".format(name = name),
        srcs = [
            ":{generated_file}.body.formatted.yaml".format(generated_file = generated_file),
        ],
        outs = [generated_file],
        cmd = "(echo \"" + _YAML_START_DOCUMENT_MARKER + "\" ; echo -e \"" + _AUTOGEN_HEADERS.format(generation_cmd = generation_cmd) + "\"; cat $(location :{generated_file}.body.formatted.yaml) ) > $@".format(generated_file = generated_file),
    )

    diff_test(
        name = "check_{name}".format(name = name),
        failure_message = "Please run: {generation_cmd}".format(generation_cmd = generation_cmd),
        file1 = ":{generated}".format(generated = generated_file),
        file2 = target,
    )

def sync_target(
        name,
        root_path,
        targets,
        src_requirement_file):
    write_file(
        name = "gen_{name}".format(name = name),
        out = "{name}.sh".format(name = name),
        content = [
            # This depends on bash, would need tweaks for Windows
            "#!/usr/bin/env sh",
            # Bazel gives us a way to access the source folder!
            "cd $BUILD_WORKSPACE_DIRECTORY",
        ] + [
            # Paths are now relative to the workspace.
            # We can copy files from bazel-bin to the sources
            "cp -fv bazel-bin/{root_path}/{generated} {target}".format(
                root_path = root_path,
                generated = value["generated"],
                # Convert label to path
                target = value["target"].lstrip("//").lstrip(":").replace(":", "/"),
            )
            for value in targets
        ],
    )

    # This is what you can `bazel run` and it can write to the source folder
    native.sh_binary(
        name = name,
        srcs = ["{name}.sh".format(name = name)],
        data = [":{generated}".format(generated = value["generated"]) for value in targets],
    )
