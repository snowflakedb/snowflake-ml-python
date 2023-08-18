INSTALLER_SCRIPT_EXT_MAP = {
    "Windows": ".exe",
    "MacOSX": ".sh",
    "Linux": ".sh",
}

CONDA_EXT_MAP = {
    "Windows": ".bat",
    "MacOSX": "",
    "Linux": "",
}

PYTHON_EXT_MAP = {
    "Windows": ".exe",
    "MacOSX": "",
    "Linux": "",
}

ENV_VAR_SEPARATOR_MAP = {
    "Windows": ";",
    "MacOSX": ":",
    "Linux": ":",
}

EXECUTE_TIMEOUT = 3600

def get_os(rctx):
    os_family = rctx.os.name.lower()
    if "windows" in os_family:
        return "Windows"
    if "mac" in os_family:
        return "MacOSX"
    if "linux" in os_family or "unix" in os_family:
        return "Linux"
    fail("Unsupported OS: {}".format(os_family))

def get_arch_windows(rctx):
    arch = rctx.os.environ.get("PROCESSOR_ARCHITECTURE")
    archw = rctx.os.environ.get("PROCESSOR_ARCHITEW6432")
    if arch in ["AMD64"] or archw in ["AMD64"]:
        return "x86_64"
    if arch in ["x86"]:
        return "x86"
    fail("Unsupported architecture: {}".format(arch))

def get_arch_mac(rctx):
    arch = rctx.execute(["uname", "-m"]).stdout.strip("\n")
    if arch in ["x86_64", "amd64"]:
        return "x86_64"
    elif arch in ["arm64"]:
        return "arm64"
    fail("Unsupported architecture: {}".format(arch))

def get_arch_linux(rctx):
    arch = rctx.execute(["uname", "-m"]).stdout.strip("\n")
    if arch in ["x86_64", "amd64"]:
        return "x86_64"
    if arch in ["aarch64", "aarch64_be", "armv8b", "armv8l", "arm64"]:
        return "aarch64"
    if arch in ["ppc64le", "ppcle", "ppc64", "ppc", "powerpc"]:
        return "ppc64le"
    if arch in ["s390x", "s390"]:
        return "s390x"
    fail("Unsupported architecture: {}".format(arch))

def get_arch(rctx):
    os = get_os(rctx)
    if os == "Windows":
        return get_arch_windows(rctx)
    if os == "MacOSX":
        return get_arch_mac(rctx)
    return get_arch_linux(rctx)

TMP_SCRIPT_TEMPLATE = """
@echo off
if "%OS%"=="Windows_NT" setlocal
{envs}
call {args}
set "EXITCODE=%ERRORLEVEL%"
if "%OS%"=="Windows_NT" ( endlocal & exit /b "%EXITCODE%" )
exit /b "%EXITCODE%""
"""

def execute_waitable_windows(rctx, args, environment = {}, tmp_script = "tmp.bat", **kwargs):
    rctx.file(
        tmp_script,
        content = TMP_SCRIPT_TEMPLATE.format(
            envs = "\n".join(["set \"{}={}\"".format(k, v) for k, v in environment.items()]),
            args = " ".join([str(a) for a in args]),
        ),
    )
    result = rctx.execute([rctx.path(tmp_script)], **kwargs)
    rctx.delete(tmp_script)
    return result

def windowsify(path):
    return str(path).replace("/", "\\")

PATH_SCRIPT = """
@echo off
call echo %PATH%
set "EXITCODE=%ERRORLEVEL%"
if "%OS%"=="Windows_NT" ( endlocal & exit /b "%EXITCODE%" )
exit /b "%EXITCODE%""
"""

# Returns a clean PATH environment variable sufficient for conda installer and commands.
def get_path_envar(rctx):
    os = get_os(rctx)
    if os == "Windows":
        tmp_script = "tmp.bat"
        rctx.file(
            tmp_script,
            content = PATH_SCRIPT,
        )
        getconf_result = rctx.execute([rctx.path(tmp_script)])
        rctx.delete(tmp_script)
    else:
        getconf_result = rctx.execute(["getconf", "PATH"])
    if getconf_result.return_code:
        fail("Unable to get PATH.\nstderr: {}".format(getconf_result.stderr))
    return getconf_result.stdout.strip()
