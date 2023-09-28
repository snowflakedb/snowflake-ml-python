PYTHON_EXT_MAP = {
    "linux": "",
    "osx": "",
    "win": ".exe",
}

EXECUTE_TIMEOUT = 3600

def get_os(rctx):
    os_family = rctx.os.name.lower()
    if "windows" in os_family:
        return "win"
    if "mac" in os_family:
        return "osx"
    if "linux" in os_family or "unix" in os_family:
        return "linux"
    fail("Unsupported OS: {}".format(os_family))

def get_arch_windows(rctx):
    arch = rctx.os.environ.get("PROCESSOR_ARCHITECTURE")
    archw = rctx.os.environ.get("PROCESSOR_ARCHITEW6432")
    if arch in ["AMD64"] or archw in ["AMD64"]:
        return "64"
    fail("Unsupported architecture: {}".format(arch))

def get_arch_mac(rctx):
    arch = rctx.execute(["uname", "-m"]).stdout.strip("\n")
    if arch in ["x86_64", "amd64"]:
        return "64"
    elif arch in ["arm64"]:
        return "arm64"
    fail("Unsupported architecture: {}".format(arch))

def get_arch_linux(rctx):
    arch = rctx.execute(["uname", "-m"]).stdout.strip("\n")
    if arch in ["x86_64", "amd64"]:
        return "64"
    if arch in ["aarch64", "aarch64_be", "armv8b", "armv8l", "arm64"]:
        return "aarch64"
    if arch in ["ppc64le", "ppcle", "ppc64", "ppc", "powerpc"]:
        return "ppc64le"
    fail("Unsupported architecture: {}".format(arch))

def get_arch(rctx):
    os = get_os(rctx)
    if os == "win":
        return get_arch_windows(rctx)
    if os == "osx":
        return get_arch_mac(rctx)
    return get_arch_linux(rctx)

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
