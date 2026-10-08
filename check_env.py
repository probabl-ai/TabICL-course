import sys
from importlib.metadata import PackageNotFoundError, version

OK = "\x1b[42m[ OK ]\x1b[0m"
FAIL = "\x1b[41m[FAIL]\x1b[0m"

try:
    from packaging.version import Version
except ImportError:
    print(
        FAIL, "'packaging' package not installed, install it with conda or pip"
    )
    sys.exit(1)

# first check the python version
print("Using python in", sys.prefix)
print(sys.version)
pyversion_str = f"{sys.version_info.major}.{sys.version_info.minor}"
pyversion = Version(pyversion_str)

if pyversion < Version("3.11"):
    print(
        FAIL,
        (
            "Python version 3.11 or above is required,"
            f" but {pyversion_str} is installed."
        ),
    )
    sys.exit(1)
print()


def check_version(pkg, min_ver):
    try:
        ver = version(pkg)
    except PackageNotFoundError:
        print(FAIL, f"{pkg} not installed.")
        return
    if Version(ver) < Version(min_ver):
        print(
            FAIL,
            f"{pkg} version {min_ver} or higher required, but {ver} installed.",
        )
    else:
        print(OK, f"{pkg} version {ver}")


requirements = {
    "scikit-learn": "1.8",
    "pandas": "1",
    "matplotlib": "3.0",
    "skrub": "0.1",
    "skore": "0.1",
    "tabicl": "2.1",
    "jupyterlab": "4.0",
}

# now the dependencies
for pkg, min_ver in requirements.items():
    check_version(pkg, min_ver)
