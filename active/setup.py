from distutils.core import setup
from pathlib import Path


def local_dep(package: str) -> str:
    cwd = Path.cwd()
    return f"{package}@file://localhost/{cwd.parent}/value-alignment-verification/{package}/"


setup(
    name="active",
    version="0.1",
    packages=["active",],
    install_requires=["scipy", "numpy", local_dep("driver")],
)
