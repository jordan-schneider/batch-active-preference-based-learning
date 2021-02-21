from distutils.core import setup
from pathlib import Path


def local_dep(package: str) -> str:
    # TODO(joschnei): This requires the value-alignment-verification folder to be installed in the
    # user's home directory, which is not necessarily the case. This should be fixed but I don't
    # know how. Probably the best thing to do is just clean up the driver env and publish that
    # separately, and then pull the remote version instead of this local hack.
    return f"{package}@file://localhost/{Path.home()}/value-alignment-verification/{package}/"


# TODO(joschnei): Add typing info

setup(
    name="active",
    version="0.1",
    packages=["active",],
    install_requires=["scipy", "numpy", local_dep("driver")],
)
