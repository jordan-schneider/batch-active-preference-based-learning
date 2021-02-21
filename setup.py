import os
from distutils.core import setup


def local_dep(package: str) -> str:
    return f"{package}@file://localhost/{os.getcwd()}/{package}/"


# TODO(joschnei): Publish driver code on separate github so I can speify a remote dependency here.
setup(
    name="value-alignment-verification",
    packages=["active"],
    install_requires=[local_dep("driver"),],
)
