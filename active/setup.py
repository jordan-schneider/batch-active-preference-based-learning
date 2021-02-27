from distutils.core import setup
from pathlib import Path

# TODO(joschnei): Add typing info

setup(
    name="active",
    version="0.1",
    packages=["active",],
    install_requires=[
        "scipy",
        "numpy",
        "driver @ git+https://github.com/jordan-schneider/driver-env.git#egg=driver",
    ],
    package_data = {
        'active': ['py.typed'],
    },
)
