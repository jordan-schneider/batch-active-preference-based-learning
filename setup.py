import os
from distutils.core import setup

# TODO(joschnei): Publish driver code on separate github so I can speify a remote dependency here.
setup(
    name="value-alignment-verification",
    packages=["active"],
    install_requires=[
        "numpy",
        "torch",
        "fire",
        "argh",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "seaborn",
        "scipy",
        "TD3 @ git+https://github.com/jordan-schneider/TD3.git#egg=TD3"
        "driver @ git+https://github.com/jordan-schneider/driver-env.git#egg=driver",
    ],
    package_data={"value-alignment-verification": ["py.typed"],},
)
