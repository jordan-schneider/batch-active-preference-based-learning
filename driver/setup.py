from distutils.core import setup

setup(
    name="driver",
    version="0.1",
    packages=["driver",],
    install_requires=["numpy", "theano", "gym", "mujoco_py", "matplotlib", "pyglet",],
)
