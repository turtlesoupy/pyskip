from setuptools import setup, find_namespace_packages

setup(
    name="skimpy_blox",
    version="0.0.1",
    author="Taylor Gordon, Thomas Dimson",
    description="A skimpy wrapper library for loading and operating on Minecraft assets",
    packages=find_namespace_packages(),
    install_requires=["skimpy>=0.0.1", "nbt", "tqdm", "pillow"],
)
