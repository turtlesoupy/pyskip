import glob
import os
import platform
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext


def platform_args(common_args=[], windows_args=[], posix_args=[]):
    if platform.system() == "Windows":
        return common_args + windows_args
    else:
        return common_args + posix_args


class CMakeExtension(Extension):
    """Provides a CPython extension build via CMake."""
    def __init__(self, name, target, root_dir=""):
        Extension.__init__(self, name, sources=[])
        self.root_dir = os.path.abspath(root_dir)
        self.target = target


class CMakeBuild(build_ext):
    """Build command used to build CMake extensions."""
    def run(self):
        # Fail if the required cmake version is not available.
        subprocess.check_output(["cmake", "--version"])

        base_build_temp = self.build_temp
        # Build each extension.
        for ext in self.extensions:
            self.build_temp = str(Path(base_build_temp) / ext.name)
            self.build_extension(ext)
        self.build_temp = base_build_temp

    def build_extension(self, ext):
        # Load default environment variables and set CXXFLAGS to match Python install.
        env = os.environ.copy()
        env["CXXFLAGS"] = env.get("CXXFLAGS", "") + f' -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        # Set the CMake output directory to the usual Python extension output directory.
        output_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Set the build configuration type.
        config = "Debug" if self.debug else "Release"

        # Run CMake generation.
        print(f"Generating cmake build from {self.build_temp}...")
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            args=platform_args(
                common_args=[
                    "cmake",
                    ext.root_dir,
                    f"-DPYTHON_EXECUTABLE={sys.executable}",
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
                    f"-DSKIMPY_BUILD_BENCHMARKS=OFF",
                ],
                windows_args=[
                    "-Ax64",
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output_dir}",
                ],
                posix_args=[
                    f"-DCMAKE_BUILD_TYPE={config}",
                    f"-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
                ],
            ),
            cwd=self.build_temp,
            env=env,
        )

        # Run CMake build.
        print("Executing cmake build...")
        subprocess.check_call(
            args=[
                "cmake",
                "--build",
                ".",
                "--target",
                ext.target,
                "--config",
                config,
                "--parallel",
                str(os.cpu_count()),
            ],
            cwd=self.build_temp,
        )


# List of all public header files.
HEADERS = glob.glob("include/**/*.hpp")

setup(
    name="skimpy",
    version="0.1.5",
    author="Taylor Gordon, Thomas Dimson",
    description="RLE-compressed tensor library",
    long_description="",
    headers=HEADERS,
    ext_modules=[
        CMakeExtension("skimpy", "skimpy_ext"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", exclude="tests"),
    test_suite="nose.collector",
    tests_require=["nose"],
    zip_safe=False,
    setup_requires=["nose>=1.0"],
    install_requires=["numpy"],
)
