import os
import platform
import subprocess
import sys

import colorama
import termcolor

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
OUTPUT_DIR = os.path.join(BUILD_DIR, "out")


def log(msg, **kwargs):
  termcolor.cprint(msg, "green", **kwargs)


def init():
  # Enable color logging.
  colorama.init()

  # Create build directory.
  log("Creating build into: " + BUILD_DIR)
  os.makedirs(BUILD_DIR, exist_ok = True)


def platform_args(common_args = [], windows_args = [], posix_args = []):
  if platform.system() == "Windows":
    return common_args + windows_args
  else:
    return common_args + posix_args


def generate(config):
  log(f"Generating {config} build...")
  subprocess.run(
      args = platform_args(
          common_args = [
              "cmake",
              ROOT_DIR,
              f"-DPYTHON_EXECUTABLE={sys.executable}",
              f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={OUTPUT_DIR}",
              f"-DCMAKE_BUILD_TYPE={config}",
              "-DPYSKIP_BUILD_BENCHMARKS=OFF",
          ],
          windows_args = [
              "-Ax64",
              f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={OUTPUT_DIR}",
          ],
          posix_args = [
              f"-DCMAKE_BUILD_TYPE={config}",
          ]
      ),
      cwd = BUILD_DIR,
      check = True,
      stdout = sys.stdout,
      stderr = sys.stderr,
  )


def build(config):
  log(f"Executing {config} build...")
  subprocess.run(
      ["cmake", "--build", ".", "--config", config],
      cwd = BUILD_DIR,
      check = True,
      stdout = sys.stdout,
      stderr = sys.stderr,
  )


def test(config):
  log(f"Executing {config} tests...")
  subprocess.run(
      ["ctest", "-C", config],
      cwd = BUILD_DIR,
      check = True,
      stdout = sys.stdout,
      stderr = sys.stderr,
  )


if __name__ == "__main__":
  init()
  generate("Release")
  build("Release")
  test("Release")
