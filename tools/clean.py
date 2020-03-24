import os
import shutil

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))

shutil.rmtree(os.path.join(ROOT_DIR, "build"), ignore_errors = True)
shutil.rmtree(os.path.join(ROOT_DIR, "dist"), ignore_errors = True)
shutil.rmtree(os.path.join(ROOT_DIR, "happy.egg-info"), ignore_errors = True)
