import os
import pathlib

if __name__ == '__main__':
  path = pathlib.Path(__file__).resolve().parent
  path /= '.\\.virenv\\Scripts\\tox.exe'
  path = f"\"{path}\""
  os.system(path)
