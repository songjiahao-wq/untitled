import os
from pathlib import Path
path = r'D:\yanyi\untitled\youhua\数据集相关\yolo2voc\spilt.py'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
path_test = Path(path)
print(path_test.name)
os.path.exists()