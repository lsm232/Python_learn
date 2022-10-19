import os

def check_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise ValueError(f"not find {path}")
