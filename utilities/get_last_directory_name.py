import os

def get_last_directory_name(path):
    if os.path.isdir(path):
        last_directory_name = os.path.basename(os.path.normpath(path))
        return last_directory_name
    else:
        return None