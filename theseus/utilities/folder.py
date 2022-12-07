from pathlib import Path
import os.path as osp

def get_new_folder_name(folder_name):
    i = 0
    new_folder_name = folder_name
    while osp.exists(new_folder_name):
        new_folder_name = folder_name + f"_{i}"
        i+=1
    return new_folder_name

def find_file_recursively(source_folder, filename=None, file_ext=None):
    paths = []
    if filename:
        paths = list(Path(source_folder).rglob(filename))
    elif file_ext:
        paths = list(Path(source_folder).rglob(f'*.{file_ext}'))
    return paths