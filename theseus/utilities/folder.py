import os.path as osp

def get_new_folder_name(folder_name):
    i = 0
    new_folder_name = folder_name
    while osp.exists(new_folder_name):
        new_folder_name = folder_name + f"_{i}"
        i+=1
    return new_folder_name