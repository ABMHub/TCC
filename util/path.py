import os

def create_dir_recursively(path):
  while True:
    try:
      os.mkdir(path)
    except FileExistsError:
      return
    except FileNotFoundError:
      create_dir_recursively("/".join(path.split("/")[:-1]))

def get_extension(path : str):
  return path.split(".")[-1]

def get_file_name(path : str):
  return ".".join(path.split("/")[-1].split(".")[:-1])

def get_file_path(path : str):
  return "/".join(path.split("/")[:-1])