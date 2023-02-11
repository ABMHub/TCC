import cv2
import numpy as np
import os

loaders = {
  "npy": np.load,
  "npz": lambda path: np.load(path)["arr_0"],
  # ".avi"
}

def get_all_videos(path, extension, dest_folder, preserve_folder_structure = False):
  orig_dest_videos = {}
  files_in = os.listdir(path)
  for file in files_in:
    curr_file_path = os.path.join(path,file)
    if os.path.isdir(curr_file_path):
      if preserve_folder_structure:
        new_dest_folder = os.path.join(dest_folder, file)
        orig_dest_videos.update(get_all_videos(curr_file_path, extension, new_dest_folder))
      else:
        orig_dest_videos.update(get_all_videos(curr_file_path, extension, dest_folder))

    if file.endswith(extension):
      new_video_path = os.path.join(dest_folder, ".".join(file.split(".")[:-1]))
      orig_dest_videos[curr_file_path] = new_video_path

  return orig_dest_videos

def read_video(path):
  vid = cv2.VideoCapture(path)
  frames = []
  while vid.isOpened():
    _, frame = vid.read()
    if frame is None:
      break
    frames.append(np.transpose(frame.tolist(), (1, 0, 2)))

  return frames