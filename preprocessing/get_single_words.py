import math
import numpy as np
from util.video import get_extension, get_file_name
from preprocessing.align_processing import read_file
import os

loaders = {
  "npy": np.load,
  "npz": lambda path: np.load(path)["arr_0"],
  # ".avi"
}

def slice_video(video_path, alignment_path, dest_folder):
  video_extension = get_extension(video_path)
  video_name = get_file_name(video_path)

  video_matrix = loaders[video_extension](video_path)
  start_times, stop_times, sentences = read_file(alignment_path, True)

  for start, stop, wrd in zip(start_times, stop_times, sentences):
    first_frame = math.floor(start/1000)
    last_frame = math.ceil(stop/1000)

    new_video_name = f"{video_name}_{wrd}.npz"
    np.savez(os.path.join(dest_folder, new_video_name), video_matrix[first_frame:last_frame+1])


