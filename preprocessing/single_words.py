import math
import numpy as np
import os
import tqdm
import time
from multiprocessing import Process

from preprocessing.align_processing import read_file
from util.path import create_dir_recursively, get_extension, get_file_name, get_file_path
from util.video import get_all_videos, loaders

def slice_all_videos(videos_path, alignments_path, extension, dest_folder, dest_extension = "npz", verbose = 1):
  orig_dest_videos = get_all_videos(videos_path, extension, dest_folder)

  for orig in tqdm.tqdm(orig_dest_videos, desc="Adquirindo palavras soltas", disable=verbose<=0):
    if not os.path.isfile(orig_dest_videos[orig] + dest_extension):
      create_dir_recursively("/".join(orig_dest_videos[orig].split("/")[:-1]))
      try:
        slice_video(orig, os.path.join(alignments_path, get_file_name(orig) + ".align"), get_file_path(orig_dest_videos[orig]))
      except (IndexError, TypeError, ValueError) as err:
        print(f"Video {orig} com erro\n{err}")

def __slice_video_wrapper(video_path, alignment_path, dest_folder):
  try:
    slice_video(video_path, alignment_path, dest_folder)
  except (IndexError, TypeError, ValueError) as err:
    print(f"Video {video_path} com erro\n{err}")

def slice_all_videos_multiprocess(videos_path, alignments_path, extension, dest_folder, dest_extension = "npz", verbose = 1, process_count = 20):
  orig_dest_videos = get_all_videos(videos_path, extension, dest_folder)

  processes = []

  for orig in tqdm.tqdm(orig_dest_videos, desc="Adquirindo palavras soltas", disable=verbose<=0):
    while len(processes) >= process_count:
      deleted = False
      for i in range(len(processes)-1, -1, -1):
        if not processes[i].is_alive():
          processes[i].close()
          del processes[i]
          deleted = True

      if not deleted:
        time.sleep(0.5)

    # provavelmente deveria usar uma pool, mas funciona
    if not os.path.isfile(orig_dest_videos[orig] + dest_extension):
      create_dir_recursively("/".join(orig_dest_videos[orig].split("/")[:-1]))
      p = Process(target=__slice_video_wrapper, args=(orig, os.path.join(alignments_path, get_file_name(orig) + ".align"), get_file_path(orig_dest_videos[orig])))
      p.start()
      processes.append(p)

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