import dlib
import cv2
import numpy as np
import os
import tqdm
from multiprocessing import Pool
import time
import math

from util.video import get_all_videos
from util.path import create_dir_recursively

ffd = dlib.get_frontal_face_detector()
lmd = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

VIDEO_HEIGHT = 720
VIDEO_WIDTH  = 576

def convert_all_videos(path, extension, dest_folder, numpy_file = True, verbose = 1):
  orig_dest_videos = get_all_videos(path, extension, dest_folder)
  dest_extension = ".npz" if numpy_file else ".avi"

  for orig in tqdm.tqdm(orig_dest_videos, desc="Convertendo Video", disable=verbose<=0):
    if not os.path.isfile(orig_dest_videos[orig] + dest_extension):
      create_dir_recursively("/".join(orig_dest_videos[orig].split("/")[:-1]))
      try:
        FaceVideo(orig, verbose = 0 if verbose < 2 else 1).get_mouth_video(orig_dest_videos[orig])
      except (IndexError, TypeError, AssertionError) as err:
        print(f"Video {orig} com erro\n{err}")

def __video_class_wrapper(args):
  orig, verbose, path = args
  create_dir_recursively("/".join(path.split("/")[:-1]))
  try:
    FaceVideo(orig, verbose = 0 if verbose < 2 else 1).get_mouth_video(path)
  except (IndexError, TypeError, AssertionError) as err:
    print(f"Video {orig} com erro\n{err}")

def convert_all_videos_multiprocess(path, extension, dest_folder, verbose = 1, numpy_file = True, process_count = 12):
  orig_dest_videos = get_all_videos(path, extension, dest_folder)
  dest_extension = ".npz" if numpy_file else ".avi"

  pool = Pool(processes=process_count)
  args = []
  for orig in orig_dest_videos:
    if not os.path.isfile(orig_dest_videos[orig] + dest_extension):
      args.append((orig, verbose, orig_dest_videos[orig]))

  [None for _ in tqdm.tqdm(pool.imap_unordered(__video_class_wrapper, args), desc="Convertendo Video", disable=verbose<=0, total=len(orig_dest_videos), initial=len(orig_dest_videos) - len(args))]

class FaceVideo:
  def __init__(self, video_path : str, numpy_file : bool = True, verbose = 1):
    self.video = cv2.VideoCapture(video_path)
    self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.frames = []
    self.numpy_file = numpy_file
    self.verbose = verbose

    pbar = tqdm.tqdm(desc='Carregando video', total=self.frame_count, disable=self.verbose==0)
    while self.video.isOpened():
      _, frame = self.video.read()
      if frame is None: break

      self.frames.append(FaceFrame(frame))
      pbar.update()
    pbar.close()

  def get_mouth_video(self, path):
    for frame_obj in tqdm.tqdm(self.frames, desc="Alinhando frames", disable=self.verbose==0):
      frame_obj.transform() 

    # n_video = cv2.VideoWriter("./testetcc.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25, tuple(reversed(self.frames[0].img.shape[0:2])))
    # for frame in self.frames:
    #   n_video.write(frame.img)
    # n_video.release()

    self.mouth_imgs = np.array([cv2.resize(frame_obj.get_mouth_img(), (100, 50)) for frame_obj in tqdm.tqdm(self.frames, desc="Adquirindo recortes da boca", disable=self.verbose==0)])

    assert self.mouth_imgs.shape == (75, 50, 100, 3)

    if self.numpy_file:
      np.savez(path + ".npz", self.mouth_imgs)

    else:
      n_video = cv2.VideoWriter(path + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 25, (100, 50))
      for frame in self.mouth_imgs:
        n_video.write(frame)
      n_video.release()

class FaceFrame:
  def __init__(self, img : np.ndarray):
    self.img = img

    faces = ffd(img, 0)
    # import random
    # if len(faces) != 1: cv2.imwrite(f"./test/imtest{int(random.random()*100)}.png", self.img)
    assert len(faces) == 1, "Mais ou menos de um rosto detectado"
    self.face = faces[0]

    self.l_eye, self.r_eye, self.mouth = self.get_mouth_coords()

  def get_mouth_coords(self) -> tuple[float, float, float]:
    lm = lmd(self.img, self.face)
    
    l_eye  = self.__get_center(lm, 36, 42)
    r_eye  = self.__get_center(lm, 42, 48)
    mouth = self.__get_center(lm, 48, 68)

    return l_eye, r_eye, mouth
      
  def __get_center(self, lm, start, end):
    x_list = []
    y_list = []

    for i in range(start, end):
      x, y = lm.part(i).x, lm.part(i).y
      x_list.append(x)
      y_list.append(y)

    return np.mean(x_list), np.mean(y_list)

  def __get_point(self, lm, number) -> tuple[int, int]:
    return int(lm.part(number).x), int(lm.part(number).y)

  def __get_dest_points(self):
    r_eye = self.r_eye
    l_eye = self.l_eye
    mouth = self.mouth

    mid_eyes = np.mean([r_eye[0], l_eye[0]]), np.mean([r_eye[1], l_eye[1]])
    eyes_dist = math.dist(r_eye, l_eye)
    mouth_mid_eyes_dist = math.dist(mid_eyes, mouth)

    new_mid_eyes = (mouth[0], mouth[1] - mouth_mid_eyes_dist)
    new_l_eye = (new_mid_eyes[0] - eyes_dist/2, new_mid_eyes[1])
    new_r_eye = (new_mid_eyes[0] + eyes_dist/2, new_mid_eyes[1])

    return np.array([
      new_l_eye,
      new_r_eye,
      mouth,
    ], dtype=np.float32)

  def transform(self):
    orig = np.float32([
      self.l_eye,
      self.r_eye,
      self.mouth
    ])

    dest = self.__get_dest_points()
    matrix = cv2.getAffineTransform(orig, dest)

    self.img = cv2.warpAffine(self.img, matrix, tuple(reversed(self.img.shape[0:2])))
    self.l_eye, self.r_eye, self.mouth = dest

  def get_mouth_img(self):
    m = [int(i) for i in self.mouth]
    return self.img[m[1]-35:m[1]+35, m[0]-70:m[0]+70]

# convert_all_videos("./", "mov", "./results", verbose=1)