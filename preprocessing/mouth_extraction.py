import dlib
import cv2
import numpy as np
from typing import List, Tuple
import os
import tqdm

ffd = dlib.get_frontal_face_detector()
lmd = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def __create_dir_recursively(path):
  while True:
    try:
      os.mkdir(path)
    except FileExistsError:
      return
    except FileNotFoundError:
      __create_dir_recursively("/".join(path.split("/")[:-1]))

def __get_all_videos(path, extension, dest_folder):
  orig_dest_videos = {}
  files_in = os.listdir(path)
  for file in files_in:
    curr_file_path = os.path.join(path,file)
    if os.path.isdir(curr_file_path):
      new_dest_folder = os.path.join(dest_folder, file)
      orig_dest_videos.update(__get_all_videos(curr_file_path, extension, new_dest_folder))

    if file.endswith(extension):
      new_video_path = os.path.join(dest_folder, ".".join(file.split(".")[:-1]) + ".avi")
      orig_dest_videos[curr_file_path] = new_video_path

  return orig_dest_videos

def convert_all_videos(path, extension, dest_folder, verbose = 1):
  orig_dest_videos = __get_all_videos(path, extension, dest_folder)

  for orig in tqdm.tqdm(orig_dest_videos, desc="Convertendo Video", disable=verbose<=0):
    if not os.path.isfile(orig_dest_videos[orig]):
      __create_dir_recursively("/".join(orig_dest_videos[orig].split("/")[:-1]))
      try:
        FaceVideo(orig, 0 if verbose < 2 else 1).get_mouth_video(orig_dest_videos[orig])
      except (IndexError, TypeError, ValueError):
        print(f"Video {orig} com erro")

class FaceVideo:
  def __init__(self, video_path : str, verbose = 1):
    self.video = cv2.VideoCapture(video_path)
    self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.frames = []
    self.verbose = verbose

    pbar = tqdm.tqdm(desc='Carregando video', total=self.frame_count, disable=self.verbose==0)
    while self.video.isOpened():
      _, frame = self.video.read()
      if frame is None: break

      self.frames.append(FaceFrame(frame))
      pbar.update()
    pbar.close()

  def get_mouth_video(self, path):
    dest_img = self.frames[0]
    self.dest = np.float32([
        dest_img.face_coords[0][0],
        [dest_img.face_coords[0][1][0], dest_img.face_coords[0][0][1]],
        dest_img.mouths_center[0]
      ])

    self.frames = [FaceFrame(frame_obj.transform(self.dest)) for frame_obj in tqdm.tqdm(self.frames, desc="Alinhando frames", disable=self.verbose==0)] 

    n_video = cv2.VideoWriter("./testetcc.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25, tuple(reversed(self.frames[0].img.shape[0:2])))
    for frame in self.frames:
      n_video.write(frame.img)
    n_video.release()

    self.mouth_imgs = [cv2.resize(frame_obj.get_mouth_img(), (100, 50)) for frame_obj in tqdm.tqdm(self.frames, desc="Adquirindo recortes da boca", disable=self.verbose==0)]

    n_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (100, 50))
    for frame in self.mouth_imgs:
      n_video.write(frame)
    n_video.release()

class FaceFrame:
  def __init__(self, img : np.ndarray):
    self.img = img
    self.faces = ffd(img, 0)
    self.face_coords = self.get_face()
    self.mouths, self.mouths_center = self.get_mouth_coords()

  def get_face(self) -> List[Tuple[Tuple[int]]]:
    if len(self.faces) != 1:
      raise ValueError()

    face_coords = []

    for face in self.faces:
      tl, br = face.tl_corner(), face.br_corner()

      tlx = max(tl.x, 0)
      tly = max(tl.y, 0)

      brx = min(br.x, self.img.shape[0])
      bry = min(br.y, self.img.shape[1])

      face_coords.append(((tlx, tly), (brx, bry)))

    return face_coords

  def get_mouth_coords(self) -> any:
    mouths_coords = []
    mouths_center = []
    for face in self.faces:
      lm = lmd(self.img, face)
      
      tl_mouth = [1e9, 1e9]
      br_mouth = [0, 0]
      x_list = []
      y_list = []

      for i in range(48, 60):
        x, y = lm.part(i).x, lm.part(i).y
        x_list.append(x)
        y_list.append(y)

        tl_mouth[0] = min(tl_mouth[0], x)
        tl_mouth[1] = min(tl_mouth[1], y)
        br_mouth[0] = max(br_mouth[0], x)
        br_mouth[1] = max(br_mouth[1], y)


      mouths_coords.append((tl_mouth, br_mouth))
      mouths_center.append((int(np.mean(x_list)), int(np.mean(y_list))))

    return mouths_coords, mouths_center
      
  def transform(self, dest):
    for i in range(len(self.faces)):
      orig = np.float32([
        self.face_coords[i][0],
        [self.face_coords[i][1][0], self.face_coords[i][0][1]],
        self.mouths_center[0]
      ])
      matrix = cv2.getAffineTransform(orig, dest)
      new_image = cv2.warpAffine(self.img, matrix, tuple(reversed(self.img.shape[0:2])))

      return new_image

  def get_mouth_img(self):
    m = self.mouths[0]
    return self.img[m[0][1]:m[1][1], m[0][0]:m[1][0]]

# convert_all_videos("./", "mov", "./results", verbose=1)