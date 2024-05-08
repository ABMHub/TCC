import seaborn as sns
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from generator.post_processing import MouthOnlyCentroid, MouthJP
from generator.align_processing import Align
import tqdm
import math

# folder = lambda a : os.path.join("C:/Users/lucas/OneDrive", a)
folder = lambda a : os.path.join("./visualization", a)
path = "D:/Documentos/GRIDcorpus/timeseries_aligned/npz_mouths"
raw_path = "D:/Documentos/GRIDcorpus/raw"
align_path = "D:/Documentos/GRIDcorpus/aligns"
# path = "/mnt/d/Documentos/GRIDcorpus/timeseries_aligned/npz_mouths"

loader = lambda path: np.load(path)["arr_0"]

files = os.listdir(path)

raw_data = loader(os.path.join(path, files[0]))
_, data_moc = MouthOnlyCentroid()(raw_data, None)
_, data_jp = MouthJP()(raw_data, None)

def mkdir(path):
  try:
    os.mkdir(path)
  except:
    pass

def line_graph(data):
  data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))                  
  data_dict = {}
  for p in range(data.shape[1]):
    name = f"{p//2} "
    if p % 2 == 0:
      name += "x"
    else:
      name += "y"
    
    data_dict[name] = data[:, p]

  data_dict = pd.DataFrame(data_dict)
  data_dict.index = list(range(75))

  sns.lineplot(data=data_dict)
  plt.savefig(folder(f"./line_graph.png"))
  plt.close()

def correlation(data, xy : bool, final = "", save = True):
  if xy:
    channel_names = ["x", "y"]
    shape_index = -1

  else:
    channel_names = ["1-velx", "2-vely", "3-magvel", "4-tan", "5-cos", "6-sin", "7-dv", "8-dtheta", "9-logcur", "10-centr", "11-ctotal", "12-c"]
    shape_index = 1

  coefs = []
  for channel in range(data.shape[shape_index]):
    if xy:
      coef_matrix = np.abs(np.corrcoef([[data[:, i, channel]][0] for i in range(data.shape[1])]))

    else:
      coef_matrix = np.abs(np.corrcoef([[data[:, channel, i]][0] for i in range(data.shape[-1])]))

    if save is True:
      sns.heatmap(coef_matrix, vmin=0, vmax=1)
      plt.savefig(folder(f"./{channel_names[channel]}_heatmap.png"))
      plt.close()

    coefs.append(coef_matrix)

  mean = np.mean(coefs, axis=0)
  if save is True:
    sns.heatmap(np.mean(coefs, axis=0), vmin=0, vmax=1)
    plt.savefig(folder(f"./{final}heatmap_mean.png"))
    plt.close()

  coefs.append(mean)

  return np.array(coefs)

def correlation_person(video_folder, columns):
  ref_path = "D:/Documentos/GRIDcorpus/raw"
  for i in range(1, 35):
    if i != 21:
      dir_list = os.listdir(os.path.join(ref_path, f"s{i}", "video", "mpg_6000"))
      coefs = []
      for vid in dir_list:
        try:
          raw_data = loader(os.path.join(video_folder, vid.split(".")[0] + ".npz"))
        except Exception as e:
          # print(e)
          continue
        _, data_jp = MouthJP()(raw_data, None)
        coefs.append(np.mean(correlation_features(data_jp, columns, None, False), axis=0))
      
      coef_matrix = np.array(coefs)
      print(coef_matrix.shape)
      sns.heatmap(np.mean(coef_matrix, axis=0), vmin=0, vmax=1)
      plt.savefig(folder(f"./teste_jp_s{i}_heatmap_mean.png"))
      plt.close()

def correlation_person_xy(video_folder, columns):
  ref_path = "D:/Documentos/GRIDcorpus/raw"
  person_heat = []
  for i in range(1, 35):
    if i != 21:
      dir_list = os.listdir(os.path.join(ref_path, f"s{i}", "video", "mpg_6000"))
      coefs = []
      for vid in dir_list:
        try:
          raw_data = loader(os.path.join(video_folder, vid.split(".")[0] + ".npz"))
        except Exception as e:
          # print(e)
          continue
        _, data_jp = MouthOnlyCentroid()(raw_data, None)
        corr = np.array(correlation(data_jp, columns, None, False))[:-1]
        coefs.append(np.mean(corr, axis=0))
      
      coef_matrix = np.mean(coefs, axis=0)

      person_heat.append(coef_matrix)
      sns.heatmap(coef_matrix, vmin=0, vmax=1)
      plt.savefig(folder(f"./teste_xy_s{i}_heatmap_mean.png"))
      plt.close()

  return person_heat

def correlation_features(data, channel_names : list = None, final = "", save = True):
  if channel_names is None:
    channel_names = ["1-velx", "2-vely", "3-magvel", "4-tan", "5-cos", "6-sin", "7-dv", "8-dtheta", "9-logcur", "10-centr", "11-ctotal", "12-c"]
  coefs = []
  for channel in range(data.shape[1]):
    coef_matrix = np.abs(np.corrcoef([[data[:, channel, i]][0] for i in range(data.shape[-1])]))

    if save is True:
      sns.heatmap(coef_matrix, vmin=0, vmax=1)
      plt.savefig(folder(f"./teste{channel_names[channel]}_heatmap.png"))
      plt.close()

    coefs.append(coef_matrix)

  mean = np.mean(coefs, axis=0)
  if save is True:
    sns.heatmap(np.mean(coefs, axis=0), vmin=0, vmax=1)
    plt.savefig(folder(f"./teste{final}heatmap_mean.png"))
    plt.close()

  coefs.append(mean)

  return coefs

def correlation_video(matrix, data):
  x_coef, y_coef, gmean = matrix
  np.fill_diagonal(x_coef, 0)
  np.fill_diagonal(y_coef, 0)
  np.fill_diagonal(gmean, 0)
  video_dims : tuple = (400, 400)

  data = data[:, 48:]
  data = np.array(data)
  data -= np.reshape(data.min(axis=(0, 1)), (1, 1, 2))

  coords = (data/np.reshape(data.max(axis=(0, 1)), (1, 1, 2)) * 300 + 50).astype(np.int32)
  
  out_raw = cv2.VideoWriter(folder('mouth_landmarks.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, video_dims)
  out = cv2.VideoWriter(folder('mouth_corr.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, video_dims)
  out_th = cv2.VideoWriter(folder('mouth_corr_th.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, video_dims)
  out_g = cv2.VideoWriter(folder('mouth_gmean.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, video_dims)

  for i in range(75):
    frame = np.zeros((*video_dims, 3), np.uint8)
    for j in range(coords.shape[1]):
      frame = np.array(cv2.circle(frame, coords[i][j], 1, (255, 255, 255), 5))

    frame_th = frame.copy()
    frame_raw = frame.copy()
    frame_g = frame.copy()

    for j in range(x_coef.shape[1]):
      for k in range(x_coef.shape[0]):
        if float(x_coef[j][k]) > 0.9:
          frame_th = cv2.line(frame_th, coords[i][j], coords[i][k], (255, 255, 0), 2)

    for j in range(y_coef.shape[1]):
      for k in range(y_coef.shape[0]):
        if y_coef[j][k] > 0.99:
          frame_th = cv2.line(frame_th, coords[i][j], coords[i][k], (255, 0, 0), 2)

    for j in range(x_coef.shape[1]):
      k = np.argmax(x_coef[j])
      frame = cv2.line(frame, coords[i][j], coords[i][k], (255, 0, 0), 2)

    for j in range(x_coef.shape[1]):
      k = np.argmax(y_coef[j])
      frame = cv2.line(frame, coords[i][j], coords[i][k], (0, 255, 0), 2)

    for j in range(gmean.shape[1]):
      k = np.argmax(gmean[j])
      frame = cv2.line(frame_g, coords[i][j], coords[i][k], (0, 255, 0), 2)


    out_raw.write(frame_raw) 
    out.write(frame) 
    out_th.write(frame_th) 
    out_g.write(frame_g)
    # cv2.imshow('Frame', frame) 

  out_raw.release()
  out.release()
  out_th.release()
  out_g.release()
  cv2.destroyAllWindows() 

def get_word_stats(folder):
  people = os.listdir(folder)
  people_words = []
  for person in people:
    path = os.path.join(folder, person, "video", "mpg_6000")
    vids = os.listdir(path)
    person_words = [{}, {}, {}, {}, {}, {}]
    for vid in vids:
      vid_name = vid.split(".")[0]
      for letter in range(len(vid_name)):
        char = vid_name[letter]
        if char in person_words[letter]:
          person_words[letter][char] += 1
        else:
          person_words[letter][char] = 1

    people_words.append(person_words)
  return people_words

def get_words_corr(vid_folder, align_folder, chosen_words_small, chosen_words_big, vid_list = None, person_number = None, xy = True, save = True):
  vids = os.listdir(vid_folder)
  corr_list = [[], [], [], [], [], []]
  for vid in tqdm.tqdm(vids, desc=""):
  # for vid in vids:
    vid_name = vid.split(".")[0]
    if vid_list is not None and (vid_name) + ".mpg" not in vid_list:
      continue
    for i in range(len(chosen_words_small)):
      if vid_name[i] == chosen_words_small[i]:
        align = Align(os.path.join(align_folder, vid_name + ".align"))
        b, e, _ = align.get_sub_sentence(i, 1)

        path = os.path.join(vid_folder, vid)
        raw_data = loader(path)[b:e]

        if xy:
          _, data = MouthOnlyCentroid()(raw_data, None)
        else:
          _, data = MouthJP()(raw_data, None)

        c = correlation(data, xy, save=False)
        c[c ==  np.inf] = 0
        c[c == -np.inf] = 0
        c[np.isnan(c)]  = 0

        corr_list[i].append(c)

        # _, data_jp  = MouthJP()(raw_data, None)
        # columns = ["1-velx", "2-vely", "3-magvel", "4-tan", "5-cos", "6-sin", "7-dv", "8-dtheta", "9-logcur", "10-centr", "11-ctotal", "12-c"]
        # correlation(data_jp, columns, "jp_")

  corrs = []
  for i in range(len(chosen_words_big)):
    if len(corr_list[i]) > 0:
      corr_mean = np.mean(corr_list[i][2], axis=0)
      corrs.append(corr_mean)

      if save:
        sns.heatmap(corr_mean, vmin=0, vmax=1)
        plt.savefig(folder(f"./0304/{person_number}/word_{chosen_words_big[i]}_heatmap.png"))
        plt.close()

  return corrs

def boxplots(persons : np.ndarray):
  print(np.array(persons).shape)
  for i in range(len(persons)):
    persons[i] = persons[i].flatten()

  x = []
  for i in range(1, 35):
    if i != 21:
      x += [i]*400

  persons_pd = {
    "person": x,
    "dataa": np.array(persons).flatten()
  }

  print(len(persons_pd["person"]))
  print(len(persons_pd["dataa"]))

  sns.boxplot(data = pd.DataFrame(persons_pd), x="person", y="dataa")
  plt.savefig(folder(f"./boxplot_xy_heatmap.png"))
  plt.close()

  sns.boxplot(data = pd.DataFrame(persons_pd), x="dataa", y="person")
  plt.savefig(folder(f"./boxplot_xy_heatmap_inv.png"))
  plt.close()

def get_word_person_corr(folder, align, chosen_words_small, chosen_words_big, person, save = True):
  ref_path = "D:/Documentos/GRIDcorpus/raw"
  dir_list = os.listdir(os.path.join(ref_path, f"s{person}", "video", "mpg_6000"))
  return get_words_corr(folder, align, chosen_words_small, chosen_words_big, dir_list, person, xy = False, save=save)

def psnr(img1, img2):
  mse = np.mean(np.square(np.subtract(img1, img2)))
  # if mse == 0:
    # return np.Inf
  # PIXEL_MAX = 1
  # ret = 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)  
  return mse

def gen_mean_heatmaps(corrs):
  number_of_people, number_of_words, _, _ = np.shape(corrs)
  matrix_person = []
  for i in range(number_of_people):
    psnr_person_list = []
    for j in range(number_of_people):
      psnr_list = []
      for ik in range(number_of_words):
        for jk in range(ik, number_of_words):
          psnr_list.append(psnr(corrs[i][ik], corrs[j][jk]))

      psnr_person_list.append(np.mean(psnr_list))
    matrix_person.append(psnr_person_list)

  matrix_word = []
  for i in range(number_of_words):
    psnr_word_list = []
    for j in range(number_of_words):
      psnr_list = []
      for ik in range(number_of_people):
        for jk in range(ik, number_of_people):
          psnr_list.append(psnr(corrs[ik][i], corrs[jk][j]))

      psnr_word_list.append(np.mean(psnr_list))
    matrix_word.append(psnr_word_list)

  matrix_person = np.array(matrix_person)
  matrix_word = np.array(matrix_word)

  matrix_person -= matrix_person.min()
  matrix_person /= matrix_person.max()
  matrix_person = 1 - matrix_person

  matrix_word -= matrix_word.min()
  matrix_word /= matrix_word.max()
  matrix_word = 1 - matrix_word

  sns.heatmap(matrix_person, vmin=0)
  plt.title("Correlação entre pessoas diferentes para palavras p__z7a")
  plt.savefig(folder(f"./1004/matrix_person_heatmap.png"))
  plt.close()

  sns.heatmap(matrix_word, vmin=0)
  plt.title("Correlação entre palavras para as 10 primeiras pessoas")
  plt.xticks([0.5, 1.5, 2.5, 3.5], ["please", "z", "7", "again"])
  plt.yticks([0.5, 1.5, 2.5, 3.5], ["please", "z", "7", "again"])
  plt.savefig(folder(f"./1004/matrix_word_heatmap.png"))
  plt.close()

  matrix_person_bool = np.zeros(matrix_person.shape)
  for i in range(len(matrix_person)):
    matrix_person_bool[i][matrix_person[i].argmax()] = True

  matrix_word_bool = np.zeros(matrix_word.shape)
  for i in range(len(matrix_word)):
    matrix_word_bool[i][matrix_word[i].argmax()] = True

  sns.heatmap(matrix_person_bool, vmin=0)
  plt.title("Máxima correlação entre pessoas diferentes para palavras p__z7a")
  plt.savefig(folder(f"./1004/matrix_person_heatmap_bool.png"))
  plt.close()

  sns.heatmap(matrix_word_bool, vmin=0)
  plt.title("Máxima correlação entre palavras para as 10 primeiras pessoas")
  plt.xticks([0.5, 1.5, 2.5, 3.5], ["please", "z", "7", "again"])
  plt.yticks([0.5, 1.5, 2.5, 3.5], ["please", "z", "7", "again"])
  plt.savefig(folder(f"./1004/matrix_word_heatmap_bool.png"))
  plt.close()

def gen_all_heatmaps(corrs):
  number_of_people, number_of_words, _, _ = np.shape(corrs)
  matrix_person = []
  for i in range(number_of_people):
    psnr_person_list = []
    for j in range(number_of_people):
      psnr_list = []
      for ik in range(number_of_words):
        for jk in range(ik, number_of_words):
          psnr_list.append(psnr(corrs[i][ik], corrs[j][jk]))

      psnr_person_list.append(psnr_list)
    matrix_person.append(psnr_person_list)

  matrix_word = []
  for i in range(number_of_words):
    psnr_word_list = []
    for j in range(number_of_words):
      psnr_list = []
      for ik in range(number_of_people):
        for jk in range(ik, number_of_people):
          psnr_list.append(psnr(corrs[ik][i], corrs[jk][j]))

      psnr_word_list.append(psnr_list)
    matrix_word.append(psnr_word_list)

  matrix_person = np.array(matrix_person)
  matrix_word = np.array(matrix_word)

  word_list = ["please", "z", "7", "again"]

  l = -1
  for i in range(number_of_words):
    for k in range(number_of_words - i):
      l += 1
      matrix_person_s = matrix_person[:, :, l]

      matrix_person_s -= matrix_person_s.min()
      matrix_person_s /= matrix_person_s.max()
      matrix_person_s = 1 - matrix_person_s

      sns.heatmap(matrix_person_s, vmin=0)
      plt.title(f"Correlação pessoas diferentes para as palavras {word_list[i]} e {word_list[i+k]}")
      plt.savefig(folder(f"./240417/matrix_person_heatmap_{i}_{i+k}.png"))
      plt.close()

      matrix_person_bool = np.zeros(matrix_person_s.shape)
      for j in range(len(matrix_person_s)):
        matrix_person_bool[j][matrix_person_s[j].argmax()] = True

      sns.heatmap(matrix_person_bool, vmin=0)
      plt.title(f"Máxima correlação entre pessoas diferentes para as palavras {word_list[i]} e {word_list[i+k]}")
      plt.savefig(folder(f"./240417/matrix_person_heatmap_bool_{i}_{i+k}.png"))
      plt.close()


  l = -1
  for i in range(number_of_people):
    for k in range(number_of_people - i):
      l += 1
      matrix_word_s = matrix_word [:, :, l]

      matrix_word_s -= matrix_word_s.min()
      matrix_word_s /= matrix_word_s.max()
      matrix_word_s = 1 - matrix_word_s

      sns.heatmap(matrix_word_s, vmin=0)
      plt.title(f"Correlação das palavras entre as pessoas {i} e {i+k}")
      plt.xticks([0.5, 1.5, 2.5, 3.5], word_list)
      plt.yticks([0.5, 1.5, 2.5, 3.5], word_list)
      plt.savefig(folder(f"./240417/matrix_word_heatmap_{i}_{i+k}.png"))
      plt.close()

      matrix_word_bool = np.zeros(matrix_word_s.shape)
      for j in range(len(matrix_word_s)):
        matrix_word_bool[j][matrix_word_s[j].argmax()] = True

      sns.heatmap(matrix_word_bool, vmin=0)
      plt.title("Máxima correlação entre palavras para as 10 primeiras pessoas")
      plt.xticks([0.5, 1.5, 2.5, 3.5], word_list)
      plt.yticks([0.5, 1.5, 2.5, 3.5], word_list)
      plt.savefig(folder(f"./240417/matrix_word_heatmap_bool_{i}_{k}.png"))
      plt.close()

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from visualization.cuda_dtw import DTW

def get_all_word_features(path, align_folder, raw_path, padding = False, flatten=False):
  d = {
    "data": [],
    "word": [],
    "person": [],
  }

  g1 = 0
  g2 = 1e8

  for person in tqdm.tqdm(range(5)):
    if person == 20: continue

    vids = os.listdir(path)
    dir_list = os.listdir(os.path.join(raw_path, f"s{person+1}", "video", "mpg_6000"))

    for vid in vids:
      vid_name = vid.split(".")[0]

      if vid_name + ".mpg" not in dir_list: continue

      align = Align(os.path.join(align_folder, vid_name + ".align"))
      
      vid_path = os.path.join(path, vid)
      raw_data = loader(vid_path)

      for i in range(6):
        b, e, cp = align.get_sub_sentence(i, 1)

        size = e-b

        if size < 3: continue

        cut_data = raw_data[b:e]

        _, data = MouthJP()(cut_data, None)

        s = np.shape(data)
        # print("s", s)

        if flatten:
          d["data"].append(np.array(data).flatten())
        else:
          d["data"].append(np.array(data))

        d["word"].append(cp.sentence[0])
        d["person"].append(person)

        g1 = max(g1, len(d["data"][-1]))
        g2 = min(g2, len(d["data"][-1]))

  # d["data"] = np.array(d["data"],  dtype=object)

  if padding:
    # pad_size = np.shape(d["data"][0][0])
    # zeros = np.zeros(pad_size)
    for i in range(len(d["data"])):
      d["data"][i] = np.pad(d["data"][i], pad_width=(0, g1 - len(d["data"][i])), mode="constant", constant_values=0)

  # d["data"] = np.array(d["data"])

  d["g"] = g2
  d["word"] = np.array(d["word"])
  d["person"] = np.array(d["person"])
  d["data"] = np.array(d["data"], dtype=object)

  return d

import torch

def build_tsne(metric = "euclidian"):
  d = get_all_word_features(path, align_path, raw_path)

  # fi = (((d["word"] == "z") | (d["word"] == "again") | (d["word"] == "please") | (d["word"] == "seven")))
  # fi = (d["person"] < 5)

  # d["data"] = d["data"][fi]
  # d["word"] = d["word"][fi]
  # d["person"] = d["person"][fi]

  data_amount = 4000
  # amount_filter = np.concatenate((np.ones(data_amount, dtype=bool), np.zeros(len(d["word"])-data_amount, dtype=bool)))
  amount_filter = np.random.choice(range(len(d["data"])), data_amount, False)

  d["data"] = d["data"][amount_filter]
  d["word"] = d["word"][amount_filter]
  d["person"] = d["person"][amount_filter]

  p = PCA(3, svd_solver="auto", whiten=True, random_state=42)

  # print(np.shape(d["data"]))
  n_m = []
  for i in range(len(d["data"])):
    data_in = d["data"][i]
    # print("in", data_in.shape)
    n_m.append([a.flatten() for a in data_in])
    # data_in = np.swapaxes(data_in, 0, 1)
    # print(data_in.shape)
    # n_m.append(data_in.flatten())
    # print(n_m[-1].shape)
  

  # n_m = p.fit_transform(d["data"])
  n_m = np.array(n_m, object)
  print(np.shape(n_m[0]))

  print(np.shape(n_m))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  dtw = DTW(True)

  def metric(a, b):
    # return fastdtw(n_m[int(a[0])], n_m[int(b[0])], radius=1)[0]
    a = torch.FloatTensor([n_m[int(a[0])]])
    a.to(device)
    b = torch.FloatTensor([n_m[int(b[0])]])
    b.to(device)
    a = a.cuda()
    b = b.cuda()
    return dtw.forward(a, b)[0]

  t1 = TSNE(verbose=10, n_jobs=-1, random_state=42, metric=metric)
  # r1 = t1.fit_transform([[i] for i in range(len(n_m))], d["word"])
  # r1 = t1.fit_transform(np.array(range(100))*np.ones((100, 100)))

  # t2 = TSNE(verbose=1, n_jobs=-1, random_state=42, metric=metric)
  r1 = t1.fit_transform([[i] for i in range(len(n_m))], d["person"])

  return r1, None, d



# mkdir("visualization")
# line_graph(data_moc)
# matrix_moc = correlation(data_moc, ["x", "y"])
# columns = ["1-velx", "2-vely", "3-magvel", "4-tan", "5-cos", "6-sin", "7-dv", "8-dtheta", "9-logcur", "10-centr", "11-ctotal", "12-c"]
# correlation(data_jp, columns, "jp_")
# # correlation_features(data_jp, ["1-velx", "2-vely", "3-magvel", "4-tan", "5-cos", "6-sin", "7-dv", "8-dtheta", "9-logcur", "10-centr", "11-ctotal", "12-c"], "jp_")
# correlation_video(matrix_moc, raw_data)
# # correlation_person(path, columns)
# persons = correlation_person_xy(path, ["x", "y"])
# boxplots(persons)
  
# stats = get_word_stats(raw_path)
# print(get_word_stats(raw_path)[0])
# [print(stat[2]) for stat in stats]
# get_words_corr(path, align_path, ["p", "r", "w", "z", "7", "a"], ["place", "red", "with", "z", "seven", "again"])
# number_of_people = 10
# corrs = []
# for i in range(1, number_of_people + 1):
#   corrs.append(get_word_person_corr(path, align_path, ["p", "", "", "z", "7", "a"], ["place", "red", "with", "z", "seven", "again"], i, save=False))

# corrs eh uma lista de tamanho n_pessoas
# que eh uma lista de tamanho n_palavras
# que eh uma matriz de correlacao

# gen_mean_heatmaps(corrs)
# gen_all_heatmaps(corrs)
# sns.scatterplot(x = r1[:, 0], y = r1[:, 1], hue=dic["person"])
# plt.savefig(folder(f"./240422/matrix.png"))
# plt.close()

# metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]
# for metric in metrics:
#   r1, r2, dic = build_tsne(metric)

#   plt.figure(figsize=(12, 12))
#   sns.scatterplot(x = r1[:, 0], y = r1[:, 1], hue=dic["person"])
#   plt.savefig(folder(f"./240422/person_limited_{metric}.png"))
#   plt.close()

# from dtw import *
# from fastdtw import fastdtw

# def fdtw(x, y, **kwargs):
#   # print("debug")
#   return fastdtw(x, y, radius=1)[0]
# metric = lambda x, y, **kwargs : fastdtw(x, y, radius=1 **kwargs)[0]
# metric = lambda x, y, **kwargs : dtw(x, y, **kwargs).distance
# metric = fastdtw


r1, r2, dic = build_tsne(None)
plt.figure(figsize=(12, 12))
sns.scatterplot(x = r1[:, 0], y = r1[:, 1], hue=dic["person"])
plt.savefig(folder(f"./240506/person_limited_{'dtw'}.png"))
plt.close()



  # plt.figure(figsize=(12, 12))
  # sns.scatterplot(x = r2[:, 0], y = r2[:, 1], hue=dic["word"])
  # plt.savefig(folder(f"./240422/words_limited_{metric}.png"))
  # plt.close()