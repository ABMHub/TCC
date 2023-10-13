from typing import List
import math

SILENCE = ["sil", "sp"]

class Align:
  def __init__(self, align_path : str = None):
    if align_path is not None:
      ret = self.read_file(align_path)

      self.start    : list[int] = [math.floor(elem/1000) for elem in ret[0]]
      self.stop     : list[int] = [math.ceil(elem/1000) for elem in ret[1]]
      self.sentence : list[str] = ret[2]

      self.length = len(self.start)

      self.number_string : list[int] = self.sentence2number(self.sentence)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, f"Align index {index} is out of range"
    return self.start[index], self.stop[index], self.sentence[index]
  
  def get_sub_sentence(self, index, size):
    if size > 6:
      return 0, 74, self.number_string
    
    beg = self.start[index]         if index > 0      else 0
    end = self.stop[index+(size-1)] if index+size < 6 else 74
    
    y = self.sentence[index:index+size]

    cp = Align()

    cp.start = self.start
    cp.stop = self.stop
    cp.sentence = y
    cp.length = size
    cp.number_string = self.sentence2number(cp.sentence)

    return beg, end, cp

  @staticmethod
  def sentence2number(sentence : list[str]) -> list[int]:
    char_string = []
    for wrd in sentence:
      for char in wrd:
        if ord("a") <= ord(char) <= ord('z'):
          char_string.append((ord(char) - ord('a'))) # converte letra a letra para numero
      char_string.append(26)

    return char_string[:-1]

  @staticmethod
  def read_file(file_path) -> tuple:
    sentence = []
    start_time = []
    stop_time = []
    f = open(file_path)
    lines = f.readlines() # timestamp timestamp palavra

    for line in lines:
      start, stop, wrd = line.split() # palavra
      if wrd not in SILENCE:
        start_time.append(int(start))
        stop_time.append(int(stop)) # fazer uma struct
        sentence.append(wrd)

    return start_time, stop_time, sentence

  @staticmethod
  def add_padding(char_string : List[int], expected_size : int):
    return char_string + [-2] + [-1]*((expected_size) - len(char_string))