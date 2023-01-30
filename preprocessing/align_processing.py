import os
from typing import List

__SILENCE = ["sil"]

alignment = list[int]

def sentence2number(sentence : list[str]) -> alignment:
  char_string = []
  for wrd in sentence:
    for char in wrd:
      if ord("a") <= ord(char) <= ord('z'):
        char_string.append((ord(char) - ord('a'))) # converte letra a letra para numero
    char_string.append(26)

  return char_string[:-1]

def read_file(path : str) -> list[str]:
  sentence = []
  f = open(path)
  lines = f.readlines() # timestamp timestamp palavra

  for line in lines:
    wrd = line.split()[-1] # palavra
    if wrd not in __SILENCE:
      sentence.append(wrd)
  
  return sentence

def process_folder(path : str):
  d = dict()
  list_dir = os.listdir(path)

  for file in list_dir:
    if file.endswith(".align"):
      d[file] = sentence2number(read_file(os.path.join(path, file)))
      
  return d

def add_padding(char_string : List[int], expected_size : int):
  return char_string + [-2] + [-1]*((expected_size) - len(char_string))
    
          