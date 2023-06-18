from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState, ctc_decoder
import torch
import nltk

from multiprocessing import Pool

import numpy as np
predictions = np.ndarray

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

chars = dict()
chars[26] = " "
chars[27] = "_"
for i in range(26):
  chars[i] = chr(i + 97)

class NgramLM(CTCDecoderLM):
  """Create a Python wrapper around `language_model` to feed to the decoder."""
  def __init__(self, language_model):
    CTCDecoderLM.__init__(self)
    self.language_model = language_model
    self.sil = 28
    self.states = {}

    self.chars = chars
    self.chars[28] = "<s>"
        
  def start(self, start_with_nothing: bool = False):
    self.states = {}
    state = CTCDecoderLMState()

    ch = self.chars[self.sil]
    score = self.language_model.logscore(ch)

    self.states[state] = ([ch], score)
    return state

  def score(self, state: CTCDecoderLMState, token_index: int):
    outstate = state.child(token_index)
    if outstate not in self.states:
      parents = self.states[state][0][-4:]
      ch = self.chars[token_index]
      score = self.language_model.score(ch, parents)
      self.states[outstate] = (parents + [ch], score)

    score = self.states[outstate][1]

    return outstate, score

  def finish(self, state: CTCDecoderLMState):
    ret = self.score(state, self.sil)
    return ret
  
class NgramDecoder: # salvar e carregar com pickle
  def __init__(self, train_sentences : list[str], n = 5):
    sentences = [[*s] for s in train_sentences]
    train, padded = nltk.lm.preprocessing.padded_everygram_pipeline(5, sentences)
    m = nltk.lm.MLE(5)
    m.fit(train, padded)

    self.LM = NgramLM(m)
    self.decoder = ctc_decoder(
      None,
      list("abcdefghijklmnopqrstuvwxyz -|"),
      beam_size = 200,
      lm=self.LM,
      lm_weight=0.5
    )

  def __call__(self, preds : list[list[float]]):
    return self.decoder(torch.from_numpy(preds))
  
def ctc_decode_multiprocess(batches : list[predictions], workers : int, strings : list[str] = None, greedy = False) -> list[list[int]]:
  """_summary_

  Args:
      batches (list[predictions]): each element of the list will be processed in a different python process
      workers (int): number of maximum simultaneous python processes
      strings (list[str]): strings to train ngram model. If None, decoder will not use a language model

  Returns:
      list: list of dimensions [batch, prediction, letter_index]
  """
  if greedy:
    sentences = []
    for b in batches:
      for pred in b:
        sentence = np.argmax(pred, -1)
        sentences.append([[[sentence]]])
    return sentences
  
  pool = Pool(workers)
  return pool.map(_decode_wrapper, list(zip(batches, [strings]*len(batches)))) # talvez usar o chunksize ao inves de passar batches

def _decode_wrapper(a):
  return NgramDecoder(a[1], n=5)(a[0])