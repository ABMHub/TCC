from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState, ctc_decoder
import torch
import nltk

from multiprocessing import Pool

class NgramLM(CTCDecoderLM):
  """Create a Python wrapper around `language_model` to feed to the decoder."""
  def __init__(self, language_model):
    CTCDecoderLM.__init__(self)
    self.language_model = language_model
    self.sil = 28
    self.states = {}

    self.chars = dict()
    self.chars[26] = " "
    self.chars[28] = "<s>"
    for i in range(26):
      self.chars[i] = chr(i + 97)
        
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
  
def decode_multiprocess_multiprocess(sections, workers, strings):
  pool = Pool(workers)
  return pool.map(_f, list(zip(sections, [strings]*workers)))

def _f(a):
  return NgramDecoder(a[1], n=5)(a[0])