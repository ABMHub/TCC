from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState, ctc_decoder
import torch
import nltk
import os

from multiprocessing import Pool

import numpy as np
predictions = np.ndarray

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

chars = dict()
chars[26] = " "
chars[27] = "_"
chars[28] = ""
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
  def __init__(self, train_sentences : list[str], n = 5, lm = True):
    self.LM = None
    if lm:
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
      lm_weight=0.5 if self.LM is not None else 0
    )

  def __call__(self, preds : list[list[float]]):
    return self.decoder(torch.from_numpy(preds))
  
class Decoder():
  def __init__(self):
    self.name = "Empty"

class NgramCTCDecoder(Decoder):
  def __init__(self):
    self.name = 'ngram ctc decoder'

  def __call__(self, batches : list[predictions], workers : int, strings : list[str] = None, greedy = False) -> list[list[int]]:
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
    ret = pool.map(self._decode_wrapper, list(zip(batches, [strings]*len(batches), [True]*len(batches)))) # talvez usar o chunksize ao inves de passar batches

    decoded = []
    for vec in ret:
      decoded += list(vec)

    sentences = []
    for p in decoded:
      sentence = []
      for chr in p[0][0]:
        sentence.append(chars[int(chr)])

      sentences.append("".join(sentence))
    
    return sentences

  def _decode_wrapper(self, a):
    return NgramDecoder(a[1], n=5, lm = a[2])(a[0])
  
class RawCTCDecoder(Decoder):
  def __init__(self):
    self.name = 'raw ctc decoder'

  def __call__(self, batches : list[predictions], workers : int, strings : list[str] = None, greedy = False) -> list[list[int]]:
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
    ret =  pool.map(self._decode_wrapper, list(zip(batches, [strings]*len(batches), [False]*len(batches)))) # talvez usar o chunksize ao inves de passar batches

    decoded = []
    for vec in ret:
      decoded += list(vec)

    sentences = []
    for p in decoded:
      sentence = []
      for chr in p[0][0]:
        sentence.append(chars[int(chr)])

      sentences.append("".join(sentence))
    
    return sentences

  def _decode_wrapper(self, a):
    return NgramDecoder(a[1], n=5, lm = a[2])(a[0])

import re
import string
from collections import Counter

# Spell decoding from https://github.com/rizkiarm/LipNet

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

class Spell(object):
    def __init__(self, path):
        self.dictionary = Counter(list(string.punctuation) + self.words(open(path).read()))

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    # Correct words
    def corrections(self, words):
        return [self.correction(word) for word in words]

    # Correct sentence
    def __call__(self, sentence):
        return untokenize(self.corrections(tokenize(sentence)))
    
class SpellCTCDecoder(Decoder):
  def __init__(self):
    print(os.getcwd())
    self.name = 'spell ctc decoder'
    self.path = os.path.join(os.path.dirname(__file__), "..", "util/grid.txt")
    self.obj = Spell(self.path)

  def __call__(self, batches : list[predictions], workers : int, strings : list[str] = None, greedy = False) -> list[list[int]]:
    sentences = RawCTCDecoder()(batches, workers, strings, greedy)
    
    for i in range(len(sentences)):
      sentences[i] = self.obj(sentences[i])

    return sentences
    # pool = Pool(workers)
    # return pool.map(self._decode_wrapper, list(zip(batches, [strings]*len(batches), [False]*len(batches)))) # talvez usar o chunksize ao inves de passar batches

  def _decode_wrapper(self, a):
    return Spell(self.path)(a[0])