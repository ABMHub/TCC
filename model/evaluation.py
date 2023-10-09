import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from os.path import join

def bleu(references : list[str], predictions : list[str], multigram : bool = False) -> float:
  references_p = [[reference.split()] for reference in references]
  predictions_p = [prediction.split() for prediction in predictions]
  wheights = [0.25, 0.25, 0.25, 0.25]
  if not multigram:
    wheights = [1, 0, 0, 0]
  return np.mean([sentence_bleu(ref, pred, wheights) for ref, pred in zip(references_p, predictions_p)])

class Evaluation:
  def __init__(self):
    self.data = {
      "experiment_name": None,

      "architecture_name": None,
      "preprocessing_type": None,
      "postprocessing_type": None,

      "description": None,
      "datetime": None,

      "cer": None,
      "wer": None,
      "bleu": None,
      "bleu_multigram": None,

      "cer_nolm": None,
      "wer_nolm": None,
      "bleu_nolm": None,
      "bleu_multigram_nolm": None,

      "params": None,
      # "seconds_per_batch": None,
      # "batches_per_epoch": None,
      # "epochs_trained": None,
      "prediction_time": None,
    }

  def to_csv(self, folder_path : str):
    # todo salvar no modelo e um geral
    pd.DataFrame(data=[self.data], index=[0]).to_csv(join(folder_path, "info.csv"), mode="a", index=False)

  def from_csv(self, folder_path : str):
    df = pd.read_csv(join(folder_path, "info.csv"))
    self.data = df.replace(pd.NA, None).iloc[-1].to_dict()
    print(self.data)