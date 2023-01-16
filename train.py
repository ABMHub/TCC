import argparse
from architeture import get_model, compile_model
from keras.models import load_model
from keras.callbacks import TensorBoard
import datetime
import os
from preprocessing.preprocessing import get_training_data

argparse.ArgumentParser

def main():
  ap = argparse.ArgumentParser(
    prog="LCANet",
    description="Rede Neural de leitura labial automática",
    add_help=True
  )

  ap.add_argument("dataset_path", help="Caminho para a pasta com todos os videos.")
  ap.add_argument("alignment_path", help="Caminho para a pasta com todos os alinhamentos.")
  ap.add_argument("save_model_path", help="Caminho e nome do arquivo para salvar o modelo.")
  ap.add_argument("batch_size", help='Tamanho de cada batch para o treinamento.', type=int)
  ap.add_argument("epochs", help='Número de épocas para o treinamento.', type=int)

  ap.add_argument("-m", "--trained_model_path", required=False, help="Opção para continuar treinamento prévio. Caminho para o modelo previamente treinado.")
  ap.add_argument("-l", "--logs_folder", required=False, help="Opção para salvar logs do tensorboard. Caminho para a pasta de logs.")

  args = vars(ap.parse_args())
  print(args)

  if args["trained_model_path"] is None:
    model = get_model()
  
  else:
    model = load_model(args["trained_model_path"])
    compile_model(model)

  callback_list = []
  if args["logs_folder"] is not None:
    tb = TensorBoard(os.path.join(args["logs_folder"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)
    callback_list.append(tb)

  data = get_training_data(args["dataset_path"], args["alignment_path"], batch_size = args["batch_size"], val_size = 0.2)
  model.fit(x=data["train"], validation_data=data["validation"], epochs = args["epochs"], callbacks=callback_list)

  model.save(args["save_model_path"])

if __name__ == '__main__':
	main()