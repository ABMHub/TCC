import argparse
from model.architeture import LCANet

def main():
  ap = argparse.ArgumentParser(
    prog="LCANet",
    description="Rede Neural de leitura labial automática",
    add_help=True
  )

  subparsers = ap.add_subparsers(dest="mode")
  train = subparsers.add_parser("train")

  train.add_argument("dataset_path", help="Caminho para a pasta com todos os videos.")
  train.add_argument("alignment_path", help="Caminho para a pasta com todos os alinhamentos.")
  train.add_argument("save_model_path", help="Caminho e nome do arquivo para salvar o modelo.")
  train.add_argument("batch_size", help='Tamanho de cada batch para o treinamento.', type=int)
  train.add_argument("epochs", help='Número de épocas para o treinamento.', type=int)

  train.add_argument("-m", "--trained_model_path", required=False, help="Opção para continuar treinamento prévio. Caminho para o modelo previamente treinado.")
  train.add_argument("-l", "--logs_folder", required=False, help="Opção para salvar logs do tensorboard. Caminho para a pasta de logs.")

  train.add_argument("-s", "--skip_evaluation", required=False, action="store_true", default=False, help='Opção para pular geração de métricas "CER" e "WER"')

  test = subparsers.add_parser("test")

  test.add_argument("dataset_path", help="Caminho para a pasta com todos os videos.")
  test.add_argument("alignment_path", help="Caminho para a pasta com todos os alinhamentos.")
  test.add_argument("trained_model_path", help="Caminho para o modelo treinado.")
  test.add_argument("batch_size", help='Tamanho de cada batch para o teste.', type=int)

  preprocess = subparsers.add_parser("preprocess")

  preprocess.add_argument("dataset_path", help="Caminho para os vídeos crus. Caso a extração de bocas seja ignorada, é o caminho para os vídeos das bocas em .npz.")
  preprocess.add_argument("results_folder", help="Caminho para a pasta onde o dataset processado estará. Será criada a pasta npz_mouths e single_words")
  preprocess.add_argument("-sm", "--skip_mouths", required=False, action="store_true", help="Opção para pular a extração de bocas.")
  preprocess.add_argument("-ss", "--skip_single_mouths", required=False, action="store_true", help="Opção para pular a geração de palavras soltas.")

  args = vars(ap.parse_args())
  # print(args)

  mode = args["mode"]

  if mode == "train" or mode == "test":
    model = LCANet(args["trained_model_path"])
    model.load_data(args["dataset_path"], args["alignment_path"], batch_size = args["batch_size"], validation_slice = 0.2, validation_only=(mode == "test"))

    if mode == "train":
      model.fit(args["epochs"], args["logs_folder"], args["save_model_path"] + "_best")
      model.save_model(args["save_model_path"])

    if mode == "test" or not args["skip_evaluation"]:
      cer, wer = model.evaluate_model()
      print(f"CER: {cer}\nWER: {wer}")

  elif mode == "preprocess":
    pass

if __name__ == '__main__':
	main()