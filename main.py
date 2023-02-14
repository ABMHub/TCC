import argparse

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
  train.add_argument("-c", "--calc_standardization", required=False, action="store_true", default=False, help='Opção para re-calcular a media e o desvio padrão do dataset, ao invés de usar os valores default.')

  test = subparsers.add_parser("test")

  test.add_argument("dataset_path", help="Caminho para a pasta com todos os videos.")
  test.add_argument("alignment_path", help="Caminho para a pasta com todos os alinhamentos.")
  test.add_argument("trained_model_path", help="Caminho para o modelo treinado.")
  test.add_argument("batch_size", help='Tamanho de cada batch para o teste.', type=int)

  test.add_argument("-c", "--calc_standardization", required=False, action="store_true", default=False, help='Opção para re-calcular a media e o desvio padrão do dataset, ao invés de usar os valores default.')

  preprocess = subparsers.add_parser("preprocess")

  preprocess.add_argument("dataset_path", help="Caminho para os vídeos crus. Caso a extração de bocas seja ignorada, é o caminho para os vídeos das bocas em .npz.")
  preprocess.add_argument("results_folder", help="Caminho para a pasta onde o dataset processado estará. Será criada a pasta npz_mouths e single_words")
  preprocess.add_argument("-ss", "--single_words", required=False, help="Path para a pasta de alignments. Habilita extração de palavras soltas.")
  preprocess.add_argument("-sm", "--skip_mouths", required=False, action="store_true", help="Opção para pular a extração de bocas.")

  args = vars(ap.parse_args())

  mode = args["mode"]

  if mode == "train" or mode == "test":
    from model.architeture import LCANet

    model = LCANet(args["trained_model_path"])
    model.load_data(
      x_path = args["dataset_path"],
      y_path = args["alignment_path"], 
      batch_size = args["batch_size"],
      validation_slice = 0.2,
      validation_only=(mode == "test"),
      recalc_standardization = args["calc_standardization"]
    )

    if mode == "train":
      model.fit(args["epochs"], args["logs_folder"], args["save_model_path"] + "_best")
      model.save_model(args["save_model_path"])

    if mode == "test" or not args["skip_evaluation"]:
      cer, wer = model.evaluate_model()
      print(f"CER: {cer}\nWER: {wer}")

  elif mode == "preprocess":
    from preprocessing.mouth_extraction import convert_all_videos_multiprocess
    from preprocessing.single_words import slice_all_videos_multiprocess
    import os

    raw_video_path = args["dataset_path"]
    mouths_path = os.path.join(args["results_folder"], "npz_mouths")
    single_words_path = os.path.join(args["results_folder"], "single_words")
    alignments_path = args["single_words"]

    if not args["skip_mouths"]:
      convert_all_videos_multiprocess(raw_video_path, ".mpg", mouths_path, True)

    else:
      mouths_path = raw_video_path

    if alignments_path is not None:
      slice_all_videos_multiprocess(mouths_path, alignments_path, "npz", single_words_path)

if __name__ == '__main__':
	main()