import argparse
from model.architeture import LCANet

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
  ap.add_argument("-l", "--logs_folder", required=False, help="Opção para salvar ogs do tensorboard. Caminho para a pasta de logs.")

  ap.add_argument("-s", "--skip_training", required=False, action="store_true", default=False)
  ap.add_argument("-e", "--evaluate_model", required=False, action="store_true", default=False)

  args = vars(ap.parse_args())
  print(args)

  model = LCANet(args["trained_model_path"])
  model.load_data(args["dataset_path"], args["alignment_path"], batch_size = args["batch_size"], validation_slice = 0.2, validation_only=args["skip_training"])

  if not args["skip_training"]:
    model.fit(args["epochs"], args["logs_folder"])
    model.save_model(args["save_model_path"])

  if args["evaluate_model"]:
    cer, wer = model.evaluate_model()
    print(f"CER: {cer}\nWER: {wer}")

if __name__ == '__main__':
	main()