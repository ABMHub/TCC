import argparse
import os
import time

def main(args = None):
  ap = argparse.ArgumentParser(
    prog="LipReader",
    description="Rede Neural de leitura labial automática",
    add_help=True
  )

  subparsers = ap.add_subparsers(dest="mode")
  train = subparsers.add_parser("train")

  train.add_argument("dataset_path", help="Caminho para a pasta com todos os videos.", nargs="+")
  train.add_argument("alignment_path", help="Caminho para a pasta com todos os alinhamentos.")
  train.add_argument("save_model_path", help="Caminho e nome do arquivo para salvar o modelo.")
  train.add_argument("batch_size", help='Tamanho de cada batch para o treinamento.', type=int)
  train.add_argument("epochs", help='Número de épocas para o treinamento.', type=int)

  train.add_argument("-m", "--trained_model_path", required=False, help="Opção para continuar treinamento prévio. Caminho para o modelo previamente treinado.")
  # train.add_argument("-l", "--logs_folder", required=False, help="Opção para salvar logs do tensorboard. Caminho para a pasta de logs.")
  train.add_argument("-u", "--unseen_speakers", required=False, action="store_true", default=False, help='Opção para fazer testes com pessoas não vistas pelo treino')
  train.add_argument("-s", "--skip_evaluation", required=False, action="store_true", default=False, help='Opção para pular geração de métricas "CER", "WER" e "BLEU"')
  train.add_argument("-g", "--choose_gpu", required=False, help="Opção para escolher uma GPU específica para o teste ou treinamento.")
  train.add_argument("-a", "--architecture", required=False, default = "lipnet", help="Opção para escolher uma arquitetura diferente para treino. Opções: [lipnet, lcanet, bilstm, lipformer].")
  train.add_argument("-p", "--patience", required=False, default = 25, type=int, help="Paciencia para o early stopping.")

  # train.add_argument("-lm", "--landmark_features", required=False, action="store_true", default=False, help="Opção para habilitar passagem de landmark features para o modelo")
  # train.add_argument("-hf", "--half_frame", required=False, action="store_true", default=False, help="Opção para treinar e testar apenas com metade do rosto em cada quadro")
  # train.add_argument("-fs", "--frame_sample", required=False, default=1, type=float, help="Opção para treinar e testar com frames subamostrados")
  # train.add_argument("-lts", "--lazy_ts_mode", required=False, default=0, type=int, help="Modo de processamento para time series. 1: todos os pontos; 2: apenas boca; 3: boca normalizada pelo centroide")

  train.add_argument("-n", "--experiment_name", required=False, default = None, type=str, help="O nome do experimento, será inserido nos logs.")
  train.add_argument("-d", "--description", required=False, default = None, type=str, help="A descrição do experimento, será inserida nos logs.")
  train.add_argument("-pt", "--preprocessing_type", required=False, default = None, type=str, help="Uma curta descrição do pré-processamento utilizado, será inserido nos logs")

  test = subparsers.add_parser("test")

  test.add_argument("dataset_path", help="Caminho para a pasta com todos os videos.")
  test.add_argument("alignment_path", help="Caminho para a pasta com todos os alinhamentos.")
  test.add_argument("trained_model_path", help="Caminho para o modelo treinado.")
  test.add_argument("batch_size", help='Tamanho de cada batch para o teste.', type=int)

  test.add_argument("-u", "--unseen_speakers", required=False, action="store_true", default=False, help='Opção para fazer testes com pessoas não vistas pelo treino')
  test.add_argument("-g", "--choose_gpu", required=False, help="Opção para escolher uma GPU específica para o teste ou treinamento.")
  test.add_argument("-s", "--save_results", required=False, action="store_true", default=False, help="Opção para salvar os resultados adquiridos na pasta do modelo.")
  test.add_argument("-lm", "--landmark_features", required=False, action="store_true", default=False, help="Opção para habilitar passagem de landmark features para o modelo")

  test.add_argument("-n", "--experiment_name", required=False, default = None, type=str, help="O nome do experimento, será inserido nos logs. Se vazio, será usado o que estiver nos logs.")
  test.add_argument("-d", "--description", required=False, default = None, type=str, help="A descrição do experimento, será inserida nos logs. Se vazio, será usado o que estiver nos logs.")

  preprocess = subparsers.add_parser("preprocess")

  preprocess.add_argument("dataset_path", help="Caminho para os vídeos crus. Caso a extração de bocas seja ignorada, é o caminho para os vídeos das bocas em .npz.")
  preprocess.add_argument("results_folder", help="Caminho para a pasta onde o dataset processado estará. Será criada a pasta npz_mouths e single_words")
  preprocess.add_argument("width", help="Largura da extração. 160 para lipformer, 100 para as outras", type=int)
  preprocess.add_argument("height", help="Altura da extração. 80 para lipformer, 50 para as outras", type=int)
  preprocess.add_argument("crop_mode", help="Modo de extração de boca. Tipo de recorte que será feito após a transformação afim. Opções: [resize, crop]")
  preprocess.add_argument("-lm", "--landmark_features", required=False, action="store_true", help="Indica a extração das features de landmark")
  preprocess.add_argument("-ts", "--time_series", required=False, action="store_true", help="Opção para extrair apenas as landmark features como serie temporal")

  args = args or vars(ap.parse_args())

  mode = args["mode"]

  if mode == "train" or mode == "test":
    from model.architectures.video import LipNet, LCANet, m3D_2D_BLSTM, LipFormer
    from model.architectures.time_series import LipNet1D, Conformer
    from model.decoder import RawCTCDecoder, NgramCTCDecoder, SpellCTCDecoder
    from model.model import LipReadingModel
    from generator.augmentation import JitterAug, MirrorAug, CosineLandmarkFeatures, MouthJP, CosineDiff
    
    architectures = {
      "lipnet": LipNet,
      "lipnet1d": LipNet1D,
      "lcanet": LCANet,
      "blstm":  m3D_2D_BLSTM,
      "lipformer": LipFormer,
      "lipformerdiff": LipFormer,
      "conformer": Conformer,
    }

    # lazy_ts = {
    #   1: None,
    #   2: MouthOnly,
    #   3: MouthOnlyCentroid,
    #   4: MouthJP
    # }

    augs = []
    architecture = args["architecture"].lower()

    if architecture in ["lipformer", "lipformerdiff"]:
      augs = [
        MirrorAug([True, True], [False, True], [2, None]),
        JitterAug([True, True]),
        CosineLandmarkFeatures([False, True]),
      ]
      if architecture == "lipformerdiff":
        augs.append(CosineDiff([False, True]))

    elif architecture in ["lipnet", "lcanet", "blstm"]:
      augs = [
        MirrorAug([True], [False], [2]),
        JitterAug([True])
      ]

    else:
      raise NotImplementedError

    # post_processing = None
    # standardize = True
    # if args["half_frame"]:
    #   post_processing = HalfFrame()

    # elif args["frame_sample"] > 1:
    #   post_processing = FrameSampler(rate = args["frame_sample"])

    # elif args["lazy_ts_mode"] > 1:
    #   post_processing = lazy_ts[args["lazy_ts_mode"]]()
    #   if args["lazy_ts_mode"] == 3:
    #     standardize = False

    multi_gpu = False
    if args["choose_gpu"] is not None:
      if int(args["choose_gpu"]) < 0:
        multi_gpu = True
      else:
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{args['choose_gpu']}"

    checkpoint_path = None
    arch_obj = None

    if mode == "train":
      assert architecture in architectures.keys(), f"Arquitetura {architecture} não implementada"
      
      checkpoint_path = os.path.join("saved_models", args["save_model_path"] + "_best")

      # arch_obj = architectures[architecture](half_frame = args["half_frame"], frame_sample = args["frame_sample"])
      arch_obj = architectures[architecture](half_frame = False, frame_sample = False)

    model = LipReadingModel(
      model_path      = args["trained_model_path"],
      architecture    = arch_obj,
      multi_gpu       = multi_gpu,
      experiment_name = args["experiment_name"],
      description     = args["description"],
      pre_processing  = args.get("preprocessing_type"),
    )

    model.load_data(
      x_path            = args["dataset_path"],
      y_path            = args["alignment_path"], 
      batch_size        = args["batch_size"],
      validation_only   = False,
      unseen_speakers   = args["unseen_speakers"],
      augmentation      = augs,
      standardize       = True,
    )

    if mode == "train":
      model.fit(
        epochs = args["epochs"],
        tensorboard_logs = os.path.join("logs", args["save_model_path"]),
        checkpoint_path = checkpoint_path,
        patience=args["patience"]
      )
      model.save_model(os.path.join("saved_models", args["save_model_path"]))

    if mode == "test" or not args["skip_evaluation"]:
      metrics_path = "./"
      # current_model_path = args["save_model_path"] if mode == "train" else args["trained_model_path"]

      # if mode != "test" or args["save_results"] is True:
        # metrics_path = current_model_path

      decoders = [RawCTCDecoder(), NgramCTCDecoder(), SpellCTCDecoder()]
      # decoders = [SpellCTCDecoder()]
      pred_time = time.time()
      predictions = model.predict()
      pred_time = int(time.time() - pred_time)

      for decoder in decoders:
        model.post_processing = decoder 
        model.evaluate_model(predictions, pred_time, save_metrics_folder_path = metrics_path)

      if mode == "train":
        model.load_model(checkpoint_path)
        print("LOSS Best model:")

        pred_time = time.time()
        predictions = model.predict()
        pred_time = int(time.time() - pred_time)

        for decoder in decoders:
          model.post_processing = decoder 
          model.evaluate_model(predictions, pred_time, save_metrics_folder_path = metrics_path)

        print("WER Best model:")
        model.load_model(checkpoint_path + "_wer")

        pred_time = time.time()
        predictions = model.predict()
        pred_time = int(time.time() - pred_time)

        for decoder in decoders:
          model.post_processing = decoder 
          model.evaluate_model(predictions, pred_time, save_metrics_folder_path = metrics_path)

  elif mode == "preprocess":
    from preprocessing.mouth_extraction import convert_all_videos_multiprocess
    # from preprocessing.single_words import slice_all_videos_multiprocess

    raw_video_path = args["dataset_path"]
    mouths_path = os.path.join(args["results_folder"], "npz_mouths")

    convert_all_videos_multiprocess(
      path = raw_video_path, 
      extension = ".mpg",
      dest_folder = mouths_path,
      landmark_features = args["landmark_features"],
      shape = (args["width"], args["height"]),
      mode = args["crop_mode"],
      time_series = args["time_series"]
    )

if __name__ == '__main__':
	main()