# Lipreading

Trabalho de conclusão de curso da graduação, na Universidade de Brasília (UnB).

O trabalho tem 3 modos de execução: treino, teste e pré-processamento. Para uso personalizado, por favor utilize o help do programa. Exemplo: `python main.py train -h`.

Para utilizar o modo de treino ou teste, primeiramente você deve processar os dados. Cada elemento do seu conjunto de dados deve ser um recorte da boca do orador com extensão .npy ou .npz, oposto a um vídeo de rosto completo e com compressão. O conjunto de dados cru pode ser encontrado em [https://spandh.dcs.shef.ac.uk/gridcorpus/](https://spandh.dcs.shef.ac.uk/gridcorpus/). Para pré-processar, veja as opções pelo comando `python main.py preprocess -h`. Para os resultados obtidos na monografia, utilize largura 100, altura 50, e escolha entre crop e resize. As outras configurações indicadas no help são referentes a rede LipFormer, que não foi finalizada durante a graduação.

Com o conjunto de dados processado, é simples realizar treinos e testes. Olhe as opções em `python main.py train -h` e `python main.py test -h`. Aa pasta com os vídeos foi gerada pelo pré-processamento. A pasta com os alinhamentos pode ser baixada no site do conjunto de dados cru.

No caso do treinamento, recomendo utilizar o maior tamanho de batch possível para sua quantidade de V-RAM disponível. Com 12GB de V-RAM, consegui um tamanho de 64. O treinamento completo pode variar entre 100 e 300 épocas, dependendo do cenário. Portanto, é interessante configurar uma paciência para encerrar o treino prematuramente, com a flag `-p`.

O teste nada mais é que um script para validação do modelo obtido. É reportado o Character Error Rate (CER), Word Error Rate (WER) e BLEU para todos os testes realizados. As métricas são reportadas com o uso do modelo de linguagem e sem o modelo de linguagem.
