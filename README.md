# SpacingModule
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"/>

Korean sentence spacing module is based on Deep Learning model(bi-LSTM) trained from small corpus. This module is a package that converts non-spaced sentences into spaced sentences through the published model(exp0_KoreanSpacingModule_BiLSTM). Also, You can use this module to train yourself.

## Environment Setting
* Python version > 3.6.13 (Recommend Anaconda)
* [Tensorflow](https://tensorflow.org, "Tensorflow link") tensorflow-gpu version > 2.3.0 
* AWS Environment
   * [GPU Tesla M60](https://images.nvidia.com/content/pdf/tesla/tesla-m60-product-brief.pdf, "GPU Tesla M60 Spec")
     : The Tesla M60 has 16 GB GDDR5 memory (8 GB per GPU) and a 300 W maximum power limit. 
   * Memory 128GB
   * CPU 16 core

## Methodelogy
### Generate corpus.
 Split train corpus and test corpus from origin corpus data.
  ```
  cat review_shuffled.txt | head -2000 > review_shuffled_test.txt
  cat review_shuffled.txt | tail +2002 > review_shuffled_train.txt
  ```
  And then, We need to convert the shape of the data for training.
  ```
  python3 generate_corpus.py < review_shuffled_test.txt > test.txt
  python3 generate_corpus.py < review_shuffled_train.txt > train.txt
  ```
 ### Train Neural Network
  You can set the data path (train_fn/test_fn/val_fn) in train.py
  ```
    python train.py -e [int, epochs] \
                    -l [float, learning_rate] \
                    -b [int, batch_size] \
                    -a [str, activation_function] \
                    -c [int, experiment_count]
  ```
 ### Test model
   ```
    python test.py
   ```
 ### Reference 
  1. https://jins-sw.tistory.com/m/37
----------------------------------------------------------------------------------------------------------------------------
