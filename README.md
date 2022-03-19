# SpacingModule
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"/>

## Requirement
* Python version > 3.x (Recommend Anaconda)
* [Tensorflow](https://tensorflow.org, "Tensorflow link") version > 2.x

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
