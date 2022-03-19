'''
=========== Project Folder Structure ===========
    -- train.py               # train code
    -- test.py                # test code
    -- models.py              # model structure
    -- generator_corpus.py  
    
    -- [corpus]
      -- etc... train/test corpus
      
    -- [utils]
       -- basic_utils.py
       -- cp_utils.py
       -- dataset_utils.py
       -- learning_env_setting.py
       -- train_validation_test.py

    -- [exp*_*_*]  # 훈련 돌리면 나오는 결과물들
       -- confusion_matrix
       -- model
       -- losses_accs.npz
       -- losses_accs_visualization.png
       -- test_result.txt
=========== ======================= ===========
'''

import os

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from utils.learning_env_setting import dir_setting, continue_setting, get_classification_metrics, argparser
from utils.dataset_utils import load_processing_data
from utils.basic_utils import resetter, training_reporter
from utils.train_validation_test import train, validation, test
from utils.cp_utils import save_metrics_model, metric_visualizer
from models import SpaceModel

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

'''===== Learning Setting [START]====='''
exp_name = 'KoreanSpacingModule'
CONTINUE_LEARNING = False  # False : 처음 돌리거나 지금까지 돌린게 있으면 모두 지우고 훈련 진행.

batch_size = 128

epochs = 6
save_period = 1
learning_rate = 0.001
train_fn='corpus/train.txt'
val_fn='corpus/test.txt'
test_fn='corpus/test.txt'

exp_idx, epochs, learning_rate, batch_size, activation = argparser(epochs=epochs,
                                                                  learning_rate=learning_rate,
                                                                  batch_size=batch_size)
exp_name = 'exp' + str(exp_idx) + '_' + exp_name + '_BiLSTM'

'''===== Learning Setting [END]====='''

loss_object = SparseCategoricalCrossentropy()
path_dict = dir_setting(exp_name,CONTINUE_LEARNING)

train_ds = load_processing_data(train_fn, batch_size, path_dict, option='train')
validation_ds = load_processing_data(val_fn, batch_size, path_dict, option='test')
test_ds = load_processing_data(test_fn, batch_size, path_dict, option='test')

model = SpaceModel()

optimizer = Adam(learning_rate=learning_rate)
model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model)


metric_objects = get_classification_metrics()

for epoch in range(start_epoch, epochs):
   train(train_ds, model, loss_object, optimizer, metric_objects)
   validation(validation_ds, model, loss_object, metric_objects)

   training_reporter(epoch, losses_accs, metric_objects)
   save_metrics_model(epoch, model, losses_accs, path_dict, save_period)

   metric_visualizer(losses_accs, path_dict['cp_path'])
   resetter(metric_objects)

test(test_ds, model, loss_object, metric_objects, path_dict)
