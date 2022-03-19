from pyparsing import countedArray
import tensorflow as tf
import numpy as np
import json, os
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MyDataset:
    def __init__(self, corpus_filename, max_characters, char2idx_fn, option):
        self.corpus_filename = corpus_filename
        self.max_characters = max_characters
        self.char2idx_fn = char2idx_fn
        self.option = option
        if self.option=='train': self.character_to_index = self.build_character_to_index()
        elif self.option=='test': self.character_to_index = self.load_character_to_index()
        else: 
            print('[ERROR] check option. please choose either train or test.' )
            exit(1)
        print('[INFO] length of charter_to_index : ', len(self.character_to_index))
        
    def data_generator(self):  
        x = ['BOL', ]
        y = [0.0, ]
        for line in open(self.corpus_filename):
            if line == '\n':
                x.append('EOL')
                y.append(0.0)
                x_padded = np.pad(x, (0, self.max_characters+2-len(x)), 'constant', constant_values='PAD')
                y_padded = np.pad(y, (0, self.max_characters+2-len(x)), 'constant', constant_values=0.0)
                yield ([self.character_to_index.get(c, self.character_to_index['UNK']) for c in x_padded], y_padded)
                
                x = ['BOL', ]
                y = [0.0, ]
            else:
                if self.max_characters == 0 or len(x) <= self.max_characters:
                    character, label = line.rstrip().split('\t')
                    x.append(character)
                    y.append(float(label))
        x_padded = np.pad(x, (0, self.max_characters+2-len(x)), 'constant', constant_values='PAD')
        y_padded = np.pad(y, (0, self.max_characters+2-len(x)), 'constant', constant_values=0.0)
        yield ([self.character_to_index.get(c, self.character_to_index['UNK']) for c in x_padded], y_padded)
    
    def load_character_to_index(self):
        print('[INFO] Load char2idx.info file.')
        with open(self.char2idx_fn, 'r') as json_file:
            self.character_to_index = json.load(json_file)
        return self.character_to_index
    
    def build_character_to_index(self):
        characters = set()
        character_list = ['PAD', 'UNK', 'BOL', 'EOL']
        
        for line in open(self.corpus_filename):
            line=line.strip()

            if line:
                character, _ = line.split('\t')
                characters.add(character)
                
        character_list.extend(list(sorted(characters)))
        
        self.character_to_index = {c:i for i, c in enumerate(character_list)}
        
        if not os.path.exists(self.char2idx_fn):
            print('[INFO] Create char2idx.info file.')
            with open(self.char2idx_fn, 'w') as json_file:
                json.dump(self.character_to_index, json_file, ensure_ascii=False)
        
        return self.character_to_index
   
def load_processing_data(corpus_filename, batch_size, path_dict, option):
    # AUTOTUNE = tf.data.AUTOTUNE
    dataset = MyDataset(corpus_filename, 
                        max_characters=300, 
                        char2idx_fn=path_dict['cp_path']+'/char2idx.info',
                        option=option)
    ds = tf.data.Dataset.from_generator(dataset.data_generator,
                                        (tf.int64, tf.float64),
                                        (tf.TensorShape([302]), tf.TensorShape([302])),
                                        args=())
    train_ds = ds.batch(batch_size)
    return train_ds

if __name__=="__main__":
    _generator = data_generator('corpus/test.txt')
    print(next(_generator))