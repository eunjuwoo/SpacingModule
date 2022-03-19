import numpy as np
import tensorflow as tf
import json

from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants

class SpacingTester:
    def __init__(self, graph, filename, test_option):
        self.model = graph
        self.test_fn = filename
        self.option = test_option
        self.character_to_index = self.load_character_to_index()
    
    def load_character_to_index(self):
        print('[INFO] Load char2idx.info file.')
        with open(self.option['char2idx_fn'], 'r') as json_file:
            self.character_to_index = json.load(json_file)
        return self.character_to_index
    
    def data_generator(self, corpus_filename):  
        x = ['BOL', ]
        y = [0.0, ]
        for line in open(corpus_filename):
            if line == '\n':
                x.append('EOL')
                y.append(0.0)
                x_padded = np.pad(x, (0, self.option['max_characters']+2-len(x)), 'constant', constant_values='PAD')
                y_padded = np.pad(y, (0, self.option['max_characters']+2-len(x)), 'constant', constant_values=0.0)
                yield ([self.character_to_index.get(c, self.character_to_index['UNK']) for c in x_padded], y_padded)
                
                x = ['BOL', ]
                y = [0.0, ]
            else:
                if self.option['max_characters'] == 0 or len(x) <= self.option['max_characters']:
                    character, label = line.rstrip().split('\t')
                    x.append(character)
                    y.append(float(label))
        x_padded = np.pad(x, (0, self.option['max_characters']+2-len(x)), 'constant', constant_values='PAD')
        y_padded = np.pad(y, (0, self.option['max_characters']+2-len(x)), 'constant', constant_values=0.0)
        yield ([self.character_to_index.get(c, self.character_to_index['UNK']) for c in x_padded], y_padded)

    
    def load_processing_data(self):
        ds = tf.data.Dataset.from_generator(self.data_generator,
                                        (tf.int64, tf.float64),
                                        (tf.TensorShape([302]), tf.TensorShape([302])),
                                        args=(self.test_fn,))
        test_ds = ds.batch(self.option['batch_size'])
        return test_ds
    
    def to_result(self, characters, labels):
        index_to_character = dict((value, key) for (key, value) in self.character_to_index.items())
        result = []
        for c, p in zip(characters, labels):
            if c in [0, 1, 2, 3]:
                continue
            if p == 1:
                result.append(' ')
            result.append(index_to_character[c])
        return ''.join(result).strip()
    
    def decoder(self):  
        test_ds = self.load_processing_data()

        correct = 0
        total = 0
        for features, labels in test_ds:
            predictions = self.model(features)
            for feat, label, pred in zip(features, labels, predictions):
                feat, label, pred = feat.numpy(), label.numpy(), pred.numpy()
                _pred = []
                
                for p in pred[0]:
                    _pred.append(p.argmax())
                    
                test_result = self.to_result(feat, _pred)
                answer = self.to_result(feat, label)
                
                if test_result.strip() == answer.strip():
                    correct += 1
                else:
                    print('[test_result] : {}'.format(test_result))
                    print('[  answer   ] : {}\n'.format(answer))
                
                total += 1
                
        print('Accuracy : {}%'.format((correct/total)*100))
        return
    
def load_graph(frozen_graph_fn):
    saved_model_loaded = tf.saved_model.load(frozen_graph_fn, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]   # signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"
    graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
    return graph_func


if __name__=="__main__":
    '''===== Test Setting [START]====='''
    
    frozen_graph_dir='exp0_KoreanSpacingModule_BiLSTM/model/epoch_5'
    test_fn='corpus/test.txt'

    
    option = {
            'batch_size' : 1,
            'max_characters' : 300,
            'char2idx_fn' : frozen_graph_dir.split('/model/')[0] + '/char2idx.info'
            }
    
    ''' ===== ===== ===== ===== ===== '''
    
    graph = load_graph(frozen_graph_dir)
    st = SpacingTester(graph, test_fn, option)
    st.decoder()
