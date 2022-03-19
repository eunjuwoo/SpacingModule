from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, TimeDistributed

class SpaceModel(Model):
    def __init__(self, vocab_size=14422, hidden_dim=128, embedding_dim=512, activation='relu'):
        super(SpaceModel, self).__init__()
        
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.lstm = Bidirectional(LSTM(hidden_dim, return_sequences=True, name='lstm1'))
        self.lstm2 = Bidirectional(LSTM(hidden_dim*2, name='lstm2'))
        self.fc = TimeDistributed(Dense(2, activation=('softmax')))
    
    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x

