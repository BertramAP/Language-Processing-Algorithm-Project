import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

class myModel(Model):
    def __init__(self):
        super(myModel, self).__init__()
        #Her bliver nn's lager defineret
        self.em = Embedding(input_dim=1000, output_dim=64)
        self.lstm = LSTM(128)
        self.d = Dense(10)

    #Når model bliver kaldet, vil vi passerer dataen igennem lagerne
    def call(self, x):
        self.em(x)
        self.lstm(x)
        return self.d(x)

if __name__ == '__main__':
    #Laver en instans af myModel
    model = myModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
