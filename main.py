import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Activation
from keras.optimizers import RMSprop
from nltk.tokenize import RegexpTokenizer
from tensorflow.python.client import device_lib
print("TensorFlow version:", tf.__version__)
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""
class myModel(Model):
    def __init__(self):
        super(myModel, self).__init__()
        #Her bliver nn's lager defineret
        self.em = Embedding(input_dim=1000, output_dim=64)
        self.lstm = LSTM(128, input_shape=(n_words, len(unique_tokens)))
        self.d = Dense(len(unique_tokens))

    #Når model bliver kaldet, vil vi passerer dataen igennem lagerne
    def call(self, x):
        self.em(x)
        self.lstm(x)
        return self.d(x)

@tf.function
def trainModel(sentence, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(sentence, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def testModel(sentence, labels):
    predictions = model(sentence, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

def sanitize(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation)).lower()
"""
if __name__ == '__main__':
    n_words = 5
    input_words = []
    next_words = []

    path = 'data.txt'
    text = open(path, "r", encoding='utf-8').read().lower()
    print('length of the corpus is: :', len(text))
    # Tokineser ordende
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    # fjerner gentagende tokens
    unique_tokens = np.unique(tokens)
    # giver hver token en id, vha. en dictionary
    unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
    # definer mængden af ord modellen skal bruge til at forudsige det næste


    for i in range(len(tokens)-n_words):
        input_words.append(tokens[i:i+n_words])
        next_words.append(tokens[i + n_words])
    #Features
    x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
    #Labels
    y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)
    for i, words in enumerate(input_words):
        for j, word in enumerate(words):
            x[i, j, unique_token_index[word]] = 1
        y[i, unique_token_index[next_words[i]]] = 1

    model = Sequential()
    model.add(LSTM(128, input_shape=(n_words, len(unique_tokens))))
    model.add(Dense(len(unique_tokens)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
    model.fit(x, y, batch_size=128, epochs=10, shuffle=True, validation_split=0.05)

    #Gemmer modellen som en fil
    model.save("Shelorck holmes model 1.h5")




