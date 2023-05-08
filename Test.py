from nltk.tokenize import RegexpTokenizer
import numpy as np
from tensorflow import keras
from keras import Model
from keras.models import load_model, Sequential
from keras.optimizers import RMSprop

from keras.layers import Dense, LSTM, Embedding, Activation

path = 'data.txt'
text = open(path, "r", encoding='utf-8').read().lower()
# Tokineser ordende
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(text)
# fjerner gentagende tokens
unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
path = 'wonderland.txt'
text = open(path, "r", encoding='utf-8').read().lower()
n_words = 5
def update_tokens(text):
    global unique_tokens, tokens, unique_token_index
    x_old_words = len(unique_tokens)
    newTokens = tokenizer.tokenize(text)
    unique_tokens = np.unique(tokens + newTokens)
    tokens = newTokens
    unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
    x_new_words = len(unique_tokens)
    print(x_old_words, x_new_words)
    if x_new_words == x_old_words:
        return True
    else:
        return False

model = load_model("Shelorck holmes model 1.h5")
can_load_weights = update_tokens(text)
if can_load_weights:
    model.save_weights("./checkpoints")

input_words = []
next_words = []
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

#Features
x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
#Labels
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)
for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1
newModel = Sequential()
newModel.add(LSTM(128, input_shape=(n_words, len(unique_tokens))))
newModel.add(Dense(len(unique_tokens)))
newModel.add(Activation("softmax"))
newModel.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
if can_load_weights:
    newModel.load_weights("./checkpoints")

newModel.fit(x, y, batch_size=128, epochs=3, shuffle=True)