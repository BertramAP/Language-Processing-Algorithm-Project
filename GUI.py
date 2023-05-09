import sys
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget, QGridLayout, QPushButton, QTextEdit, QMainWindow
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
from nltk.tokenize import RegexpTokenizer

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#print(os.getenv("TF_GPU_ALLOCATOR"))

gpus = tf.config.experimental.list_physical_devices("GPU")
#setup af gpu vram
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def sherlock_setup():
    global n_words, unique_tokens, unique_token_index, tokenizer, tokens
    #data indlæsning
    path = 'data.txt'
    text = open(path, "r", encoding='utf-8').read().lower()
    # Tokineser ordende
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    #definerer mængden af tokens
    unique_tokens = np.unique(tokens)
    # Mængden af kontext model har brug for
    n_words = 5
    #Definerer alle ord ai'en kender, i ud fra dataset, i form a tokens
    unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
    #Slet data, der ikke længere skal bruges
    del path, text
"""
def wonderland_setup():
    #data indlæsning
    global n_words, unique_tokens, unique_token_index, tokenizer, tokens
    path = 'wonderland.txt'
    text = open(path, "r", encoding='utf-8').read().lower()
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    #definerer mængden af tokens
    unique_tokens = np.unique(tokens)
    print(len(unique_tokens))
    # Mængden af kontext model har brug for
    tokens = tokens
    n_words = 5
    #Definerer alle ord ai'en kender, i ud fra dataset, i form a tokens
    unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
    #Slet data, der ikke længere skal bruges
    del path, text
"""
class MainScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = load_model("Shelorck holmes model 1.h5")
        self.modelON = True

        self.setWindowTitle('RNN')

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.textArea = QTextEdit()
        self.textArea.setReadOnly(False)
        self.textArea.textChanged.connect(self.text_changed)

        self.buttons = [QPushButton("Tryk her 1"),
                        QPushButton("Tryk her 2"),
                        QPushButton("Tryk her 3")
        ]

        self.buttons[0].clicked.connect(self.ins_option_1)
        self.buttons[1].clicked.connect(self.ins_option_2)
        self.buttons[2].clicked.connect(self.ins_option_3)

        self.options = {0: " ", 1: " ", 2: " "}

        self.layout.addWidget(self.textArea, 1, 0, 1, 3)
        self.layout.addWidget(self.buttons[0], 2, 0)
        self.layout.addWidget(self.buttons[1], 2, 1)
        self.layout.addWidget(self.buttons[2], 2, 2)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.menubar = self.menuBar()
        self.filemenu = self.menubar.addMenu("file")
        self.filemenu.addAction("Vælg data", self.open_file_dialog)


    def ins_option_1(self):
        self.textArea.insertPlainText(" " + self.options[0])

    def ins_option_2(self):
        self.textArea.insertPlainText(" " + self.options[1])

    def ins_option_3(self):
        self.textArea.insertPlainText(" " + self.options[2])

    def text_changed(self):
        print("NO")
        text = self.words_exist(self.textArea.toPlainText().lower().split(" "))

        if self.modelON:
            print("nogen har trykket")
            #Tekst her
            print(self.textArea.toPlainText(), type(text))
            possible = np.array([unique_tokens[idx] for idx in self.predict_next_word(text.split(" "), 3)])
            possible = np.array_str(possible)
            words = []
            word = ""
            for i in range(len(possible)):
                if possible[i].isspace() or possible[i] == "]":
                    words.append(word)
                    word = ""
                elif possible[i] != "[" and possible[i] != "'":
                    word = word + possible[i]

            for index, word in enumerate(words):
                self.buttons[index].setText(word)
                self.options[index] = word
        else:
            pass

    def words_exist(self, text):
        print(text)
        print("Whole text is: ", text)
        if len(text) > n_words:
            sentence = ""
            for i in range(len(text)-n_words, len(text)):
                print(f"{text[i]} in unique_tokens? ", text[i] in unique_token_index)
                if text[i] in unique_tokens:
                    sentence = sentence + " " + text[i]
                if text[i] not in unique_tokens and not i == len(text)-1:
                    sentence = ""
        else:
            sentence = ""
            for i in range(len(text)):
                print(f"{text[i][0]} in unique_tokens? ", text[i] in unique_token_index)
                if text[i] in unique_token_index:
                    sentence = sentence + " " + text[i]
                elif text[i] not in unique_token_index and not i == len(text) - 1:
                    sentence = ""
                else:
                    sentence = sentence
            print(sentence)
        print("Sentence2 is: ", type(sentence))
        return sentence

    def open_file_dialog(self):
        filename = QFileDialog.getOpenFileName(self, filter="Text (*.txt)", caption="Select a text file", directory="C:/Users")
        print(filename)
        if filename[0] != "":
            #Turn of model
            self.modelON = False
            path = filename[0]
            print("Hej")

            #TODO: let model train on loaded data file
            print("Hej0")
            text = open(path, "r", encoding='utf-8').read().lower()

            self.textArea.setReadOnly(True)

            self.textArea.setText("Training model, it will take some time")
            QApplication.processEvents()


            can_load_weights = update_tokens(text)
            print("Hej1")

            input_words = []
            next_words = []

            for i in range(len(tokens) - n_words):
                input_words.append(tokens[i:i + n_words])
                next_words.append(tokens[i + n_words])

            # Features
            x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
            # Labels
            y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)
            print("Hej2")
            for i, words in enumerate(input_words):
                for j, word in enumerate(words):
                    x[i, j, unique_token_index[word]] = 1
                y[i, unique_token_index[next_words[i]]] = 1
            print("Hej3")


            print("Hej4")
            if not can_load_weights:
                newModel = Sequential()
                newModel.add(LSTM(128, input_shape=(n_words, len(unique_tokens))))
                newModel.add(Dense(len(unique_tokens)))
                newModel.add(Activation("softmax"))
                newModel.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01),
                                 metrics=["accuracy"])
                newModel.fit(x, y, batch_size=16, epochs=3, shuffle=True)
                self.model = newModel
                del newModel

            if can_load_weights:
                self.model.fit(x, y, batch_size=16, epochs=3, shuffle=True)
            print("Hej5")


            self.textArea.setText("")
            self.textArea.setReadOnly(False)
            print("Hej6")

            self.modelON = True

            del text, filename, x, y, can_load_weights



    def predict_next_word(self, input_text, n_best):
        text = ""
        print(len(input_text))
        if len(input_text) > 5:
            input_text = input_text.split(" ")
            for i in range(len(input_text)-n_words, len(input_text)-1):
                text = text + " " + input_text[i]
        else:
            for i in range(len(input_text)):
                print("I am: ", input_text[i])
                text = text + " " + input_text[i]
        print(text)
        X = np.zeros((1, n_words, len(unique_tokens)))
        for i, word in enumerate(text.split()):
            X[0, i, unique_token_index[word]] = 1
        print("Hello")
        predictions = self.model.predict(X)[0]
        return np.argpartition(predictions, -n_best)[-n_best:]

def update_tokens(text):
    global unique_tokens, tokens, unique_token_index,tokenizer
    x_old_words = len(unique_tokens)
    print(x_old_words)
    newTokens = tokenizer.tokenize(text)
    print("goddav")
    unique_tokens = np.unique(tokens + newTokens)
    tokens = newTokens
    unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
    x_new_words = len(unique_tokens)
    print(x_old_words, x_new_words)
    if x_new_words == x_old_words:
        return True
    else:
        return False

sherlock_setup()


app = QApplication(sys.argv)
window = MainScreen()

window.showMaximized()
app.exec()


app = QApplication(sys.argv)
window = MainScreen()

window.showMaximized()
app.exec()

