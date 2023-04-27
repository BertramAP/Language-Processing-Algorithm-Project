import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
import tensorflow_datasets as tfds
import string
class myModel(Model):
    def __init__(self):
        super(myModel, self).__init__()
        #Her bliver nn's lager defineret
        self.em = Embedding(input_dim=1000, output_dim=64)
        self.lstm = LSTM(128, input_shape=(None, 1))
        self.d = Dense(10)

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

if __name__ == '__main__':
    ds_train, ds_test = tfds.load(
        'glue',
        split=["train", "test"],
        try_gcs=True,
    )
    assert isinstance(ds_train, tf.data.Dataset)
    ds_train = ds_train.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    """
    for i in ds_train:
        idx_train, label_train, sentence_train = i["idx"], i["label"], i["sentence"]
        print(sentence_train)
    for i in ds_train:
        (idx_test, label_test, sentence_test) = i["idx"], i["label"], i["sentence"]
    """
    #skal lave databehandling
    #Laver en instans af myModel
    model = myModel()

    #Sætter loss funktion til Sparse Cossentropy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #Sætter optimizer til Adam
    optimizer = tf.keras.optimizers.Adam()

    #Definere værdier der skal måles for hver epoch
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    EPOCHS = 5

    for epoch in range(EPOCHS):
        #Nulstiller de målte værdier
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for i in ds_train:
            for sentence in i["sentence"]:
                words = tf.strings.split(sentence)
                for i in range(1, np.size(words) - 2):
                    trainModel(words[0:i], words[i])
        #Træner modellen (mangler data)
        """print(len(sentence_train))
        for j in range(len(sentence_train)):
            words = tf.strings.split(sentence_test)
            for i in range(1, np.size(words)-2):
                trainModel(words[0:i], words[i])
        #Tester modelllen
        for j in range(len(sentence_test)):
            words = tf.strings.split(sentence_test)
            for i in range(1, np.size(words) - 2):
                trainModel(words[0:i], words[i])"""
        for i in ds_test:
            for sentence in i["sentence"]:
                ords = tf.strings.split(sentence)
                for i in range(1, np.size(words) - 2):
                    trainModel(words[0:i], words[i])

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
