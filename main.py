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

@tf.function
def trainModel(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def testModel(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

if __name__ == '__main__':
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

        #Træner modellen (mangler data)
        for images, labels in trainDS:
            trainModel(images, labels)
        #Tester modelllen
        for test_images, test_labels in testDS:
            testModel(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
