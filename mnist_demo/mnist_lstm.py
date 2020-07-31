import keras
import os
from AbstractRNNClassifier import AbstractRNNClassifier
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Lambda, LSTM, Dense
from keras.models import load_model
from keras.models import Model
import numpy as np


class MnistLSTMClassifier(AbstractRNNClassifier):
    def __init__(self):
        # Classifier
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 128  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = 10  # mnist classes/labels (0-9)
        self.batch_size = 128  # Size of each batch
        self.n_epochs = 20

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        # self.model.summary()

    def load_hidden_state_model(self, model_path):
        """
        return the rnn model with return_sequence enabled.
        """
        input = Input(shape=(self.time_steps, self.n_inputs))
        lstm = LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True)(input)
        last_timestep = Lambda(lambda x: x[:, -1, :])(lstm)
        dense = Dense(10, activation='softmax')(last_timestep)
        model = Model(inputs=input, outputs=[dense, lstm])
        model.load_weights(model_path)
        self.model = model

    def train(self, save_path):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = self.input_preprocess(x_train)
        x_test = self.input_preprocess(x_test)

        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "model.h5"))

    def evaluate(self, model=None):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_test = self.input_preprocess(x_test)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x_test, y_test)
        print(test_loss)

    def input_preprocess(self, data):
        data = data.reshape(data.shape[0], self.n_inputs, self.n_inputs)
        data = data.astype('float32')
        data /= 255
        return data

    def profile_train_data(self, profile_save_path):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = self.input_preprocess(x_train)
        output = self.model.predict(x_train)
        os.makedirs(profile_save_path, exist_ok=True)
        np.save(os.path.join(profile_save_path, "states_profile.npy"), output[1])

    def get_state_profile(self, inputs):
        inputs = self.input_preprocess(inputs)
        output = self.model.predict(inputs)
        return output[1]


if __name__ == "__main__":
    save_path = "test/rnn_model"

    lstm_classifier = MnistLSTMClassifier()
    # train an rnn model
    # lstm_classifier.create_model()
    # lstm_classifier.train(save_path)
    # lstm_classifier.evaluate()

    # Load a trained model with return_sequence enabled.
    profile_path = "test/output/profile_save"
    lstm_classifier.load_hidden_state_model(os.path.join(save_path, "model.h5"))
    lstm_classifier.profile_train_data(profile_path)
