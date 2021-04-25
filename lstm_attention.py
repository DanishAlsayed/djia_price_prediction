import numpy as np
from tensorflow.keras import Input
import tensorflow as tf
from tensorflow.keras.layers import *

from tensorflow.keras.models import load_model, Model
import feature_engineering as fe
from attention import Attention
import os

def main():
    PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(PATH, 'data')
    IMAGES_PATH = os.path.join(PATH, 'images/train')
    MODEL_PATH = os.path.join(PATH, 'models')

    # Dummy data. There is nothing to learn in this example.
    num_samples, time_steps, input_dim, output_dim = 100, 5, 2, 1
    # data_x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    # data_y = np.random.uniform(size=(num_samples, output_dim))

    data_x, validation_generator, test_generator = fe.load_time_series_data(image_path=IMAGES_PATH)
    #print(train_generator.x_cols)

    # Define/compile the model.
    #model_input = Input(shape=(time_steps, input_dim))
    #x = LSTM(64, return_sequences=True)(model_input)

    model = tf.keras.models.Sequential([
        ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(5, 255, 255, 3)),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(1, activation='sigmoid'),

    ])
    model  = ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(5, 255, 255, 3))
    model = Attention(32)(model)
    #
    # x = ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(5, 255, 255, 3))
    # x = Attention(32)(x)
    # x = Dense(1)(x)
    # model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    print(data_x)
    # train.
    model.fit(x=data_x[0][0],y=data_x[0][1], epochs=10)

    # test save/reload model.
    pred1 = model.predict(data_x)
    model.save('test_model.h5')
    model_h5 = load_model('test_model.h5')
    pred2 = model_h5.predict(data_x)
    np.testing.assert_almost_equal(pred1, pred2)
    print('Success.')


if __name__ == '__main__':
    main()