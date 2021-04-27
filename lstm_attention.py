import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import *
from attention import Attention
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import feature_engineering as fe
import os

def main():
    path = os.path.dirname(__file__)
    IMAGES_PATH = os.path.join(path, 'images/train')
    # Dummy data. There is nothing to learn in this example.
    #num_samples, time_steps, input_dim, output_dim = 100, 10, 1, 1
    # data_x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    # data_y = np.random.uniform(size=(num_samples, output_dim))

    train_generator, validation_generator, test_generator = fe.load_time_series_data(image_path=IMAGES_PATH)

    num_frames = 5  # 10 seconds of video at 24 ips.
    h, w, c = 255, 255, 3  # def not a HD video! 32x32 color.

    inputs = Input(shape=(num_frames, h, w, c))
    # push num_frames in batch_dim to process all the frames independently of their orders (CNN features).
    x = Lambda(lambda y: K.reshape(y, (-1, h, w, c)))(inputs)
    # apply convolutions to each image of each video.
    x = Conv2D(16, 5)(x)
    x = MaxPool2D()(x)
    # re-creates the videos by reshaping.
    # 3D input shape (batch, timesteps, input_dim)
    num_features_cnn = np.prod(K.int_shape(x)[1:])
    x = Lambda(lambda y: K.reshape(y, (-1, num_frames, num_features_cnn)))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Attention(32)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)


    # Define/compile the model.
    # model_input = Input(shape=(num_frames, h, w, c))
    # x = LSTM(64, return_sequences=True)(model_input)
    # x = Attention(32)(x)
    # x = Dense(1)(x)
    # model = Model(model_input, x)

    # model = tf.keras.models.Sequential([
    #     #  First Convolution
    #     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(255, 255, 3)),
    #     BatchNormalization(),
    #     Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
    #     BatchNormalization(),
    #     Flatten(),
    #     Dropout(0.2),
    #     # Output layer
    #     Dense(1, activation='sigmoid')]
    # )

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    # train.
    history = model.fit(x=train_generator,
                        epochs=10,
                        validation_data=validation_generator,
                        verbose=1)

    # test save/reload model.
    pred1 = model.predict(test_generator)
    model.save('test_model.h5')
    model_h5 = load_model('test_model.h5')
    pred2 = model_h5.predict(test_generator)
    np.testing.assert_almost_equal(pred1, pred2)
    print('Success.')


if __name__ == '__main__':
    main()