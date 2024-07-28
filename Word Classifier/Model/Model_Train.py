from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import os


if __name__ == '__main__':
    dataset = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Dataset\\', 'Dataset.csv')
    model_save = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model_Data\\', 'WordClassifier_Model.h5')

    x_dataset = np.loadtxt(dataset, delimiter=',', dtype='O', usecols=(1))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int64', usecols=(0))

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.8, random_state = 42)

    model = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
    hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.summary()

    es_callback = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)

    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[es_callback]
    )

    model.save(model_save, include_optimizer=False)
