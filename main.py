import requests
from tensorflow import *
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from flask import request


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')

model.save('digits.model')
'''


model = tf.keras.models.load_model('digits.model')
'''
for x in range(1, 6):
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = tf.keras.utils.normalize(np.invert(np.array([img])))
    print(img)

    prediction = model.predict(img)
    print(f'Result: {np.argmax(prediction)}\n')

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

'''

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    jdata = request.get_json()
    array = np.transpose(np.array(jdata["pixels"]))
    # plt.imshow(array, cmap=plt.cm.binary)
    # plt.show()
    array = np.array([array])
    prediction = str(np.argmax(model.predict(array)))
    print(prediction)
    return prediction, 200


@app.route('/', methods=['GET'])
def hello():
    return '<h1>Hello World</h1>'


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=False, port=2222)

