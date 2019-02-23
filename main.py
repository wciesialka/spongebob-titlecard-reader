import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from PIL import Image

alphabet = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
            'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

inv_alphabet = {v: k for k, v in alphabet.items()}


def image_to_data(path):
    data = []
    img = Image.open(path)
    img = img.resize((24, 24), Image.NEAREST)
    img = img.convert(mode="LA")
    size = img.size
    for y in range(0, size[1]):
        for x in range(0, size[0]):
            l = img.getpixel((x, y))[0]
            data.append(l/255)
    return data


def empty_array():
    a = []
    for i in range(576):
        a.append(None)
    return a


def load_data():
    raw = None
    with open("letters/letters.json") as f:
        raw = json.load(f)
    longest = max((len(v)) for k,v in raw.items())
    x_train = []
    y_train = []
    for key in raw:
        y_train.append(alphabet[key])
        x_sub = []
        for sub in raw[key]:
            img_data = image_to_data(f"letters/{key}/{sub}")
            x_sub.append(img_data)
        i = 0
        m = len(x_sub)
        while len(x_sub) < longest:
            x_sub.append(x_sub[i])
            i = (i+1) % m
        x_train.append(x_sub)
    return x_train, y_train


def main():
    x_train, y_train = load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())  # input layer
    # hidden layer. 128 neurons
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    # another hidden layer. 128 neurons
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    # output layer. 26 classifications (26 letters)
    model.add(tf.keras.layers.Dense(26, activation=tf.nn.softmax))

    model.compile(optimizer='adam',  # optimizer function
                  loss='sparse_categorical_crossentropy',  # what is loss?
                  metrics=['accuracy']  # what to track
                  )

    model.fit(x_train, y_train, epochs=30)  # 3 iterations

    predictions = model.predict([x_train])
    prediction = np.argmax(predictions[0])

    print(inv_alphabet[y_train[0]],inv_alphabet[prediction])


if __name__ == "__main__":
    main()
