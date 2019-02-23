import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import sys
from PIL import Image

alphabet = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
            'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

inv_alphabet = {v: k for k, v in alphabet.items()}

W,H = (36,36)

def image_to_data(path):
    data = []
    img = Image.open(path)
    img = img.resize((W, H), Image.NEAREST)
    img = img.convert(mode="LA")
    size = img.size
    for x in range(0, size[0]):
        _x = []
        for y in range(0, size[1]):
            l = img.getpixel((x, y))[0]
            _x.append(l/255)
        data.append(_x)
    return data


def load_data():
    raw = None
    with open("letters/letters.json") as f:
        raw = json.load(f)
    longest = max((len(v)) for k, v in raw.items())
    x_train = []
    y_train = []
    for key in raw:
        for sub in raw[key]:
            img_data = image_to_data(f"letters/{key}/{sub}")
            x_train.append(img_data)
            y_train.append(alphabet[key])
    return x_train, y_train

model_file = "titlecard_letters.h5"

def train():
    try:
        model = keras.models.load_model(model_file)
    except:
        x_train, y_train = load_data()

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(W,H)))  # input layer
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

        model.fit(x_train, y_train, epochs=30)  # 30 iterations

        model.save(model_file)
    finally:
        return model

def predict(file):
    model = train()

    x_test = image_to_data(file)

    predictions = model.predict([[x_test]])[0]
    probable_prediction = np.argmax(predictions)

    return probable_prediction,predictions

def main(args):
    most_likely, predictions = predict(args[1])
    for i,prediction in enumerate(predictions):
        letter = inv_alphabet[i]
        probability = round((100*prediction),2)
        print(f"{letter}:\t{probability}%")
    print(f"Most likely letter is {inv_alphabet[most_likely]}")

if __name__ == "__main__":
    main(sys.argv)
