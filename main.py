import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from PIL import Image


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

def load_data():
    raw = None
    with open("letters/letters.json") as f:
        raw = json.load(f)
    x_train = []
    y_train = []
    for key in raw:
        y_train.append([key])
        x_sub = []
        for sub in raw[key]:
            img_data = image_to_data(f"letters/{key}/{sub}")
            x_sub.append(img_data)
        x_train.append(x_sub)
    return np.array(x_train), np.array(y_train)


def main():
    x_train, y_train = load_data()
    print(x_train)

if __name__ == "__main__":
    main()
