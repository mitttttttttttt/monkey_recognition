
import keras
import sys, os
import numpy as np
from PIL import Image
from keras.models import load_model

imsize = (64, 64)

testpic     = img_source
keras_param = "./cnn.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

model = load_model(keras_param)
img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd) # 精度の表示
prelabel = np.argmax(prd, axis=1)
if prelabel == 0:
    print(">>> ゴリラ")
elif prelabel == 1:
    print(">>> マントヒヒ")
elif prelabel == 2:
    print(">>> ニホンザル")
elif prelabel == 3:
    print(">>> オラウータン")
elif prelabel == 4:
    print(">>> リスザル")
elif prelabel == 5:
    print(">>> チンパンジー")
