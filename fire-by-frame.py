import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("D:/Github/Fire-Surveillance/Model/fire-detection.h5")


def sol_check(sample):
    img = image.load_img(sample, target_size=(150,150))
    plt.imshow(img)
    Y = image.img_to_array(img)
    X = np.expand_dims(Y,axis = 0)
    val = model.predict(X)
    print(val)
    if val==1:
        print("Fire Absent")
    elif val== 0:
        print("Fire Exists")



#Non-fire

sol_check("D:/Github/Fire-Surveillance/fire-or-nofire/No-Fire/coast_cdmc933.jpg")


#Fire

sol_check("D:/Github/Fire-Surveillance/fire-or-nofire/Fire/19.jpg")
        
