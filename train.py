import sys
import time
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from aerialDatasetParser import aerialDataTrain
from unetModel import UnetModel

seed = 42
np.random.seed = seed

X_train, Y_train = aerialDataTrain(1000)
print("Dl listy X_train: {} \nDl listy Y_train: {}".format(len(X_train), len(Y_train)))

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
model = UnetModel()
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=100, callbacks=callbacks)
#model.save(r"C:\Users\macie\PycharmProjects\pythonProject\AerialDataSet_p2_bs10_4ktrainimg")