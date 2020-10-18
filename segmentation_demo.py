import sys
import os
import time
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')
seed = 42
np.random.seed = seed


from Aerial_DataSet_Parser import aerialDataPrepare
from Aerial_DataSet_Parser import aerialValidationData
from Unet_Model import UnetModel


X_val = aerialValidationData(100)  # load image


print("Loading saved model...")
model_path = os.path.join('Trained_models', 'AerialDataSet_p2_bs10_4ktrainimg')
model = tf.keras.models.load_model(model_path)
print("Model has been loaded!")


X_val = X_val[0:1]
try:
    time_start = time.time()
    print('Segmentation of {} images in progress...'.format(len(X_val)))
    predition_Xval = model.predict(X_val, verbose=1)
    time_stop = time.time()
    print("It took: {} seconds".format(time_stop - time_start))
except MemoryError:
    print("Insufficient level of RAM...")
    sys.exit()


# Prepare report
ix = random.randint(0, len(X_val) - 1)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Segmentation of aerial images', fontsize=15)

im1 = ax1.imshow(np.squeeze(X_val[0].astype(np.uint8)))
ax1.set_title('Image that neural net has never seen')
im2 = ax2.imshow(np.squeeze(predition_Xval[0]))
ax2.set_title('Evaluated result')
plt.show()
