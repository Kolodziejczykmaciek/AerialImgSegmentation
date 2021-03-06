import sys
import time
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from aerialDatasetParser import aerialValidationData


seed = 42
np.random.seed = seed



X_val = aerialValidationData(100)

# for i in range(20): # Can use this loop to look through loaded train/val images
#
#     ix = random.randint(0, len(X_val) - 1)
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('Zdjecie z listy numer:  {}'.format(ix), fontsize=15)
#
#     im1 = ax1.imshow(X_val[ix])
#     ax1.set_title('Zdj, którego sieć nie zna')
#     im2 = ax2.imshow(np.squeeze(X_val[ix]))
#     ax2.set_title('Maska binarna z {}'.format("pies"))
#     plt.show()




print("Loading model from save...")
model = tf.keras.models.load_model(
    r"C:/Users/macie/PycharmProjects/AerialImgSegmentation/Trained_models/AerialDataSet_p2_bs10_4ktrainimg")
print("Model has been loaded!")

X_val = X_val[0:1]
try:
    time_start = time.time()
    print('Segmentaion of {} images in progres...'.format(len(X_val)))
    predition_Xval = model.predict(X_val, verbose=1)
    time_stop = time.time()
    print("It took: {} seconds".format(time_stop - time_start))
except MemoryError:
    print("Za mało ramu !!!")
    sys.exit()

for n in range(1):
    ix = random.randint(0, len(X_val) - 1)
    # Perform a sanity check on some random training samples
    # Display some validation images
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Segmentation of aerial images', fontsize=15)

    im1 = ax1.imshow(np.squeeze(X_val[0].astype(np.uint8)))
    ax1.set_title('Image that neural net has never seen')
    # im2 = ax2.imshow(np.squeeze(Y_val[ix]))
    # ax2.set_title('Maska binarna z {}'.format(classes))
    im2 = ax2.imshow(np.squeeze(predition_Xval[0]))
    ax2.set_title('Evaluated result')
    plt.show()