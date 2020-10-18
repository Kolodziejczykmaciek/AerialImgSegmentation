
import random
import numpy as np
import os
import cv2


np.random.seed = 1000

'''
    Dowanload dataset useed in this project in https://project.inria.fr/aerialimagelabeling/
    Keep in mind that it weights about 20Gb.
'''

def aerialDataPrepare(num_train_img):
    '''
        This method returns given number of training images and corresponding masks as a nd.array.
        Images in data set are 5000x5000 pixels this method cuts every image and its binary mask into 512x512
        training images.
        One 5000x5000 pixel image is cut into 100 images.

        For now the method is hardcoded to evaluate training 5000x5000 pixel images into 512x512 but
        it can by transormed to return images in different resolution. As far as i konw the resolution must be always
        divisible by 32.
    '''

    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    IMG_CHANNELS = 3

    X_train = np.zeros((num_train_img, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((num_train_img, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    # TODO: remove it
    image_directory = r'C:/Users/macie/PycharmProjects/pythonProject/aerial_dataSet/NEW2-AerialImageDataset/AerialImageDataset/train/images/'
    mask_directory = r'C:/Users/macie/PycharmProjects/pythonProject/aerial_dataSet/NEW2-AerialImageDataset/AerialImageDataset/train/gt/'

    image_directory = os.path.join('aerial_dataSet', 'NEW2-AerialImageDataset', 'AerialImageDataset', 'train', 'images')
    mask_directory = os.path.join('aerial_dataSet', 'NEW2-AerialImageDataset', 'AerialImageDataset', 'train', 'gt')

    cords = (0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4487)
    image_resolution = 512

    train_images = os.listdir(image_directory)
    random.shuffle(train_images)
    n = 0
    for i, image_name in enumerate(train_images):
        print("Interation no.{}".format(i))
        if i == num_train_img/100: break
        if '.tif' in image_name:
            img = cv2.imread(image_directory + image_name)
            label = cv2.imread(mask_directory + image_name)

            # sub_image = 0
            for i in cords:
                for j in cords:
                    new_img = img[i:i + image_resolution, j:j + image_resolution]
                    X_train[n] = np.array(new_img)

                    new_label = label[i:i + image_resolution, j:j + image_resolution]
                    new_label = new_label[:, :, 0:1]
                    new_label = new_label / 255
                    Y_train[n] = new_label
                    n+=1
                    # sub_image+=1
                    # new_img_name = image_name[0:-4]+'_{}'.format(sub_image)
                    # cv2.imwrite(img_dir+new_img_name+".tif", new_img)

    return X_train, Y_train

def aerialValidationData(num_val_img):
    '''
        This method returns given number of 100xvalidation images as a nd.array.
        Images in data set are 5000x5000 pixels this method cuts every image into 512x512 validation images.
        One 5000x5000 pixel image is cut into 100 images.

        For now the method is hardcoded to evaluate validation 5000x5000 pixel images into 512x512 but
        it can by transormed to return images in different resolution. As far as i konw the resolution must be always
        divisible by 32.
    '''

    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    IMG_CHANNELS = 3 #RGB

    X_val = np.zeros((num_val_img, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    # TODO: remove it
    image_directory = r'C:/Users/marcin/Desktop/workSpace/telephoners/DRONY/AerialImgSegmentation/aerial_dataSet/NEW2-AerialImageDataset/AerialImageDataset/test/images/' #specify path to validation images

    image_directory = os.path.join('aerial_dataSet', 'NEW2-AerialImageDataset', 'AerialImageDataset', 'test', 'images')

    cords = (0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4487)
    image_resolution = 512

    validation_images = os.listdir(image_directory)
    random.shuffle(validation_images)
    print(validation_images)
    n = 0
    for i, image_name in enumerate(validation_images):
        print("Interation no.{}".format(i))
        if i == num_val_img/100: break
        if '.tif' in image_name:
            img = cv2.imread(image_directory + image_name)

            for i in cords:
                for j in cords:
                    new_img = img[i:i + image_resolution, j:j + image_resolution]
                    X_val[n] = np.array(new_img)
                    n+=1
    return X_val
