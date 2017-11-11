from model_FCN32 import FCN_Vgg16_32s
import os, sys
import dataset.bridge as bridge
import utils.preprocess as pp
import numpy as np
import keras
from PIL import Image
from keras.optimizers import SGD
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.utils import to_categorical

def load_full_images(datapath):
	bridge_masks_path = 'bridge_masks'
	bridge_images_path = 'images'
	imgs = []
	masks = []
	for root, dirs, files in os.walk(datapath):
            if root.split('/')[-1] in ['deck_a', 'deck_c', 'deck_d', 'deck_e']:
                print(root)
                for f in files:
                        mask = Image.open(os.path.join(root, f))
                        mask = crop_center(mask)
                        imgpath = os.path.join(root.split('/')[0], bridge_images_path, root.split('/')[-1], f.split('.')[-2]+'.jpg')
                        img = Image.open(imgpath)
                        img = crop_center(img)
                        mask = np.asarray(mask)
                        mask.flags.writeable = True
                        masks.append(mask)
                        img = np.asarray(img)
                        img.flags.writeable = True
                        imgs.append(img)
	return imgs, masks

def crop_center(img):
    half_width = img.size[0] // 2
    half_height = img.size[1] // 2

    w = 1024//2
    h = 768//2

    cropped_img = img.crop(
            (
                half_width - w,
                half_height - h,
                half_width + w,
                half_height + h
            )
        )
    return cropped_img


if __name__ == "__main__":

    # import data
    datapath = sys.argv[1]
    data_augmentation = False
    epochs = sys.argv[2]

    imgs, masks = load_full_images(datapath)
    np_imgs = [im/255 for im in imgs]
    #np_masks = [m[np.where(m > 0)]=1 for m in masks]
    # Correcting gray values from jpg compression. Save them again?
    for mask in masks:
        mask[mask > 0] = 1


    input_shape = imgs[0].shape
    # input_shape = (None, None, 3)
    model = FCN_Vgg16_32s(input_shape=input_shape, classes=2) 
    optimizer = SGD(lr=0.01, momentum=0.9)
    loss = sparse_categorical_crossentropy
    metric = sparse_categorical_accuracy
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    np_imgs = np.array(imgs)
    np_masks = np.array(masks)
    # np_masks = to_categorical(np_masks, 2)
    # np_masks = np.reshape(np_masks, (len(masks), masks[0].shape[0], masks[0].shape[1], 2))
    np_masks = np.expand_dims(np_masks, axis=-1)
