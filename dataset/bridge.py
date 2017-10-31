from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
import keras
from keras.preprocessing.image import ImageDataGenerator

#carefull, the samples will be ordered after this
#num_samples_factor = 2 means we will have twice as many samples of class class_label
def augment_set(x_samples, y_samples, num_samples_factor = 2.0, class_label = 1):
    
    if x_samples.shape[0] != y_samples.shape[0]:
        print('number of samples doesnt equal number of labels, something is wrong!\n')
    
    #create array of only the samples of the selected class
    x_myclass = x_samples[y_samples[:,0] == class_label,...]
    y_myclass = y_samples[y_samples[:,0] == class_label,...]
    
    #differnt array for the samples of the other class
    #if memory is critical this line can also be moved to the only use of x_remainder, but its mor readable this way
    x_remainder = x_samples[y_samples[:,0] != class_label,...]
    y_remainder = y_samples[y_samples[:,0] != class_label,...]
    
    datagen = ImageDataGenerator(
    rotation_range=90,
    zoom_range=[1, 1.4],#[lower, upper] = [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    preprocessing_function=None) # maybe include the strech histogram here

    datagen.fit(x_myclass, augment=True, seed=42)
    
    num_samples_to_draw = int(num_samples_factor * x_myclass.shape[0])
    
    #reserve space for the new samples (shape is the same as before, exept with a new number of samples)
    x_augmented = np.zeros((((x_remainder.shape[0] + num_samples_to_draw),) + x_myclass.shape[1:]), dtype='uint8')
    y_augmented = np.zeros((((y_remainder.shape[0] + num_samples_to_draw),) + y_myclass.shape[1:]), dtype='uint8')
    
    x_augmented[0:x_remainder.shape[0],...] = x_remainder
    y_augmented[0:y_remainder.shape[0],...] = y_remainder
    
    #you could set batchsize to num_samples_to_draw, but that might be a way higher than the developers probalby had in mind when writing the function, so it might cause problems
    batchsize = 32
    i=0
    offset = x_remainder.shape[0]
    for x_batch, y_batch in datagen.flow(x_myclass, y_myclass, batch_size=batchsize):
        if x_batch.shape[0] == batchsize:
            x_augmented[offset+i : offset+i+x_batch.shape[0], ...] = x_batch
            y_augmented[offset+i : offset+i+y_batch.shape[0], ...] = y_batch
        else:
            print('unexpected batchsize: ' , x_batch.shape[0] , ' instead of ' , batchsize , '  i: ' , i , 'numsamplestodraw: ' , num_samples_to_draw ,'\n')
            x_augmented[offset+i : offset+i+x_batch.shape[0], ...] = x_batch
            y_augmented[offset+i : offset+i+y_batch.shape[0], ...] = y_batch
        i += 1
        if i >= int(num_samples_to_draw/batchsize):
            break
    #the last ones to fill up the remainder thats not devisible by batchsize
    x_batch, y_batch = datagen.flow(x_myclass, y_myclass, batch_size=batchsize)
    x_augmented[-(num_samples_to_draw-int(num_samples_to_draw/batchsize)), ...] = x_batch[-(num_samples_to_draw-int(num_samples_to_draw/batchsize)), ...]
    y_augmented[-(num_samples_to_draw-int(num_samples_to_draw/batchsize)), ...] = y_batch[-(num_samples_to_draw-int(num_samples_to_draw/batchsize)), ...]
    
    return x_augmented, y_augmented


def load_data(percent_of_damage=0.2):
    root_dir = '/src/workspace/dataset/'
    dataset = 'bridge_masks_patches/'
    patch_dir = '100/'
    deck_dirs = ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']
    image_list, label_list = [], []
    data_dict = {}
    img_id = 0
    for deck_dir in deck_dirs:
        mask_fpath = os.path.join(root_dir, dataset, patch_dir, deck_dir)
        image_dirs = os.listdir(mask_fpath)
        for image_dir in image_dirs:
            image_names = sorted(
                os.listdir(os.path.join(mask_fpath, image_dir))
            )
            for image_name in image_names:
                img_fpath = os.path.join(
                    root_dir,
                    'images_patches',
                    patch_dir,
                    deck_dir
                )
                img_path = os.path.join(img_fpath, image_dir, image_name)
                mask_path = os.path.join(mask_fpath, image_dir, image_name)
                img = np.asarray(Image.open(img_path))
                mask = np.asarray(Image.open(mask_path))
                damage_pixels = (mask / 255).sum()
                if damage_pixels > percent_of_damage * mask.size:
                    label = 1
                else:
                    label = 0
                data_dict[img_id] = {'path' : img_path, 'mask_path' : mask_path,
                                'img' : img, 'mask' : mask, 'label' : label }
                img_id += 1

    train_set_idcs, test_set_idcs = train_test_split(list(data_dict.keys()), test_size=0.33)

    x_train = []
    y_train = []
    for idx in train_set_idcs:
        x_train.append(data_dict[idx]['img'])
        y_train.append(data_dict[idx]['label'])

    x_test = []
    y_test = []
    for idx in test_set_idcs:
        x_test.append(data_dict[idx]['img'])
        y_test.append(data_dict[idx]['label'])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)[:,np.newaxis]
    y_test = np.array(y_test)[:,np.newaxis]
    return data_dict, train_set_idcs, test_set_idcs, x_train, y_train, \
            x_test, y_test


if __name__ == '__main__':
    data_dict, train_set_idcs, test_set_idcs, x_train, y_train, x_test, y_test = load_data()
