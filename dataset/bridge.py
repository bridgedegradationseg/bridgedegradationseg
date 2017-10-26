from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os


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
