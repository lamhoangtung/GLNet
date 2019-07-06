import glob
import os
import shutil
from random import shuffle

data_path = '/mnt/data/linus/2DOCR/data/all_prj/'

train_path = os.path.join(data_path, 'train', 'images')
train_img = glob.glob(os.path.join(train_path, '*.png'))

def move_file(img_list, set_type):
    for old_img in img_list:
        new_img = old_img.replace('/train/','/{}/'.format(set_type))
        old_mask = old_img.replace('.png', '_mask.png').replace('/images/','/mask/')
        new_mask = old_mask.replace('/train/','/{}/'.format(set_type))
        shutil.move(old_img, new_img)
        shutil.move(old_mask, new_mask)

shuffle(train_img)
val_img = train_img[:107]
train_img = train_img[107:]
shuffle(train_img)
test_img = train_img[:35]
train_img = train_img[35:]
move_file(val_img, 'val')
move_file(test_img, 'test')
