import glob
import os

import cv2
import tqdm

from utils import CLASSES

DATA_DIR = 'Data/source'
SETS = ['First Set', 'Second Set']
SAVE_DIR = 'Data/data'


def prepare_data():
    IMG_COUNT = {v: 0 for v in CLASSES}
    for set_ in SETS:
        for set_dir in os.listdir(os.path.join(DATA_DIR, set_)):
            images_list = glob.glob(os.path.join(DATA_DIR, set_, set_dir, '*.jpg'))
            cname = CLASSES[0] if CLASSES[0] in set_dir else CLASSES[1]
            sd = os.path.join(SAVE_DIR, cname)
            os.makedirs(sd, exist_ok=True)
            for img_path in tqdm.tqdm(images_list, desc='[INFO] Preparing Images :'):
                img = cv2.imread(img_path)
                IMG_COUNT[cname] += 1
                sp = os.path.join(sd, 'IMG_{0}.jpg'.format(str(IMG_COUNT[cname]).zfill(3)))
                cv2.imwrite(sp, img)


if __name__ == '__main__':
    prepare_data()
