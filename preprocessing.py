import glob
import os.path

import cv2
import tqdm

from utils import CLASSES


def median_filter(img):
    return cv2.medianBlur(img, 5)


def contrast_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    lab_planes[0] = clahe_.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(path):
    img = cv2.imread(path)
    mf = median_filter(img)
    ce = contrast_enhance(mf)
    return {"Median Filtered": mf, "Contrast Enhanced": ce}


if __name__ == "__main__":
    DATA_DIR = "Data/data"
    SAVE_DIR = "Data/preprocessed"
    for c in CLASSES:
        images_list = sorted(glob.glob(os.path.join(DATA_DIR, c, "*.jpg")))
        save_dir = os.path.join(SAVE_DIR, c)
        os.makedirs(save_dir, exist_ok=True)
        for img_path in tqdm.tqdm(
                images_list, desc="[INFO] Preprocessing Images From => {0} :".format(c)
        ):
            s_path = os.path.join(save_dir, os.path.basename(img_path))
            pp = preprocess(img_path)
            cv2.imwrite(s_path, pp["Contrast Enhanced"])
