import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

idx = 106

def load_data(path):
    images = sorted(glob(os.path.join(path, "images/sample.png")))
    masks = sorted(glob(os.path.join(path, "masks/mask.png")))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path, augment=True):
    global idx
    H = 128
    W = 128

    for x, y in tqdm(zip(images, masks), total=len(images)):

        print(x)
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = CenterCrop(100, 100, p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x1, x2, x3, x4, x5]
            save_masks =  [y, y1, y2, y3, y4, y5]
        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """

        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            tmp_img_name = f"{image_name}.{image_extn}"
            tmp_mask_name = f"{mask_name}.{mask_extn}"

            os.mkdir('train/img'+str(idx))
            os.mkdir('train/img' + str(idx)+'/images')
            os.mkdir('train/img' + str(idx) + '/masks')
            image_path = os.path.join('train/img' + str(idx)+'/images', tmp_img_name)
            mask_path = os.path.join('train/img' + str(idx)+'/masks', tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

if __name__ == "__main__":
    for i in range(1, 106):
        path = "train/img" + str(i) + "/"
        images, masks = load_data(path)

        create_dir("new_data/images")
        create_dir("new_data/masks")
        augment_data(images, masks, "new_data", augment=True)

        images, masks = load_data("new_data/")