import numpy as np
import cv2
import imageio


def create_gif(out_path, images, fps=24):
    imageio.mimwrite(out_path, images, fps=fps)
    print(f'Save gif to {out_path}')


def preprocess(img):
    img = np.squeeze(img)[-1][..., np.newaxis]
    img = np.repeat(img.astype(np.uint8), 3, axis=2)
    img = cv2.resize(img, (210, 160), interpolation=cv2.INTER_CUBIC)
    return img
