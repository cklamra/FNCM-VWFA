import os
import cv2
import skimage
from PIL import Image
from numpy import asarray
import numpy as np
from tqdm import tqdm

# img = skimage.io.imread('wordsets_1000/train/aider/aider0.jpg')

noise_amount_list = [x/10 for x in range(0, 11)]

output_path = 'wordsets_1000_noise/'

for noise_amount in tqdm(noise_amount_list):
    for root, dirs, files in os.walk('wordsets_1000/train/'):
        if not dirs:
            output_path_nr = os.path.join(
                output_path,
                f"{noise_amount}",
                root[-1 * root[::-1].find('/'):])
            os.makedirs(output_path_nr, exist_ok=True)
            for file in files:
                img = Image.open(os.path.join(root, file))
                img_n = skimage.util.random_noise(
                    asarray(img),
                    mode='s&p',
                    clip=True,
                    amount=noise_amount)
                noise_img = np.array(255*img_n, dtype = 'uint8')
                pil_image=Image.fromarray(np.array(noise_img)).convert('L')
                pil_image.save(os.path.join(output_path_nr, file))

#img = Image.open('wordsets_1000/train/aider/aider0.jpg')
#img_n = skimage.util.random_noise(asarray(img), mode='s&p', clip=True, amount=0.05)
#noise_img = np.array(255*img_n, dtype = 'uint8')
#pil_image=Image.fromarray(np.array(noise_img)).convert('L')
#pil_image.save('aider0_noise.jpg')
#Image.fromarray(img_n).save('aider0_noise.jpg')
#cv2.imwrite('aider0_noise.png', img_n)

#for img in images:
#    image = io.imread('/home/onur/Downloads/datasets/LFW-a/lfw2/' + img)
#    image = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True)
#    noise_img = np.array(255*image, dtype = 'uint8')
#    pil_image=Image.fromarray(np.array(noise_img))
#    pil_image.save('/Users/eric/Desktop/goproject4/test.jpg')
