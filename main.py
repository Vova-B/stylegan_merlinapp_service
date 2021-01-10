import os
import sys
import time

sys.path.append('.')

import cv2
import dlib
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from argparse import Namespace
import torchvision.transforms as transforms

from models.psp import pSp
from utils.common import tensor2im, log_input_image
from scripts.align_all_parallel import align_face

PRETRAINED_WEIGHT = 'psp_ffhq_encode.pt'
TEST_DATA_DIR = './test_data'
RESULT_DATA_PATH = './test_data_res'
LATENT_DIRECTION_PATH = './latent_directions'
FACE_LANDMARKS_PATH = './shape_predictor_68_face_landmarks.dat'
ALIGN_IMAGE = True
COEFFS = [-15., -12., -9., -6., -3., 0., 3., 6., 9., 12.]

AGE_LATENT_DIR = './latent_directions/age.npy'
SMILE_LATENT_DIR = './latent_directions/smile.npy'
GENDER_LATENT_DIR = './latent_directions/gender.npy'

INFERENCE_DATA_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def main():
    test_images_paths = [os.path.join(TEST_DATA_DIR, img) for img in os.listdir(TEST_DATA_DIR) if
                         img.split('.')[1] == 'jpg' or img.split('.')[1] == 'png']
    num_images = len(test_images_paths)
    print('{} Images'.format(num_images))

    # Load pretraiden weights
    ckpt = torch.load(PRETRAINED_WEIGHT, map_location='cpu')

    opts = ckpt['opts']
    print(opts)

    # Update training options
    opts['checkpoint_path'] = PRETRAINED_WEIGHT
    opts['test_batch_size'] = 1
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded')

    predictor = dlib.shape_predictor(FACE_LANDMARKS_PATH)

    for i in tqdm(range(num_images)):
        start_time = time.time()
        res_path = os.path.join(RESULT_DATA_PATH, str(i))
        aligned_image = align_face(filepath=test_images_paths[i], predictor=predictor)
        aligned_image = aligned_image.convert('RGB')
        inp_img = INFERENCE_DATA_TRANSFORMS(aligned_image)

        with torch.no_grad():
            inference_time = time.time()
            res_img, latents_dir = net(inp_img.to('cuda').float().unsqueeze(0), randomize_noise=False,
                                       return_latents=True)
            print('Inference time: {}'.format(time.time() - inference_time))
            res_img = tensor2im(res_img[0])

        if not os.path.exists(os.path.join(RESULT_DATA_PATH, str(i))):
            os.mkdir(os.path.join(RESULT_DATA_PATH, str(i)))

        res_img.save(os.path.join(res_path, '{}-encode.png'.format(i)))
        np.save(os.path.join(res_path, '{}-latent-dir.npy'.format(i)), latents_dir.data.cpu().numpy()[0])


if __name__ == "__main__":
    main()
