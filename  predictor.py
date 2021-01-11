import os 
import sys
import time 

import cv2
import torch 
import dlib
import numpy as np
from PIL import Image
from argparse import Namespace
import torchvision.transforms as transforms


from models.psp import pSp
from utils.common import tensor2im
from scripts.align_all_parallel import align_face



class Predictor:
    def __init__(self, 
                    model_path, 
                    landmark_predictor_path, 
                    save_res_path, 
                    latent_dirs_p):

        self.inference_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.landmark_prediction_path = landmark_predictor_path

        #load pretrained model 
        ckpt = torch.load(model_path, map_location='cpu')
        self.opts = ckpt['opts']
        self.opts['checkpoint_path'] = model_path
        self.opts['test_batch_size'] = 1
        
        if 'learn_in_w' not in self.opts:
            self.opts['learn_in_w'] = False

        self.opts = Namespace(**self.opts)
        self.net = pSp(self.opts)
        self.net.eval()
        self.net.cuda()

        #dlib face landmarks predictor
        self.predictor = dlib.shape_predictor(landmark_predictor_path)
        
        #StyleGan2 Latent Directions 
        self.age_direction = np.load(os.path.join(latent_dirs_p, 'age.npy')).astype('float32')
        self.smile_direction = np.load(os.path.join(latent_dirs_p, 'smile.npy')).astype('float32')
        self.gender_direction = np.load(os.path.join(latent_dirs_p, 'gender.npy')).astype('float32')
    
    def __preprocess_input(self, image):
        pass

    def __get_latent_directions(self, image, net):
        pass

    def __move_latent_directions(self, latent_vector, latent_directions, coeff, net):
        pass

    def get_prediction(self):
        pass


if __name__ == "__main__":
    pass