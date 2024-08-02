import argparse
import sys
sys.path.append('core')
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from featup.util import norm, unnorm, pca
from featup.plotting import plot_feats, plot_lang_heatmaps
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
DEVICE = 'cuda'

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', div = 8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // div) + 1) * div - self.ht) % div
        pad_wd = (((self.wd // div) + 1) * div - self.wd) % div
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    
transform = T.Compose([
        T.ToTensor(),
        norm
    ])
def load_image(path):
    image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    return image
def save_feature_image(feature, save_file):
    [feats_pca], _ = pca([feature.unsqueeze(0)])
    plt.imsave(save_file, feats_pca[0].permute(1,2,0).detach().cpu().numpy())
    
def save_images_as_video(images, output_file):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()
    
    dino_out_dir = os.path.join(args.data_dir, 'dino')
    os.makedirs(dino_out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(args.data_dir, 'color', '*')))
    num_imgs = len(img_files)
    
    sample = load_image(img_files[0])
    padder = InputPadder(sample.shape,div=14)
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(DEVICE)
    print("Extracting DINOv2 features")
    images = []
    features = []
    pca_mat = None
    pca_mat_save = None
    for i in tqdm(range(num_imgs)):
        imfile = img_files[i]
        image = load_image(imfile)
        image = padder.pad(image)[0]
        h,w = image.shape[-2:]
        save_file = os.path.join(dino_out_dir,
                                         '{}_norm.npy'.format(os.path.splitext(os.path.basename(imfile))[0]))
        lr_feats = upsampler.model(image)[0]
        hr_feats = upsampler(image)[0]
        
        hr_feats = F.interpolate(hr_feats[None], size=(h, w), mode='bilinear', align_corners=True)
        hr_feats = padder.unpad(hr_feats)
        hr_feats = hr_feats/ hr_feats.norm(dim=1, keepdim=True)
        [feats_pca], pca_mat = pca([hr_feats],fit_pca=pca_mat)
        [feats_pca_save], pca_mat_save = pca([hr_feats],dim=32,fit_pca=pca_mat_save)
        # feats_pca_save = feats_pca_save / feats_pca_save.norm(dim=1, keepdim=True)
        images.append((feats_pca[0].permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8))
        features.append(feats_pca_save.squeeze().detach().cpu().numpy())
        # np.save(save_file, hr_feats.squeeze().detach().cpu().numpy())
        del lr_feats, hr_feats, image
        torch.cuda.empty_cache()
    save_images_as_video(images, os.path.join(dino_out_dir, 'dino_feats.mp4'))
    np.save(os.path.join(dino_out_dir, 'dino_feats_pca.npy'), np.stack(features))
    

    