import torch 
import os
import numpy as np
import pickle
from PIL import Image
import glob
class LargeTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_file_paths, device='cuda'):
        self.tensor_file_paths = tensor_file_paths
        self.device = device

    def __len__(self):
        return len(self.tensor_file_paths)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(np.load(self.tensor_file_paths[idx])).to(self.device)
        return tensor
    
    
class pklDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_file_path, device='cuda'):
        self.pkl_file_path = pkl_file_path
        self.device = device
        with open(self.pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.seqlen = len(self.data.keys())
        self.curidx = 0
        self.seqnames = list(self.data.keys())
        
    def curlen(self):
        return self.data[self.seqnames[self.curidx]]['video'].shape[0]
    
    def curname(self):
        return self.seqnames[self.curidx]
    
    def to_cuda(self):
        try:
            self.data[self.seqnames[self.curidx]]['video'] = torch.from_numpy(self.data[self.seqnames[self.curidx]]['video']).to(self.device).float()
        except:
            self.data[self.seqnames[self.curidx]]['video'] = self.data[self.seqnames[self.curidx]]['video'].to(self.device)
        
    def to_cpu(self):
        self.data[self.seqnames[self.curidx]]['video'] = self.data[self.seqnames[self.curidx]]['video'].to('cpu')
        
    def switch_to(self, idx):
        self.curidx = idx
        print('Switched to sequence {}'.format(self.seqnames[self.curidx]))
        

    def __len__(self):
        return self.curlen()

    def __getitem__(self, idx):
        
        return self.data[self.seqnames[self.curidx]]['video'][idx]
    
    def get_gt(self):
        return self.data[self.seqnames[self.curidx]]['points'], self.data[self.seqnames[self.curidx]]['occluded']
    
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, save_memory = False, device='cuda'):
        self.img_dir = img_dir
        self.device = device
        self.img_files = sorted(glob.glob(os.path.join(img_dir, 'color', '*')))
        self.num_imgs = len(self.img_files)
        self.save_memory = save_memory
        self.images = self.load_imgs()
        if not self.save_memory:
            self.images = self.images.to(self.device)
        
        
        
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        if self.save_memory:
            return self.images[idx][None].to(self.device)
        else:
            return self.images[idx][None]

    
    def load_imgs(self):
        images = []
        for i in range(self.num_imgs):
            imfile = self.img_files[i]
            image = Image.open(imfile)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            images.append(image)
        images = torch.stack(images)
        return images
    
    