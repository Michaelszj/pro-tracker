import torch 
import os
import numpy as np
import pickle
from PIL import Image
import glob
import torch.nn.functional as F
class LargeTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_file_paths, device='cuda'):
        self.tensor_file_paths = tensor_file_paths
        self.device = device

    def __len__(self):
        return len(self.tensor_file_paths)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(np.load(self.tensor_file_paths[idx])).to(self.device)
        return tensor
    
    
    
def resize_tensor(tensor, size):
    return F.interpolate(tensor.permute(0, 3, 1, 2)/255., size=size, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)*255.

class pklDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_file_path: str, device='cuda'):
        self.pkl_file_path = pkl_file_path
        self.device = device
        with open(self.pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.tapnet_path = os.path.join(os.path.dirname(self.pkl_file_path), 'save_dict_tapnet.pkl')
        try:
            with open(self.tapnet_path, 'rb') as f:
                self.tapnet_data = pickle.load(f)
        except:
            self.tapnet_data = None
            print('No TAPNet data found')
        self.seqlen = len(self.data.keys())
        self.curidx = 0
        self.seqnames = list(self.data.keys())
        self.tapnames = list(self.tapnet_data.keys())
        self.switch_to(self.curidx)
        
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
        H, W = self.data[self.seqnames[self.curidx]]['video'].shape[1:3]
        new_size = (480,(480*W)//H)
        self.data[self.seqnames[self.curidx]]['video'] = resize_tensor(resize_tensor(torch.from_numpy(self.data[self.seqnames[self.curidx]]['video']).to(self.device).float()
                                                                                     ,(256,256)),new_size).round().int().cpu().numpy()
        

    def __len__(self):
        return self.curlen()

    def __getitem__(self, idx):
        
        return self.data[self.seqnames[self.curidx]]['video'][idx]
    
    def get_gt(self):
        return self.data[self.seqnames[self.curidx]]['points'], self.data[self.seqnames[self.curidx]]['occluded']
    
    def get_tapnet(self):
        if self.tapnet_data is not None:
            return self.tapnet_data[self.tapnames[self.curidx]]['track'], self.tapnet_data[self.tapnames[self.curidx]]['occlusion']
        else:
            return None, None
    
    
    
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
    
    