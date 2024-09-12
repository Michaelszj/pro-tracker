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
    # input tensor: (N, H, W, C)
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
        self.current_data = None
        self.total_frames = 0
        self.switch_to(self.curidx)
        self.start_frame = 0
        self.direction = 1
        
        
        
    def curlen(self):
        return self.current_data.shape[0]
    
    def curname(self):
        return self.seqnames[self.curidx]
    
        
    def switch_to(self, idx):
        self.curidx = idx
        print('Switched to sequence {}'.format(self.seqnames[self.curidx]))
        H, W = self.data[self.seqnames[self.curidx]]['video'].shape[1:3]
        new_size = (480,(480*W)//H)
        self.current_data = resize_tensor(resize_tensor(torch.from_numpy(self.data[self.seqnames[self.curidx]]['video']).to(self.device).float()
                                                                                     ,(256,256)),new_size).round().int().cpu().numpy()
        self.total_frames = self.current_data.shape[0]
        
    def set_start_frame(self, frame, direction):
        self.switch_to(self.curidx)
        self.start_frame = frame
        self.direction = direction
        if direction == 1:
            self.current_data = self.current_data[self.start_frame:]
        else:
            self.current_data = self.current_data[:self.start_frame+1][::-1]

    def __len__(self):
        return self.curlen()

    def __getitem__(self, idx):
        
        return self.current_data[idx]
    
    def get_gt(self):
        points, occluded = self.data[self.seqnames[self.curidx]]['points'], self.data[self.seqnames[self.curidx]]['occluded']
        if self.direction == 1:
            return points[:,self.start_frame:], occluded[:,self.start_frame:]
        else:
            return points[:,:self.start_frame+1][:,::-1].copy(), occluded[:,:self.start_frame+1][:,::-1].copy()
        # return self.data[self.seqnames[self.curidx]]['points'], self.data[self.seqnames[self.curidx]]['occluded']
    
    def get_tapnet(self):
        if self.tapnet_data is not None:
            return self.tapnet_data[self.tapnames[self.curidx]]['track'], self.tapnet_data[self.tapnames[self.curidx]]['occlusion']
        else:
            return None, None
    
    
class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, benchmark, folder, device='cuda'):
        print('Loading benchmark from {}'.format(benchmark))
        print('Working with folder {}'.format(folder))
        self.benchmark = benchmark
        with open(benchmark, 'rb') as f:
            self.data = pickle.load(f)
        self.folder = folder
        self.idxes = list(self.data['videos'][i]['video_idx'] for i in range(len(self.data['videos'])))
        self.curidx = 0
        self.imgs = []
        self.gt_trajectory = None
        self.gt_occlusion = None
        self.H = self.W = 0
        self.start_frame = 0
        self.direction = 1
        self.device = device
        self.total_frames = 0
        # data
        # --- 'videos' (list)
        # ---  0~99    (dict)
        # 'video_idx', 'h', 'w', 'target_points', 'occluded', 'is_traj_occluded', 
        # 'is_fg', 'occlusion_length', 'max_occlusion_length', 'query_points'
        # target_points: dict[0,5,10-245] element: (N, 250, 2)
        # query_points: dict[0,5,10-245] element: (N, 2)
        # occluded: dict[0,5,10-245] element: (N, 250) bool
        
    def switch_to(self, idx):
        self.curidx = idx
        print('Switched to sequence {}'.format(self.curidx))
        self.imgs = sorted(glob.glob(os.path.join(self.folder, str(self.idxes[self.curidx]), 'video', '*.jpg')))
        self.gt_trajectory = np.array(self.data['videos'][self.curidx]['target_points'][self.start_frame]).astype(np.float32)
        self.gt_occlusion = np.array(self.data['videos'][self.curidx]['occluded'][self.start_frame])
        self.H, self.W = self.data['videos'][self.curidx]['h'], self.data['videos'][self.curidx]['w']
        self.gt_trajectory = self.gt_trajectory / np.array([self.W, self.H]).astype(np.float32)
        self.total_frames = len(self.imgs)
        # import pdb; pdb.set_trace()
        
    def __len__(self):
        return len(self.imgs)
    
    def curname(self):
        return str(self.idxes[self.curidx])
    
    def set_start_frame(self, frame, direction):
        self.switch_to(self.curidx)
        self.start_frame = frame
        self.direction = direction
        if direction == 1:
            self.imgs = self.imgs[self.start_frame:]
        else:
            self.imgs = self.imgs[:self.start_frame+1][::-1]
            
    def __getitem__(self, idx):
        imfile = self.imgs[idx]
        image = Image.open(imfile).convert('RGB')
        image = np.array(image).astype(np.float32)
        image = torch.from_numpy(image)[None,...].to(self.device)
        new_size = (480,(480*self.W)//self.H)
        # import pdb; pdb.set_trace()
        image = resize_tensor(image, new_size).round().int().cpu().numpy()
        return image[0]
            
    def get_gt(self):
        if self.direction == 1:
            return self.gt_trajectory[:,self.start_frame:], self.gt_occlusion[:,self.start_frame:]
        else:
            return self.gt_trajectory[:,:self.start_frame+1][:,::-1].copy(), self.gt_occlusion[:,:self.start_frame+1][:,::-1].copy()
        
    
    
    
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
    
    
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_file, device='cuda',type = 'feature'):
        self.device = device
        self.feature_file = feature_file
        self.features: torch.Tensor = self.load_features()# .permute(2,1,0,3,4)[0].float()
        self.type = type
        # print(type,' loaded')
        if type == 'mask':
            # import pdb; pdb.set_trace()
            
            self.features = self.features.permute(2,1,0,3,4)[0].contiguous()
            # print('permuted',self.features.device)
            self.features = self.features.cuda()
            # print('moved to cuda')
            self.features = self.features.float()
            # print('converted to float') 
            self.features = self.features.cpu()
            # print('moved to cpu')
            try:
                self.point_map_file = self.feature_file.replace('all_mask', 'point_map')
                self.point_map = torch.load(self.point_map_file)
            except:
                self.point_map = torch.arange(self.features.shape[1])
                
            # print('Point map loaded')
        # import pdb; pdb.set_trace()
        
        self.featurelen = len(self.features)
        self.start_frame = 0
        self.direction = 1
        
    def __len__(self):
        return self.featurelen

    def __getitem__(self, idx):
        return self.features[idx].to(self.device) # (C, H, W)

    
    def load_features(self):
        features = torch.load(self.feature_file)
        return features
    
    def set_start_frame(self, frame, direction):
        self.start_frame = frame
        self.direction = direction
        if direction == 1:
            self.features = self.features[self.start_frame:]
        else:
            self.features = self.features[:self.start_frame+1].flip(0)