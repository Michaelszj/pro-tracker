import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import ipdb
dir = 'flow_vis/'
for file in glob.glob('flow_vis/*.pt'):
    flow = torch.load(file)
    flow = flow.cpu().numpy()
    flow = np.transpose(flow, (1, 2, 0))
    flow = flow / np.array([abs(flow[:,:,0]).max(),abs(flow[:,:,1]).max()])
    flow = flow * 0.5 + 0.5
    

    flow = np.concatenate((flow, np.zeros((flow.shape[0], flow.shape[1], 1))), axis=2)
    flow = flow[:,:,[0,2,1]]
    # flow = flow + 0.8
    flow = flow 
    flow[flow<0] = 0
    #import ipdb; ipdb.set_trace()
    flow = (flow*255.).astype(np.uint8)
    # flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
    Image.fromarray(flow).save(file.replace('.pt', '.png'))
    print(file)