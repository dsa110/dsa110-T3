import os

from mlmodel.unet import UNet
from mlmodel.helpers import makedir, load_data
import torch

from mlmodel.RFIMLsettings import data_path, out_path, model_path, batch_size

import numpy as np

import shutil
import argparse
import time
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def inference(model, data_loader):
    preds = []
    model.eval()
    model.cuda()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            data = data[0].cuda()
            outputs = model.forward_probs(data)
            preds.append(outputs)
    return preds

def main(data, prob_thresh=1e-7):
    segmentor = UNet()
    segmentor.load_state_dict(torch.load(model_path))
    # INFERENCE

    nfreq = data.shape[-1]
    print(f'Assuming NCHAN={nfreq}')
    
    if len(data.shape)==2:
        data = data.reshape(128, -1, 1, 2048).transpose((0, 2, 3, 1))
    elif len(data.shape)==3:
        data = data[:, None]
    elif len(data.shape)==4:
        pass
    else:
        print("Are you sure the data is the correct shape?")
        return 
    
    mean = np.median(np.mean(data, axis=(1, 2, 3)))
    std = np.median(np.std(data, axis=(1, 2, 3)))
    data = (data - mean) / std
    _, test_loader = load_data(data, batch_size=batch_size)
    print('Running Inference...')
    output = inference(segmentor, test_loader)
    mask_prob = torch.concatenate(output, dim=0).cpu().numpy()
    mask_prob = mask_prob.transpose((1, 2, 0, 3)).reshape(2048, -1)
    data = data.transpose((1, 2, 0, 3)).reshape(2048, -1)
    data[mask_prob > prob_thresh] = np.nan
    
    return mask_prob, data

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-d', type=str, default='0', help='(Device) The GPU ID to run training on')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.d
    
    makedir(out_path)

    segmentor = UNet()
    segmentor.load_state_dict(torch.load(model_path))

    # INFERENCE
    t1 = time.time()
    data = np.load(data_path)
    # data = data[None, None, :]
    t2 = time.time()
    mean = np.median(np.mean(data, axis=(1, 2, 3)))
    std = np.median(np.std(data, axis=(1, 2, 3)))
    data = (data - mean) / std
    _, test_loader = load_data(data, batch_size=batch_size)
    t3 = time.time()
    print('Running Inference...')
    output = inference(segmentor, test_loader)
    t4 = time.time()
    
    print('==================================================')
    print('Dataset Size: ', len(data))
    print('Loading Time (s):', t2 - t1)
    print('Preprocessing Time (s): ', t3 - t2)
    print('Inference Time (s): ', t4 - t3)
    print('Inference Rate (Images/s): ', len(data)/(t4-t3))
    print('==================================================')
    print('Writing out probability masks....')
    
    preds = torch.concatenate(output, dim=0).cpu().numpy()
    np.save(out_path + 'out.npy', preds)

    print('Done!')
