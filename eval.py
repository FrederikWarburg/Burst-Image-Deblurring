import os
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import numpy as np
from tools import Dataset
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from model import UNet
from helpers import get_newest_model, load_namespace
import pandas as pd
import random
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=list, default=['../../results/model'])
    parser.add_argument('--out_path', type=str, default='.')
    parser.add_argument('--NUMBER_OF_IMAGES', type=int, default=5000)
    parser.add_argument('--NUMBER_OF_PLOTS', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--KERNEL_LVL', type=float, default=3)
    parser.add_argument('--NOISE_LVL', type=float, default=1)
    parser.add_argument('--MOTION_BLUR', type=bool, default=True)
    parser.add_argument('--HOMO_ALIGN', type=bool, default=True)
    parser.add_argument('--model_iter', type=int, default=None)
    args = parser.parse_args()

    print()
    print(args)
    print()

    # Evaluation metric parameters
    SSIM_window_size = 3

    dict_ = {}
    for e, exp_path in enumerate(args.exp_paths):

        if args.model_iter == None:
            model_path = get_newest_model(exp_path)
        else:
            model_path = os.path.join(exp_path, args.model_iter)

        model_name = os.path.split(model_path)[1]
        name = str(e) + '_' + model_name.replace('.pt', '')

        dict_[name] = {}
        if not os.path.isdir((os.path.join(args.output_path,name))):
            os.mkdir(os.path.join(args.output_path,name))

        model = UNet(in_channel=3, out_channel=3)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        model = model.to(device)

        # Parameters
        params = {'batch_size': 1,
                  'shuffle': True,
                  'num_workers': 0}

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Generators
        data_set = Dataset('../../data/test/', max_images=args.NUMBER_OF_IMAGES, kernel_lvl=args.KERNEL_LVL, noise_lvl=args.NOISE_LVL, motion_blur_boolean=args.MOTION_BLUR, homo_align=args.HOMO_ALIGN)
        data_gen = data.DataLoader(data_set, **params)

        # evaluation
        evaluationData = {}

        for i, (X_batch, y_labels) in enumerate(data_gen):
            # Alter the burst length for each mini batch

            burst_length = np.random.randint(2, 9,)
            X_batch = X_batch[:, :burst_length, :, :, :]

            # Transfer to GPU
            X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)

            with torch.set_grad_enabled(False):
                model.eval()
                pred_batch = model(X_batch)

            evaluationData[str(i)] = {}
            for j in range(params['batch_size']):
                evaluationData[str(i)][str(j)] = {}

                y_label = y_labels[j, :, :, :].detach().cpu().numpy().astype(int)
                pred = pred_batch[j, :, :, :].detach().cpu().numpy().astype(int)

                y_label = np.transpose(y_label, (1, 2, 0))
                pred = np.transpose(pred, (1, 2, 0))
                pred = np.clip(pred, 0, 255)

                if i < args.NUMBER_OF_PLOTS and j == 0:
                    plt.figure(figsize=(20, 5))
                    plt.subplot(1, 2 + len(X_batch[j, :, :, :, :]), 1)
                    plt.imshow(y_label)
                    plt.axis('off')
                    plt.axis('off')
                    plt.title('GT')

                    plt.subplot(1, 2 + len(X_batch[j, :, :, :, :]), 2)
                    plt.imshow(pred)
                    plt.axis('off')
                    plt.title('Pred')

                burst_ssim = []
                burst_psnr = []
                for k in range(len(X_batch[j, :, :, :, :])):
                    x = X_batch[j, k, :, :, :].detach().cpu().numpy().astype(int)
                    burst = np.transpose(x, (1, 2, 0))

                    if i < args.NUMBER_OF_PLOTS and j == 0:
                        plt.subplot(1, 2 + len(X_batch[j, :, :, :, :]), 3 + k)
                        plt.imshow(burst)
                        plt.axis('off')
                        plt.title('Burst ' + str(k))

                    burst_ssim.append(ssim(y_label.astype(float), burst.astype(float), multichannel=True, win_size = SSIM_window_size))
                    burst_psnr.append(psnr(y_label, burst))

                SSIM = ssim(pred.astype(float), y_label.astype(float), multichannel=True, win_size = SSIM_window_size)
                PSNR = psnr(pred, y_label)
                if i < args.NUMBER_OF_PLOTS and j == 0:
                    plt.savefig(os.path.join(args.output_path,name,str(i) + '.png'), bbox_inches = 'tight', pad_inches = 0)
                    plt.cla(); plt.clf(); plt.close()

                evaluationData[str(i)][str(j)]['SSIM'] = SSIM
                evaluationData[str(i)][str(j)]['PSNR'] = PSNR
                evaluationData[str(i)][str(j)]['length'] = burst_length
                evaluationData[str(i)][str(j)]['SSIM_burst'] = burst_ssim
                evaluationData[str(i)][str(j)]['PSNR_burst'] = burst_psnr

            if i % 500 == 0 and i > 0:
                print(i)

        #######
        # Save Results
        #######

        x_ssim, y_ssim, y_max_ssim = [], [], []
        x_psnr, y_psnr, y_max_psnr = [], [], []

        for i in evaluationData:
            for j in evaluationData[i]:
                x_ssim.append(evaluationData[i][j]['length'])
                y_ssim.append(evaluationData[i][j]['SSIM'])
                y_max_ssim.append(evaluationData[i][j]['SSIM'] - max(evaluationData[i][j]['SSIM_burst']))

                x_psnr.append(evaluationData[i][j]['length'])
                y_psnr.append(evaluationData[i][j]['PSNR'])
                y_max_psnr.append(evaluationData[i][j]['PSNR'] - max(evaluationData[i][j]['PSNR_burst']))

        method = [name]*len(x_ssim)
        dict_[name]['ssim'] = pd.DataFrame(np.transpose([x_ssim, y_ssim, y_max_ssim, method]), columns = ['burst_length', 'ssim', 'max_pred_ssim', 'method'])
        dict_[name]['psnr'] = pd.DataFrame(np.transpose([x_psnr, y_psnr, y_max_psnr, method]), columns = ['burst_length', 'psnr', 'max_pred_psnr', 'method'])

        dict_[name]['ssim'].to_csv(os.path.join(args.output_path,'ssim_' + name + '.csv'))
        dict_[name]['psnr'].to_csv(os.path.join(args.output_path,'psnr_' + name + '.csv'))


if __name__ == "__main__":
    main()