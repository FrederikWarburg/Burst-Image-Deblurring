#/usr/bin/env python
from __future__ import print_function

from model import UNet
import os
from torch.utils import data
import numpy as np
import torch.optim as optim
from tools import Dataset
from burstloss import BurstLoss
import torch
import argparse
from cycliclr import CyclicLR
from helpers import get_newest_model, make_im
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from radam import RAdam

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', metavar='bs', type=int, default=2)
    parser.add_argument('--path', type=str, default='../../data')
    parser.add_argument('--results', type=str, default='../../results/model')
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr_decay', type=float, default=0.99997)
    parser.add_argument('--kernel_lvl', type=float, default=1)
    parser.add_argument('--noise_lvl', type=float, default=1)
    parser.add_argument('--motion_blur', type=bool, default=False)
    parser.add_argument('--homo_align', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()

    print()
    print(args)
    print()

    if not os.path.isdir(args.results): os.mkdir(args.results)

    PATH = args.results
    if not args.resume:
        f = open(PATH + "/param.txt", "a+")
        f.write(str(args))
        f.close()

    writer = SummaryWriter(PATH + '/runs')

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else "cpu")

    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': args.nw}

    # Generators
    print('Initializing training set')
    training_set = Dataset(args.path + '/train/', args.max_images,
                           args.kernel_lvl, args.noise_lvl, args.motion_blur, args.homo_align)
    training_generator = data.DataLoader(training_set, **params)

    print('Initializing validation set')
    validation_set = Dataset(args.path + '/test/',  args.val_size,
                             args.kernel_lvl, args.noise_lvl, args.motion_blur, args.homo_align)

    validation_generator = data.DataLoader(validation_set, **params)

    # Model
    model = UNet(in_channel=3,out_channel=3)
    if args.resume:
        models_path = get_newest_model(PATH)
        print('loading model from ', models_path)
        model.load_state_dict(torch.load(models_path))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Loss + optimizer
    criterion = BurstLoss()
    optimizer = RAdam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size = 8 // args.bs, gamma = args.lr_decay)
    if args.resume:
        n_iter = np.loadtxt(PATH + '/train.txt', delimiter=',')[:, 0][-1]
    else:
        n_iter = 0

    # Loop over epochs
    for epoch in range(args.epochs):
        train_loss = 0.0

        # Training
        model.train()
        for i, (X_batch, y_labels) in enumerate(training_generator):
            # Alter the burst length for each mini batch

            burst_length = np.random.randint(2,9)
            X_batch = X_batch[:,:burst_length,:,:,:]

            # Transfer to GPU
            X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X_batch)
            loss = criterion(pred, y_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.detach().cpu().numpy()
            writer.add_scalar('training_loss', loss.item(), n_iter)

            if i % 100 == 0 and i > 0:
                loss_printable = str(np.round(train_loss,2))

                f = open(PATH + "/train.txt", "a+")
                f.write(str(n_iter) + "," + loss_printable + "\n")
                f.close()

                print("training loss ", loss_printable)

                train_loss = 0.0

            if i % 1000 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), os.path.join(PATH,'model_' + str(int(n_iter)) + '.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(PATH, 'model_' + str(int(n_iter)) + '.pt'))

            if i % 1000 == 0:
                # Validation
                val_loss = 0.0
                with torch.set_grad_enabled(False):
                    model.eval()
                    for v, (X_batch, y_labels) in enumerate(validation_generator):
                        # Alter the burst length for each mini batch

                        burst_length = np.random.randint(2, 9)
                        X_batch = X_batch[:, :burst_length, :, :, :]

                        # Transfer to GPU
                        X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)

                        # forward + backward + optimize
                        pred = model(X_batch)
                        loss = criterion(pred, y_labels)

                        val_loss += loss.detach().cpu().numpy()

                        if v < 5:
                            im = make_im(pred, X_batch, y_labels)
                            writer.add_image('image_' + str(v), im, n_iter)

                    writer.add_scalar('validation_loss', val_loss, n_iter)

                    loss_printable = str(np.round(val_loss, 2))
                    print('validation loss ', loss_printable)

                    f = open(PATH + "/eval.txt", "a+")
                    f.write(str(n_iter) + "," + loss_printable + "\n")
                    f.close()

            n_iter += args.bs

if __name__ == "__main__":
    main()