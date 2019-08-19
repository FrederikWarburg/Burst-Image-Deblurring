import torch
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class BurstLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(BurstLoss, self).__init__(size_average, reduce, reduction)

        self.reduction = reduction
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        prewitt_filter = 1 / 6 * np.array([[1, 0, -1],
                                           [1, 0, -1],
                                           [1, 0, -1]])

        self.prewitt_filter_horizontal = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                         kernel_size=prewitt_filter.shape,
                                                         padding=prewitt_filter.shape[0] // 2).to(device)

        self.prewitt_filter_horizontal.weight.data.copy_(torch.from_numpy(prewitt_filter).to(device))
        self.prewitt_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device))

        self.prewitt_filter_vertical = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                       kernel_size=prewitt_filter.shape,
                                                       padding=prewitt_filter.shape[0] // 2).to(device)

        self.prewitt_filter_vertical.weight.data.copy_(torch.from_numpy(prewitt_filter.T).to(device))
        self.prewitt_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device))

    def get_gradients(self, img):
        img_r = img[:, 0:1, :, :]
        img_g = img[:, 1:2, :, :]
        img_b = img[:, 2:3, :, :]

        grad_x_r = self.prewitt_filter_horizontal(img_r)
        grad_y_r = self.prewitt_filter_vertical(img_r)
        grad_x_g = self.prewitt_filter_horizontal(img_g)
        grad_y_g = self.prewitt_filter_vertical(img_g)
        grad_x_b = self.prewitt_filter_horizontal(img_b)
        grad_y_b = self.prewitt_filter_vertical(img_b)

        grad_x = torch.stack([grad_x_r[:, 0, :, :], grad_x_g[:, 0, :, :], grad_x_b[:, 0, :, :]], dim=1)
        grad_y = torch.stack([grad_y_r[:, 0, :, :], grad_y_g[:, 0, :, :], grad_y_b[:, 0, :, :]], dim=1)

        grad = torch.stack([grad_x, grad_y], dim=1)

        return grad

    def forward(self, input, target):
        input_grad = self.get_gradients(input)
        target_grad = self.get_gradients(target)

        return 0.1 * F.l1_loss(input, target, reduction=self.reduction) + F.l1_loss(input_grad, target_grad,
                                                                                    reduction=self.reduction)