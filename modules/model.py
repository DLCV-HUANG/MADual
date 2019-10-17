# -*- coding: utf-8 -*-
import os
import scipy.io
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, vgg_model_path=None, c3d_model_path=None,K=1):
        super(MDNet, self).__init__()
        self.K = K
		#Note:        
		#Empirically proved that VGG-M with smaller stride is more conducive to the preformance fo VOT challenge. therefore
		#recommend utilizing VGG-M instead of VGG-S for VOT evaluation.
        # ---------------------------------VGG-S-----------------------
        self.layers_vgg = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=3, padding=1))),

            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),

            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear((512 + 512) * 7 * 7, 2048),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(2048, 2048),
                                  nn.ReLU()))]))
        # ---------------------------------VGG-S-----------------------
        #---------------------------------VGG-M-----------------------
        # self.layers_vgg = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
        #                             nn.ReLU(inplace=True),
        #                             LRN(),
        #                             nn.MaxPool2d(kernel_size=3, stride=2))),
        #     ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
        #                             nn.ReLU(inplace=True),
        #                             LRN(),
        #                             nn.MaxPool2d(kernel_size=3, stride=2))),
        #     ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
        #                             nn.ReLU(inplace=True))),
        #     ('fc4', nn.Sequential(nn.Linear(512 * 3 * 3, 512),
        #                           nn.ReLU(inplace=True))),
        #     ('fc5', nn.Sequential(nn.Dropout(0.5),
        #                           nn.Linear(512, 512),
        #                           nn.ReLU(inplace=True)))]))
        # ---------------------------------VGG-M-----------------------
        self.layers_c3d=nn.Sequential(OrderedDict([
            ('onv1_c3d', nn.Sequential(nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=(1, 1, 1)),
                                       nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))),
            ('onv2_c3d', nn.Sequential(nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1, 1, 1)),
                                       nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)))),
            ('conv3a_c3d', nn.Sequential(nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),nn.ReLU())),

            ('conv3b_c3d', nn.Sequential(
                                          nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                             nn.ReLU(),
                                          nn.MaxPool3d(kernel_size=(2, 2,2), stride=(2, 2,2)))),
            ('conv4a_c3d', nn.Sequential(nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),nn.ReLU())),

            ('conv4b_c3d', nn.Sequential(
                                         nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                            nn.ReLU(),
                                         nn.MaxPool3d(kernel_size=(2,2,2), stride=(2, 2,2)))),
            ('conv5a_c3d', nn.Sequential(nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),nn.ReLU())),

            ('conv5b_c3d', nn.Sequential(
                                     nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                        nn.ReLU(),
                                        #VGG-M
                                        #nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))))]))
                                        #VGG-S
                                     nn.MaxPool3d(kernel_size=(2, 1,1), stride=(2, 1,1))))]))
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(2048, 2)) for _ in range(K)])

        if vgg_model_path is not None:
            if os.path.splitext(vgg_model_path)[1] == '.mat':
                self.load_mat_model_vgg(vgg_model_path)

            elif os.path.splitext(vgg_model_path)[1] == '.pth':
                self.vgg_load_model(vgg_model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (vgg_model_path))
        if c3d_model_path is not None:
            if os.path.splitext(c3d_model_path)[1] == '.pickle':
                self.load_pkl_model_c3d(c3d_model_path, vgg_model_path)

            elif os.path.splitext(c3d_model_path)[1] == '.pth':
                self.c3d_load_model(c3d_model_path)

            else:
                raise RuntimeError("Unkown model format: %s" % (c3d_model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers_vgg.named_children():
            append_params(self.params, module, name)
        for name, module in self.layers_c3d.named_children():
            append_params(self.params, module, name)

        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d' % (k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:

                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x_vgg,x_c3d, k=0, in_layer_vgg='conv1',in_layer_c3d='onv1_c3d',out_layer='fc6',c3d_bra=True):
        run=False
        x_c3d1 = x_c3d
        for name, module in self.layers_c3d.named_children():
                        if name == in_layer_c3d:
                            run = True
                        if run :
                            x_c3d = module(x_c3d)

                            if name == 'onv1_c3d':
                                x_c3d1 = x_c3d
                            if name=='conv5b_c3d':
                                x_c3d = torch.squeeze(x_c3d,dim=2)
                                first_slicing = torch.chunk(x_c3d, len(x_c3d), dim=0)
                                x_fc = first_slicing[0]
                                for _ in range(int(len(x_vgg) / len(x_c3d)) - 1):
                                    x_fc = torch.cat((x_fc, first_slicing[0]), dim=0)
                                for i in range(len(x_c3d) - 1):
                                    for _ in range(int(len(x_vgg) / len(x_c3d))):
                                        x_fc = torch.cat((x_fc, first_slicing[i + 1]), dim=0)
                                x_c3d = x_fc
        run = False
        for name, module in self.layers_vgg.named_children():
            if name == in_layer_vgg:
                run = True
            if run:
                x_vgg = module(x_vgg)

                if name=='conv3':
                    if c3d_bra==True:
                        x_vgg=torch.cat((x_vgg,x_c3d),dim=1)

                    x_vgg=x_vgg.view(x_vgg.size(0),-1)
            if name == out_layer:
                    return x_vgg,x_c3d1

        x_vgg= self.branches[k](x_vgg)
        if out_layer == 'fc6':
            return x_vgg,x_c3d1
    def vgg_load_model(self, vgg_model_path):
        states = torch.load(vgg_model_path)
        shared_layers = states['shared_layers_vgg']
        self.layers_vgg.load_state_dict(shared_layers)

    def c3d_load_model(self, c3d_model_path):
        states = torch.load(c3d_model_path)
        shared_layers = states['shared_layers_c3d']
        self.layers_c3d.load_state_dict(shared_layers)

    # ---------------------------------VGG-S-----------------------
    def load_mat_model_vgg(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        for i, x in enumerate([0, 4, 7]):
            weight, bias = mat_layers[x]['weights'].item()[0]
            self.layers_vgg[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers_vgg[i][0].bias.data = torch.from_numpy(bias[:, 0])
    # ---------------------------------VGG-S-----------------------

    # ---------------------------------VGG-M-----------------------
    # def load_mat_model_vgg(self, matfile):
    #     mat = scipy.io.loadmat(matfile)
    #     mat_layers = list(mat['layers'])[0]
    #
    #     # copy conv weights
    #     for i in range(3):
    #         weight, bias = mat_layers[i * 4]['weights'].item()[0]
    #         self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
    #         self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
    # ---------------------------------VGG-M-----------------------
    def load_pkl_model_c3d(self, pklfile,matfile):
        pkl = torch.load(pklfile)
        self.layers_c3d[0][0].weight.data = pkl['conv1.weight']
        self.layers_c3d[0][0].bias.data = pkl['conv1.bias']
        self.layers_c3d[1][0].weight.data = pkl['conv2.weight']
        self.layers_c3d[1][0].bias.data = pkl['conv2.bias']
        self.layers_c3d[2][0].weight.data = pkl['conv3a.weight']
        self.layers_c3d[2][0].bias.data = pkl['conv3a.bias']
        self.layers_c3d[3][0].weight.data = pkl['conv3b.weight']
        self.layers_c3d[3][0].bias.data = pkl['conv3b.bias']
        self.layers_c3d[4][0].weight.data = pkl['conv4a.weight']
        self.layers_c3d[4][0].bias.data = pkl['conv4a.bias']
        self.layers_c3d[5][0].weight.data = pkl['conv4b.weight']
        self.layers_c3d[5][0].bias.data = pkl['conv4b.bias']
        self.layers_c3d[6][0].weight.data = pkl['conv5a.weight']
        self.layers_c3d[6][0].bias.data = pkl['conv5a.bias']
        self.layers_c3d[7][0].weight.data = pkl['conv5b.weight']
        self.layers_c3d[7][0].bias.data = pkl['conv5b.bias']

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:, 1]
        neg_loss = -F.log_softmax(neg_score)[:, 0]
        loss = pos_loss.sum() + neg_loss.sum()
        return loss

class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]

class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.data[0]


