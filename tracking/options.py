# -*- coding: utf-8 -*-
#--------------------------------------------调参------------------
from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True
#--------------------------------------------调参------------------
path='/media/datou/c0d34b51-8a78-4364-bc19-35003bf01885/VOT/MAdual/code/Original/models_test1_0/con44_0/'
opts['model_path_vgg'] = path+'vgg.pth'
opts['model_path_c3d'] = path+'c3d.pth'
#--------------------------------------------调参------------------
opts['result']='../result_test0418/'
opts['dataset_path']='/media/datou/c0d34b51-8a78-4364-bc19-35003bf01885/pytorch/py-MDNet-master/dataset/OTB2015/'
opts['img_size_vgg'] = 107
opts['img_size_c3d'] = 120
opts['padding'] = 12
opts['batch_frames']=8
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.5
opts['overlap_bbreg'] = [0.6,1]
opts['scale_bbreg'] = [1, 2]

opts['lr_init'] = 0.0001
#--------------------------------------------调参------------------
opts['n_bbreg'] = 200
opts['maxiter_init'] = 30
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000

opts['n_bbreg_sixteen'] = 1000
opts['maxiter_sixteen'] = 15
opts['n_pos_sixteen'] = 500
opts['n_neg_sixteen'] = 5000

opts['n_frames_short'] = 24
opts['long_interval'] = 16
#--------------------------------------------调参------------------

opts['overlap_pos_init'] = [0.7,1]
opts['overlap_neg_init'] = [0, 0.5]

opts['lr_update'] = 0.0002
opts['maxiter_update'] = 15
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

opts['success_thr'] = 0#～～～～
opts['n_frames_long'] = 100


opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['lr_mult'] = {'fc6':10}
opts['ft_layers'] = ['fc']
opts['n_frame_c3d']=16
opts['topk']=5
