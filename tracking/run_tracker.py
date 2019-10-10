# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import sys
import torch
print('the version of pytorch is :', torch.__version__)

import torch.optim as optim
from torch.autograd import Variable
sys.path.insert(0, '../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
from utils import *
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
def gt_change(image,gt,c_k):
    x_cro,y_row=image.size
    x_len=gt[2]
    y_len=gt[3]
    gt_ch = []
    c1=float(c_k-1)/2
    gt_ch.append(np.maximum(gt[0]-c1*gt[2],0))
    gt_ch.append(np.maximum(gt[1]-c1*gt[3],0))
    gt_ch.append(np.minimum(c_k*x_len,x_cro))
    gt_ch.append(np.minimum(c_k*y_len,y_row))
    return gt_ch

def extract_regions(image, samples, crop_size,padding):
    regions = np.zeros((len(samples), crop_size, crop_size, 3), dtype='uint8')
    for i, sample in enumerate(samples):
        regions[i] = crop_image(image, sample, crop_size, padding, True)
    regions = regions.transpose(0, 3, 1, 2)
    regions = regions.astype('float32') - 128.
    return regions

def forward_samples(args,model, img_list,i_i,result_bb, samples,i_i_i, out_layer='conv3',c3d_bra=True,flag_flase=True):
    image = Image.open(img_list[i_i]).convert('RGB')
    image_a = image.resize((opts['img_size_vgg'], opts['img_size_vgg']))
    regions = np.zeros((1, opts['img_size_vgg'], opts['img_size_vgg'], 3), dtype='uint8')

    image_a = np.asarray(image_a)
    regions[0] = image_a
    image_c3d=np.empty((0,3,opts['img_size_c3d'],opts['img_size_c3d']))
    lin_a = []
    i_i_buffer = i_i
    while len(lin_a) < 16:
        for _ in range(16):
            if i_i_buffer > 0:
                lin_a.append(i_i_buffer)
                i_i_buffer = i_i_buffer - 1
            else:
                break
        if len(lin_a)<16:
            lin_a.append(0)
    if i_i_i!=0 and i_i>0:
        lin_a.reverse()

    else:
        print('lin_a',lin_a)

    c_k = 2
    #------------------------------目標+上下文--------------------------
    for len_lin in range(16):
        if i_i_i == 0 and i_i > 0:
            if lin_a[len_lin]==0:
                k_m = lin_a[len_lin]+1
                m_image=img_list[lin_a[len_lin]]
            else:
                k_m = lin_a[len_lin]
                m_image=img_list[lin_a[len_lin]]
        else:
            if len_lin==15 and lin_a[len_lin]!=0:
                k_m = lin_a[len_lin]-1
                m_image=img_list[lin_a[len_lin]]
            else:
                k_m = lin_a[len_lin]
                m_image=img_list[lin_a[len_lin]]



        image = Image.open(m_image).convert('RGB')
        gt_buffer = gt_change(image, result_bb[k_m], c_k)
        image_a = np.array(image)
        bbox_buffer = []
        bbox_buffer.append(gt_buffer)
        bbox_buffer1=np.array(bbox_buffer)
        image_c3d = np.concatenate((image_c3d, extract_regions(image_a, bbox_buffer1, opts['img_size_c3d'], opts['padding'])), axis=0)

    image_c3d = torch.from_numpy(image_c3d).float()
    image_c3d = image_c3d.view(1, 16, 3, opts['img_size_c3d'], opts['img_size_c3d'])
    image_c3d = torch.transpose(image_c3d, 2, 1)
    image_c3d = Variable(image_c3d)

    model.eval()
    image = Image.open(img_list[i_i_i]).convert('RGB')
    if flag_flase:
        extractor = RegionExtractor(image, samples, opts['img_size_vgg'], opts['padding'], opts['batch_test'])
    else:
        extractor = RegionExtractor(image, samples, opts['img_size_vgg'], 16, opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
            image_c3d = image_c3d.cuda()
        feat,x_c3d1 = model(regions,image_c3d, out_layer=out_layer,c3d_bra=c3d_bra)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)#
    return feats

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer_SGD = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer_SGD

def train(model, criterion, optimizer, pos_feats_vgg, neg_feats_vgg, maxiter, in_layer_vgg='fc4',in_layer_c3d='fc4',c3d_bra=True):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats_vgg.size(0))
    # print(pos_idx)
    neg_idx = np.random.permutation(neg_feats_vgg.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats_vgg.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats_vgg.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next].astype(np.float64)
        pos_cur_idx = pos_feats_vgg.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next].astype(np.float64)
        neg_cur_idx = neg_feats_vgg.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch

        batch_pos_feats_vgg = Variable(pos_feats_vgg.index_select(0, pos_cur_idx))
        batch_neg_feats_vgg = Variable(neg_feats_vgg.index_select(0, neg_cur_idx))
        # batch_pos_feats_c3d = Variable(pos_feats_c3d.index_select(0, pos_cur_idx))
        # batch_neg_feats_c3d = Variable(neg_feats_c3d.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                #print('bbbbbb:',batch_neg_feats_vgg[start:end].size())
                #buffer_=batch_neg_feats[start:end].view(1,1,1,1,-1)
                score,_= model(batch_neg_feats_vgg[start:end],None, in_layer_vgg=in_layer_vgg,in_layer_c3d=in_layer_c3d,c3d_bra=c3d_bra)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats_vgg = batch_neg_feats_vgg.index_select(0, Variable(top_idx))
            #batch_neg_feats_c3d = batch_neg_feats_c3d.index_select(0, Variable(top_idx))

            model.train()

        pos_score,_ = model(batch_pos_feats_vgg,None,in_layer_vgg=in_layer_vgg,in_layer_c3d=in_layer_c3d,c3d_bra=c3d_bra)
        neg_score,_ = model(batch_neg_feats_vgg,None,in_layer_vgg=in_layer_vgg,in_layer_c3d=in_layer_c3d,c3d_bra=c3d_bra)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

def run_mdnet(args,img_list, init_bbox, gt=None, savefig_dir='', display=False):
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    print(opts['model_path_vgg'])
    print(opts['model_path_c3d'])
    model = MDNet(opts['model_path_vgg'],opts['model_path_c3d'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    #-------------------------------------------回归---------------------------------------------------
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], [0.6,1], opts['scale_bbreg'])
    bbreg_feats_vgg= forward_samples(args,model, img_list,0, result_bb,bbreg_examples,0,c3d_bra=True,flag_flase=True)#只需要主干网络的特征图，所以c3d_bra=Flase
    bbreg = BBRegressor(image.size)

    bbreg.train(bbreg_feats_vgg, bbreg_examples, target_bbox)

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats_vgg = forward_samples(args,model, img_list,0,result_bb, pos_examples,0,c3d_bra=True,flag_flase=True)#因为只更新fc6,所以特征输出到fc5，主干与c3d相集合的地方
    first_pos_feats_vgg = pos_feats_vgg
    neg_feats_vgg= forward_samples(args,model, img_list,0,result_bb, neg_examples,0,c3d_bra=True,flag_flase=True)
    feat_dim_vgg = pos_feats_vgg.size(-1)
    train(model, criterion, init_optimizer, pos_feats_vgg,neg_feats_vgg,opts['maxiter_init'])

    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)
    pos_feats_all_vgg = [pos_feats_vgg[:opts['n_pos_update']]]#
    neg_feats_all_vgg = [neg_feats_vgg[:opts['n_neg_update']]]#
    spf_total = time.time() - tic

    overlap = 0.0
    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    ii = 0
    p0 = p3 = p5 = p7 = p9 = p10 = zero = sum_AC = 0
    flag_flase=0
    Flag_padding=True
    for i in range(1, len(img_list)):
        if (i // 16 == 1 and i%16==0) :
            image_bbreg = Image.open(img_list[0]).convert('RGB')
            bbreg_examples = gen_samples(SampleGenerator('uniform', image_bbreg.size, 0.3, 1.5, 1.1),
                                         result[0], opts['n_bbreg_sixteen'], opts['overlap_bbreg'], opts['scale_bbreg'])
            bbreg_feats = forward_samples(args, model, img_list, i-1, result_bb, bbreg_examples, 0, c3d_bra=True,flag_flase=Flag_padding)  #
            bbreg = BBRegressor(image_bbreg.size)
            bbreg.train(bbreg_feats, bbreg_examples, result[0])
        # #----------------------------------------------------------------------------------
            image = Image.open(img_list[0]).convert('RGB')
            pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                                       result[0], opts['n_pos_sixteen'], [0.6,1])

            neg_examples = np.concatenate([
                gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                            result[0], (opts['n_neg_sixteen']) // 2, opts['overlap_neg_init']),
                gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                            result[0], (opts['n_neg_sixteen']) // 2, opts['overlap_neg_init'])])
            neg_examples = np.random.permutation(neg_examples)

            # Extract pos/neg features
            pos_feats_vgg = forward_samples(args, model, img_list, i-1, result_bb, pos_examples, 0,c3d_bra=True,flag_flase=Flag_padding)  # 因为只更新fc6,所以特征输出到fc5，主干与c3d相集合的地方
            neg_feats_vgg = forward_samples(args, model, img_list, i-1, result_bb, neg_examples, 0, c3d_bra=True,flag_flase=Flag_padding)
            train(model, criterion, init_optimizer, pos_feats_vgg, neg_feats_vgg, opts['maxiter_sixteen'])
            del pos_feats_vgg
            del neg_feats_vgg
        #----------------------------------------------------------------------------------



        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores= forward_samples(args,model, img_list,i,result_bb, samples, i,out_layer='fc6',flag_flase=Flag_padding)
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)
        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            flag_flase+=1
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats_vgg= forward_samples(args,model,img_list,i,result_bb, bbreg_samples,i,c3d_bra=True,flag_flase=Flag_padding)#因为是回归所以只需要输出主干网络的特征图
            bbreg_samples = bbreg.predict(bbreg_feats_vgg, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
            del bbreg_feats_vgg
        else:
            bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result[i - 1]
            bbreg_bbox = result_bb[i - 1]

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox


        if success:
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            pos_feats_vgg= forward_samples(args,model, img_list,i,result_bb, pos_examples,i,flag_flase=Flag_padding)#因为是要输出更新网络前的输出

            neg_feats_vgg= forward_samples(args,model, img_list,i,result_bb, neg_examples,i,flag_flase=Flag_padding)
            pos_feats_all_vgg.append(pos_feats_vgg)
            neg_feats_all_vgg.append(neg_feats_vgg)
            del pos_feats_vgg
            del neg_feats_vgg

            while len(pos_feats_all_vgg) > opts['n_frames_long']:
                del pos_feats_all_vgg[0]
            while len(neg_feats_all_vgg) > opts['n_frames_short']:
                del neg_feats_all_vgg[0]
        if (not success):
            nframes = min(opts['n_frames_short'], len(pos_feats_all_vgg))
            pos_data_vgg = torch.stack(pos_feats_all_vgg[-nframes:], 0).view(-1, feat_dim_vgg)
            neg_data_vgg = torch.stack(neg_feats_all_vgg, 0).view(-1, feat_dim_vgg)
            train(model, criterion, update_optimizer, pos_data_vgg,neg_data_vgg,opts['maxiter_update'])
            del pos_data_vgg
            del neg_data_vgg
        # Long term update
        elif (i % opts['long_interval'] == 0) :
            first_slicing_vgg = torch.chunk(first_pos_feats_vgg, 10, dim=0)

            for f_s in range(3):
                    pos_feats_all_vgg.append(first_slicing_vgg[f_s])
            pos_data_vgg = torch.stack(pos_feats_all_vgg, 0).view(-1, feat_dim_vgg)
            neg_data_vgg = torch.stack(neg_feats_all_vgg, 0).view(-1, feat_dim_vgg)
            train(model, criterion, update_optimizer,pos_data_vgg,neg_data_vgg,opts['maxiter_update'])
            for _ in range(3):
                    del pos_feats_all_vgg[-1]
            del pos_data_vgg
            del neg_data_vgg

        spf = time.time() - tic
        spf_total += spf

        # Display
        # if display or savefig:
        #     im.set_data(image)
        #
        #     if gt is not None:
        #         gt_rect.set_xy(gt[i, :2])
        #         gt_rect.set_width(gt[i, 2])
        #         gt_rect.set_height(gt[i, 3])
        #
        #     rect.set_xy(result_bb[i, :2])
        #     rect.set_width(result_bb[i, 2])
        #     rect.set_height(result_bb[i, 3])
        #
        #     if display:
        #         plt.pause(.01)
        #         #plt.figure()
        #         plt.draw()
        #     if savefig:
        #         fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)
        #         if overlap_ratio(gt[i], result_bb[i])[0] == 0:
        #             p0 += 1
        #         elif overlap_ratio(gt[i], result_bb[i])[0] <= 0.3:
        #             p3 += 1
        #         elif overlap_ratio(gt[i], result_bb[i])[0] <= 0.5:
        #             p5 += 1
        #         elif overlap_ratio(gt[i], result_bb[i])[0] <= 0.7:
        #             p7 += 1
        #         elif overlap_ratio(gt[i], result_bb[i])[0] <= 0.9:
        #             p9 += 1
        #         elif overlap_ratio(gt[i], result_bb[i])[0] <= 1.0:
        #             p10 += 1
        #         sum_AC += overlap_ratio(gt[i], result_bb[i])[0]

        if gt is None:
            print("Frame %d/%d, Score %.3f, Time %.3f" % \
                  (i, len(img_list), target_score, spf))
        else:
            ii = ii + 1
            overlap = overlap + overlap_ratio(gt[i], result_bb[i])[0]
            print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                  (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
    plt.close()
    averge_overlap = overlap / ii
    print('the', args.seq, "of average_ovelap is:", averge_overlap)
    fps = len(img_list) / spf_total

    res = open(opts['result'] + '/stat.txt', 'a')
    res.writelines("Seq %s, AC %.8f, Lose %d, P0 %d, P3 %d, P5 %d, P7 %d, P9 %d, P10 %d \n" % \
                   (args.seq, sum_AC / (len(img_list)-1), zero, p0, p3, p5, p7, p9, p10))
    res.close()
    return result, result_bb, fps, averge_overlap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='Singer2', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()#
    assert (args.seq != '' or args.json != '')
    path_otb = opts['dataset_path']
    path_otb_video=os.listdir(path_otb)
    kk=0
    k_overlap=0.0
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
    display =False
    # Run tracker
    result, result_bb, fps, averge_overlap = run_mdnet(args,img_list, init_bbox, gt=gt, savefig_dir=savefig_dir,
                                                       display=display)
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
