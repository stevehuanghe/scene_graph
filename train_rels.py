"""
Adapted from https://github.com/rowanz/neural-motifs/blob/master/models/train_rels.py

Add gce_loss to losses
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import dill as pkl
import pandas as pd
import time
import os
import gc
from pprint import pprint

from dataloaders.visual_genome import VGDataLoader, VG
from config import ModelConfig, BOX_SCALE, IM_SCALE, GLOVE_VEC_FN
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.poly_grader import get_neg_reward
from lib.entropy_loss import Entropy
from lib.focal_loss import FocalLoss
from lib.utils import get_neg_labels

import ipdb
import crash_on_ipy

conf = ModelConfig()

np.random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)

if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'linknet':
    from lib.rel_model_linknet2 import RelModelLinknet as RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
elif conf.model == 'arpn':
    from lib.rel_model_arpn_v2 import RelModelARPN as RelModel
elif conf.model == 'arpn_df':
    from lib.rel_model_arpn_v3 import RelModelARPN as RelModel
elif conf.model == 'rpn':
    from lib.rel_model_rpn import RelModelARPN as RelModel
else:
    raise ValueError("Invalid model {}".format(conf.model))

focal_loss = FocalLoss(gamma=conf.gamma)

train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision,
                    use_rank=conf.use_rank,
                    use_rdist=(not conf.not_rdist),
                    use_score=conf.use_score
                    )

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n, p in detector.named_parameters(
    ) if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n, p in detector.named_parameters(
    ) if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0},
              {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(
            params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler


ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.roi_fmap[1][0].weight.data.copy_(
        ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap[1][3].weight.data.copy_(
        ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap[1][0].bias.data.copy_(
        ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap[1][3].bias.data.copy_(
        ckpt['state_dict']['roi_fmap.3.bias'])

    detector.roi_fmap_obj[0].weight.data.copy_(
        ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap_obj[3].weight.data.copy_(
        ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(
        ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(
        ckpt['state_dict']['roi_fmap.3.bias'])

detector.cuda()

pause_epoch = [-1]  # do not save predictions on training data

lossfile = os.path.join(conf.save_dir, 'loss.txt')


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    all_items = []
    for b, batch in enumerate(train_loader):
        trl, item = train_batch(batch, verbose=b %
                              (conf.print_interval*10) == 0)
        tr.append(trl)  # b == 0))
        all_items.append(item)
        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("e {:2d}, b {:5d}/{:5d}, {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)

            with open(lossfile, "a+") as fout:
                pprint("e {:2d}, b {:5d}/{:5d}, {:.3f}s/batch, {:.1f}m/epoch".format(
                    epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60), fout)
                pprint(mn, fout)
                pprint('-----------', fout)

            start = time.time()
    if epoch_num in pause_epoch:
        savepath = os.path.join(conf.save_dir, "train_output_e" + str(epoch_num) + ".pkl")
        with open(savepath, "wb") as fout:
            pkl.dump(all_items, fout)
        print("train output saved at: " + savepath)
        del all_items
        gc.collect()
    return pd.concat(tr, axis=1)


def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]
    losses = {}
    if conf.model != 'stanford':
        losses['gce_loss'] = torch.nn.BCELoss()(
            result.gce_obj_dists, result.gce_obj_labels)

    losses['class_loss'] = F.cross_entropy(
        result.rm_obj_dists, result.rm_obj_labels)

    gt_rels = result.rel_labels[:, -1].contiguous().long()

    if conf.use_rank:
        ranks = result.ranks.view(-1)
        mask = (gt_rels.view(-1) > 0).float()
        inv_mask = 1.0 - mask
        if conf.rloss == 'bce':
            ranks = F.sigmoid(ranks)
            losses['rel_rank'] =  conf.lambda2 * F.binary_cross_entropy(ranks, mask)
        elif conf.rloss == 'mse':
            losses['rel_rank'] =  conf.lambda2 * F.mse_loss(ranks, mask)
        elif conf.rloss == 'margin':
            n_pos = mask.sum()
            n_neg = inv_mask.sum()
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            pos_margin = (ranks * inv_mask).max()
            neg_margin = (ranks * mask).min()
            loss_rank = w_pos * F.relu(pos_margin - ranks * mask + conf.m2).mean() \
                + w_neg * F.relu(ranks * inv_mask - neg_margin + conf.m2).mean()
            losses['rel_rank'] = conf.lambda2 * loss_rank
        else:
            raise ValueError(f'loss undefined: {conf.rloss}')

    if conf.loss == 'cce':
        neg_rels = get_neg_labels(result.rel_dists, gt_rels).long()

        rel_loss = F.cross_entropy(result.rel_dists, gt_rels, reduce=False) - \
                    F.cross_entropy(result.rel_dists, neg_rels, reduce=False) + conf.m1 
        rel_loss = conf.lambda1 * F.relu(rel_loss).mean()
    elif conf.loss == 'ce':
        rel_loss = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    elif conf.loss == 'focal':
        rel_loss = conf.lambda1 * focal_loss(result.rel_dists, gt_rels)
    else:
        raise ValueError(f'loss undefined: {conf.loss}')

    losses['rel_loss'] = rel_loss

    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()

    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)

    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.item() for x, y in losses.items()})
    # return res, (result.rel_dists.data.cpu().numpy(), relations.view(-1).data.cpu().numpy())
    if 'rpn' in conf.model:
        return res, (result.rel_dists.data.cpu().numpy(), result.rel_labels.data.cpu().numpy(),
                     result.rm_box_priors.data.cpu().numpy(), result.rel_inds.data.cpu().numpy(), result.pred_rois.data.cpu().numpy())
    return res, (result.rel_dists.data.cpu().numpy(), result.rel_labels.data.cpu().numpy())


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats()
    res = {}
    for k, v in evaluator[conf.mode].result_dict[conf.mode + '_recall'].items():
        res['all_' + str(k)] = np.mean(v)

    mc_recall = {}
    for k, v in evaluator[conf.mode].result_dict[conf.mode + '_recall_per'].items():
        mc_recall[k] = 0.0
        cnt = 0
        keys = sorted(list(v.keys()))
        for rid in keys:
            data = v[rid]
            if rid > 0:
                mc_recall[k] += np.mean(data)
                cnt += 1
            #res[f'{k}-{rid}'] = np.mean(data)
        mc_recall[k] /= cnt 
        res[f'mcr_{k}'] = mc_recall[k]
    #print(f"mean class recall: {mc_recall/cnt: .5}")
    return mc_recall[100], res
    #return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100]), res


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(
            objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


logfile = os.path.join(conf.save_dir, 'log.txt')
with open(logfile, "w+") as fout:
    pprint(conf.args, fout)

best_epoch = -1
best_mcr = 0
best_res = None
print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(
        epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            # {k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            'state_dict': detector.state_dict(),
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    mcr, res = val_epoch()
    scheduler.step(mcr)

    if mcr > best_mcr:
        best_epoch = epoch
        best_mcr = mcr
        best_res = res

    with open(logfile, "a+") as fout:
        pprint("----------", fout)
        pprint(str(epoch), fout)
        pprint(res, fout)

    gc.collect()
    if conf.early_stop:
        if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
            print("exiting training early", flush=True)
            break

with open(logfile, "a+") as fout:
    pprint("----------", fout)
    pprint(f"best epoch: {best_epoch}", fout)
    pprint(best_res, fout)

pprint(conf.args)


