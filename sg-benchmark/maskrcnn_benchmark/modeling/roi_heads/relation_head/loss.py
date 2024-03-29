# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
import math

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

from nbdt.utils import maybe_install_wordnet
from nbdt import loss as nbdtloss

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        cfg,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()
        self.cfg = cfg

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class ReweightLoss(nn.Module):
    def __init__(self, num_class, samples_per_cls):
        super(ReweightLoss, self).__init__()
        self.num_class = num_class
        weights = 1.0 / np.array(samples_per_cls)
        weights = weights / np.sum(weights) * num_class
        self.weights = torch.tensor(weights).float()
    def forward(self, logits, labels):
        weights = self.weights.to(logits.device)
        rw_losses = F.cross_entropy(input=logits, target=labels, weight=weights)
        return rw_losses

class ClassBalancedLoss(nn.Module):
    def __init__(self, num_class, factor, samples_per_cls, loss_type="softmax", beta=0.999, gamma=0):
        super(ClassBalancedLoss, self).__init__()
        self.num_class = num_class
        self.factor = factor
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_samples = samples_per_cls
        
        #原来的Weights
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_class
        self.weights = torch.tensor(weights).float()
        print("EQL loss activated!")
        self.lambda_ = 0.00083318
        self.q = 3.0
        self.eps = 1e-8
        print("SEESAW loss activated!")
        self.__calc_freq_info()

    def __calc_freq_info(self):
        total_nums = np.sum(self.num_samples)
        self.freq_info = self.num_samples / total_nums
        print(self.freq_info)
        print(np.sum(self.freq_info[1:]))
        print(np.average(self.freq_info[1:]))

    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                               torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
    
    #CB_loss weight改变，加入equalized loss weight
    def exclude_func(self):
        # instance-level weight
        weight = self.pred_class_logits.new_ones(self.n_c)
        weight[0] = 0.
        # weight = (self.num_samples != bg_ind).float()
        # weight[0] = 0.
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def CB_losss(self, labels, logits, loss_type, gamma):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        self.n_i, self.n_c = logits.size()
        self.pred_class_logits = logits

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]
        
        weights_ = self.weights.to(logits.device)

        labels_one_hot = F.one_hot(labels, self.num_class).float()

        weights = weights_.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_class)

        # target = labels.view(-1)
        # logpt = F.log_softmax(logits)
        # logpt = logpt.index_select(-1, target).diag()
        # logpt = logpt.view(-1)
        # pt = logpt.exp()
        # print(pt)
        # print(pt.shape)
        
        #Seesaw Loss
        
        # if self.q > 0:
        #     seesaw_weights = logits.new_ones(labels_one_hot.size())
        #     scores = F.softmax(logits.detach(), dim=1)
        #     self_scores = scores[
        #         torch.arange(0, len(scores)).to(scores.device).long(),
        #         labels.long()]
        #     score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
        #     index = (score_matrix > 1.0).float()
        #     compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
        #     seesaw_weights = seesaw_weights * compensation_factor

        # target = expand_label(self.pred_class_logits, labels)

        # eql_w = 1 - self.exclude_func()  * (1 - target)
        # eql_w = 1 - self.exclude_func() * self.threshold_func() * (1 - target)

        # logits = (logits - torch.min(logits, dim=1, keepdim=True)) / (torch.max(logits, dim=1, keepdim=True) - torch.min(logits, dim=1, keepdim=True))  #正则化
        #将尾部谓词放在list里
        Tail_list = [6, 7, 46, 11, 33, 16, 9, 25, 47, 19, 35, 24, 5, 14, 13, 10, 44, 4, 12, 36, 32, 42, 26, 28, 45, 2, 17, 3, 18, 34,37, 27, 39, 15]
        predict=[]
        if loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights_, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights_)
        elif loss_type == "softmax":
            # pred = logits.softmax(dim = 1)
            # cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)  
                     
            cb_loss = F.cross_entropy(input = logits, target = labels, weight = weights_)
            #注释
            l=int(labels.size()[0])
            #predict=torch.zeros([l],dtype=torch.int)
            for m in range(l):
                predict=int(torch.max(logits[m],0).indices)
                if predict != labels[m] and predict in Tail_list and labels[m] in Tail_list:
                    return cb_loss*math.pow((logits[m][predict]/logits[m][labels[m]]),self.q)


            # cb_loss = F.cross_entropy(input = logits, target = labels)#, weight = eql_w)
        
        return cb_loss
        # return torch.sum(cb_loss * eql_w * seesaw_weights) / self.n_i
        #return cb_loss

    def forward(self, logits, labels):
        
        
        cb_losses = self.CB_losss(labels, logits, self.loss_type, self.gamma)
        
        
        # 
        return cb_losses * self.factor

def NBDTLoss(cfg, criterion):
    maybe_install_wordnet()
    # print(cfg.MODEL.ROI_RELATION_HEAD.LOSS.NBDT.TYPE)
    class_criterion = getattr(nbdtloss, cfg.MODEL.ROI_RELATION_HEAD.LOSS.NBDT.TYPE)
    NBDTloss = class_criterion(dataset='VG150', criterion=criterion, sample_nums=cfg.MODEL.ROI_RELATION_HEAD.REL_SAMPLES[1:], path_graph=cfg.MODEL.ROI_RELATION_HEAD.LOSS.NBDT.PATH_GRAPH,
                                path_wnids=cfg.MODEL.ROI_RELATION_HEAD.LOSS.NBDT.PATH_WNIDS, tree_supervision_weight=cfg.MODEL.ROI_RELATION_HEAD.LOSS.NBDT.FACTOR)
    return NBDTloss


def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg,
    )

    return loss_evaluator
