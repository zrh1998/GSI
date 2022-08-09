import array
import os
import zipfile
import itertools
import six
from sklearn.linear_model import PassiveAggressiveClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
from maskrcnn_benchmark.modeling.utils import cat


def get_box_info(boxes, need_norm=True, proposal=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch.cat((boxes, center_box), 1)
    if need_norm:
        box_info = box_info / float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info

def get_box_pair_info(box1, box2):
    """
    input: 
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output: 
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:,:4].clone()
    unionbox[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox, need_norm=False)

    # intersection box
    intersextion_box = box1[:,:4].clone()
    intersextion_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    case1 = torch.nonzero(intersextion_box[:, 2].contiguous().view(-1) < intersextion_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch.nonzero(intersextion_box[:, 3].contiguous().view(-1) < intersextion_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersextion_box, need_norm=False)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch.cat((box1, box2, union_info, intersextion_info), 1)

# #求联合区域函数
# def union_location(a_x1, a_y1, a_x2, a_y2, b_x1, b_y1, b_x2, b_y2):
#     union_info=[]
#     union_x1=float()
#     union_y1=float()
#     union_x2=float()
#     union_y2=float()
#     if a_x1<=b_x1:
#         #6 1 5 7
#         if a_x2>=b_x1:
#             #6
#             if a_y2<b_y1:
#                 union_x1=a_x1
#                 union_y1=a_y1
#                 union_x2=a_x2
#                 union_y2=b_y2
#             #1
#             elif a_y2>=b_y1 and a_y2<=b_y2:
#                 union_x1=a_x1
#                 union_y1=a_y1
#                 union_x2=b_x2
#                 union_y2=b_y2
#             #5
#             elif a_y1>=b_y1 and a_y1<=b_y2:
#                 union_x1=a_x1
#                 union_y1=b_y1
#                 union_x2=b_x2
#                 union_y2=a_y2
#             #7
#             elif a_y1>b_y2:
#                 union_x1=a_x1
#                 union_y1=b_y1
#                 union_x2=b_x2
#                 union_y2=b_y2
#         #2 3 4
#         elif a_x2<b_x1:
#             #2
#             if a_y2<b_y1:
#                 union_x1=a_x1
#                 union_y1=a_y1
#                 union_x2=b_x2
#                 union_y2=b_y2
#             #4
#             elif a_y1>b_y2:
#                 union_x1=a_x1
#                 union_y1=b_y1
#                 union_x2=b_x2
#                 union_y2=a_y2
#             #3
#             else:
#                 union_x1=a_x1
#                 union_y1=a_y1
#                 union_x2=b_x2
#                 union_y2=a_y2
#     elif a_x1>b_x1:
#         #8 12
#         if a_x1<=b_x2:
#             #8
#             if a_y2>=b_y1 and a_y2<=b_y2:
#                 union_x1=b_x1
#                 union_y1=a_y1
#                 union_x2=a_x2
#                 union_y2=b_y2
#             #12
#             elif a_y1>=b_y2 and a_y1<=b_y2:
#                 union_x1=b_x1
#                 union_y1=b_y1
#                 union_x2=a_x2
#                 union_y2=b_y2
#         #9 10 11
#         elif a_x1>b_x2:
#             #9
#             if a_y2<b_y1:
#                 union_x1=b_x1
#                 union_y1=a_y1
#                 union_x2=a_x2
#                 union_y2=b_y2
#             #11
#             elif a_y1>b_y2:
#                 union_x1=b_x1
#                 union_y1=b_y1
#                 union_x2=a_x2
#                 union_y2=a_y2
#             #10
#             else:
#                 union_x1=b_x1
#                 union_y1=a_y1
#                 union_x2=a_x2
#                 union_y2=a_y2
#     else:
#         print(str(a_x1)+" "+str(a_y1)+" "+str(a_x2)+" "+str(a_y2)+" "+str(b_x1)+" "+str(b_y1)+" "+str(b_x2)+" "+str(b_y2))      
#     union_info=[union_x1,union_y1,union_x2,union_y2]
#     return union_info
# def union_location(a, b,l):
#     #最后的输出
#     union_info=torch.zeros([l,4])-1.
#     # tem_union=cat([a,b],dim=1)
#     #判别式一共6种
#     union_condition=torch.zeros([l,7]) 
#     union_condition[:,0]=a[:,0]-b[:,0]
#     union_condition[:,1]=a[:,2]-b[:,0]
#     union_condition[:,2]=a[:,3]-b[:,1]
#     union_condition[:,3]=a[:,3]-b[:,3]
#     union_condition[:,4]=a[:,1]-b[:,1]
#     union_condition[:,5]=a[:,1]-b[:,3]
#     union_condition[:,6]=a[:,0]-b[:,2]
#     union_condition_abs = torch.abs(union_condition)
#     union_condition_div = union_condition + union_condition_abs
#     # union_all=torch.cat([union_condition,a,b],dim=1)

#     for m in range(l):
#         #6
#         tem_6=torch.cat([tem_union[m,0].unsqueeze(0),tem_union[m,1].unsqueeze(0),tem_union[m,2].unsqueeze(0),tem_union[m,7].unsqueeze(0)], dim=0)
#         union_info[m]=torch.where(union_all[m,:]<0. and union_all[:,0]<=0. and union_all[:,1]>=0. and union_all[:,2]<0.,tem_6,tem_union_fu)

        # #1
        # tem_1=torch.cat([tem_union[:,0],tem_union[:,1],tem_union[:,6],tem_union[:,7]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]<=0. and union_condition[:,1]>=0. and union_condition[:,2]>=0. and union_condition[3]<=0.,tem_1,tem_union_fu)

        # #5
        # tem_5=torch.cat([tem_union[:,0],tem_union[:,5],tem_union[:,6],tem_union[:,3]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]<=0. and union_condition[:,1]>=0. and union_condition[:,4]>=0. and union_condition[5]<=0.,tem_5,tem_union_fu)

        # #7
        # tem_7=torch.cat([tem_union[:,0],tem_union[:,5],tem_union[:,6],tem_union[:,7]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]<=0. and union_condition[:,1]>=0. and union_condition[:,5]>0.,tem_7,tem_union_fu)

        # #2
        # tem_2=torch.cat([tem_union[:,0],tem_union[:,1],tem_union[:,6],tem_union[:,7]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]<=0. and union_condition[:,1]<0. and union_condition[:,2]<0.,tem_2,tem_union_fu)

        # #4
        # tem_4=torch.cat([tem_union[:,0],tem_union[:,5],tem_union[:,6],tem_union[:,3]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]<=0. and union_condition[:,1]<0. and union_condition[:,5]>0.,tem_4,tem_union_fu)

        # #3
        # tem_3=torch.cat([tem_union[:,0],tem_union[:,1],tem_union[:,6],tem_union[:,3]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]<=0. and union_condition[:,1]<0. and union_condition[:,3]>=0. and union_condition[4]<=0.,tem_3,tem_union_fu)

        # #8
        # tem_8=torch.cat([tem_union[:,4],tem_union[:,1],tem_union[:,2],tem_union[:,7]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]>0. and union_condition[:,2]>=0. and union_condition[:,3]<=0. and union_condition[6]<=0.,tem_8,tem_union_fu)

        # #12
        # tem_12=torch.cat([tem_union[:,4],tem_union[:,5],tem_union[:,2],tem_union[:,7]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]>0. and union_condition[:,4]>=0. and union_condition[:,5]<=0. and union_condition[6]<=0.,tem_12,tem_union_fu)

        # #9
        # tem_9=torch.cat([tem_union[:,4],tem_union[:,1],tem_union[:,2],tem_union[:,7]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]>0. and union_condition[:,2]<0. and union_condition[:,7]>0.,tem_9,tem_union_fu)

        # #11
        # tem_11=torch.cat([tem_union[:,4],tem_union[:,5],tem_union[:,2],tem_union[:,3]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]>0. and union_condition[:,5]>0. and union_condition[:,7]>0.,tem_11,tem_union_fu)

        # #10
        # tem_10=torch.cat([tem_union[:,4],tem_union[:,1],tem_union[:,2],tem_union[:,3]], dim=1)
        # union_info[m]=torch.where(union_info[m,:]<0. and union_condition[:,0]>0. and union_condition[:,3]>=0. and union_condition[:,4]<=0.,tem_10,tem_union_fu)

    #return union_info
# #求重合区域函数
# def coin_location(a_x1, a_y1, a_x2, a_y2, b_x1, b_y1, b_x2, b_y2):
#     coin_info=[]
#     coin_x1=float()
#     coin_y1=float()
#     coin_x2=float()
#     coin_y2=float()
#     if a_x2>=b_x1 and a_x2<=b_x2:
#         if a_y2>=b_y1 and a_y2<=b_y2:
#             coin_x1=b_x1
#             coin_y1=b_y1
#             coin_x2=a_x2
#             coin_y2=a_y2
#         elif a_y1>=b_y1 and a_y1<=b_y2:
#             coin_x1=b_x1
#             coin_y1=a_y1
#             coin_x2=a_x2
#             coin_y2=b_y2
#     elif a_x1>=b_x1 and a_x1<=b_x2:
#         if a_y2>=b_y1 and a_y2<=b_y2:
#             coin_x1=a_x1
#             coin_y1=b_y1
#             coin_x2=b_x2
#             coin_y2=a_y2
#         elif a_y1>=b_y1 and a_y1<=b_y2:
#             coin_x1=a_x1
#             coin_y1=a_y1
#             coin_x2=b_x2
#             coin_y2=b_y2   
#     coin_info=[coin_x1,coin_y1,coin_x2,coin_y2]
#     return coin_info
def box_union(boxten1, boxten2):
    assert boxten1.size() == boxten2.size()
    union_box = torch.cat((
        torch.min(boxten1[:,:2], boxten2[:,:2]),
        torch.max(boxten1[:,2:], boxten2[:,2:])
        ),dim=1)
    return union_box
#             m        n
def box_iou(boxten1, boxten2,n):
    lt_m=boxten1[:,2]-boxten1[:,0]
    rb_m=boxten1[:,3]-boxten1[:,1]
    m=lt_m*rb_m
    Iou_inf=torch.zeros([n,4]).cuda()
    #取max
    Iou_inf[:,0]=torch.where(boxten2[:,0]>boxten1[:,0],boxten2[:,0],boxten1[:,0])
    Iou_inf[:,1]=torch.where(boxten2[:,1]>boxten1[:,1],boxten2[:,1],boxten1[:,1])
    #取min
    Iou_inf[:,2]=torch.where(boxten2[:,2]<boxten1[:,2],boxten2[:,2],boxten1[:,2])
    Iou_inf[:,3]=torch.where(boxten2[:,3]<boxten1[:,3],boxten2[:,3],boxten1[:,3])
    Iou=((Iou_inf[:,2]-Iou_inf[:,0])*(Iou_inf[:,3]-Iou_inf[:,1]))/m
    Iou=Iou.type(torch.cuda.DoubleTensor)
    Iou=torch.where(Iou==1.,0.,Iou)
    return Iou

def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:,:,:,0]*inter[:,:,:,1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:,2]- boxes_flat[:,0]+1.0)*(
        boxes_flat[:,3]- boxes_flat[:,1]+1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union

def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)
        return


def obj_prediction_nms(boxes_per_cls, pred_logits, nms_thresh=0.3):
    """
    boxes_per_cls:               [num_obj, num_cls, 4]
    pred_logits:                 [num_obj, num_category]
    """
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]

    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0), boxes_per_cls.size(0), 
                              boxes_per_cls.size(1)).cpu().numpy() >= nms_thresh

    prob_sampled = F.softmax(pred_logits, 1).cpu().numpy()
    prob_sampled[:, 0] = 0  # set bg to 0

    pred_label = torch.zeros(num_obj, device=pred_logits.device, dtype=torch.int64)

    for i in range(num_obj):
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0 # This way we won't re-sample

    return pred_label 


def block_orthogonal(tensor, split_sizes, gain=1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]