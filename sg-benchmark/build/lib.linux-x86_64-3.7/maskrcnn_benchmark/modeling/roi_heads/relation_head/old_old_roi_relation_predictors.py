# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from audioop import bias
from regex import B
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
import numpy as np

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import Transformer_ObjContext, Transformer_EdgeContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from .loss import ClassBalancedLoss, NBDTLoss, FocalLoss, ReweightLoss
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import obj_edge_vectors
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .utils_relation import union_location as calc_union_location
from .utils_relation import coin_location as calc_coin_location

@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()

        # self.i = 0

        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.cat_pair = config.MODEL.ROI_RELATION_HEAD.CAT_PAIR
        self.use_ctx_gate = config.MODEL.ROI_RELATION_HEAD.USE_CTX_GATE

        self.use_focal_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_FOCAL_LOSS
        self.use_reweight_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_RW_LOSS
        self.use_class_balanced_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS
        self.use_nbdt_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if self.use_vision:
            if self.use_ctx_gate:
                self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
                layer_init(self.post_cat, xavier=True)
            if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
                self.union_single_not_match = True
                self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
                layer_init(self.up_dim, xavier=True)
            else:
                self.union_single_not_match = False
            edge_in_channels = self.pooling_dim
        else:
            edge_in_channels = self.hidden_dim * 2
        self.dropout = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.cfg = config

        #新加入的word embedding部分，将pair word embedding 进行处理
        self.embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM)

        # module construct
        self.obj_context_layer = Transformer_ObjContext(config, obj_classes, in_channels, self.embed_vecs)
        self.edge_context_layer = Transformer_EdgeContext(config, edge_in_channels)

        if self.cat_pair:
            self.lin_rel = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            layer_init(self.lin_rel, xavier=True)

        self.rel_compress = nn.Linear(self.hidden_dim, self.num_rel_cls, bias=config.MODEL.ROI_RELATION_HEAD.LINEAR_USE_BIAS)
        if config.MODEL.ROI_RELATION_HEAD.LINEAR_USE_BIAS:
            layer_init(self.rel_compress, xavier=True)

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        if self.use_focal_loss:
            self.focal_loss = FocalLoss(config.MODEL.ROI_RELATION_HEAD.LOSS.FOCAL.GAMMA,
                                        config.MODEL.ROI_RELATION_HEAD.LOSS.FOCAL.ALPHA)

        if self.use_reweight_loss:
            self.reweight_loss = ReweightLoss(self.num_rel_cls,
                                              config.MODEL.ROI_RELATION_HEAD.REL_SAMPLES)

        if self.use_class_balanced_loss:
            self.class_balanced_loss = ClassBalancedLoss(self.num_rel_cls,
                                                         config.MODEL.ROI_RELATION_HEAD.LOSS.CB_FACTOR,
                                                         config.MODEL.ROI_RELATION_HEAD.REL_SAMPLES,
                                                         loss_type=config.MODEL.ROI_RELATION_HEAD.LOSS.CB_TYPE)

        if self.use_nbdt_loss:
            criterion = nn.CrossEntropyLoss()
            # print("11111111111111111111111")
            self.nbdt_loss = NBDTLoss(config, criterion)
        
        self.conv_block= nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5,stride=1,padding=2,bias=True),   #200
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), #100
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1,bias=True), #100
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), #50
            nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1,bias=True), #50
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), #25
            nn.Conv2d(256, 512, kernel_size=3,stride=1,padding=1,bias=True), #25
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    #1
            nn.Conv2d(512, 384, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 4096, 1, 1, 0, bias=True)
        )

        self.word_feature = nn.Sequential(*[
            nn.Linear(8192, 512), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

    def feature_generate(self, roi_features, proposals, num_objs, rel_pair_idxs, union_features, num_rels, logger):
        obj_dists, obj_preds, obj_feats, obj_embed_out, pos_embed = self.obj_context_layer(roi_features, proposals,
                                                                                           logger)

        # post decode
        edge_rep = self.post_emb(obj_feats)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        pair_obj_embeds = []
        pair_pos_embeds = []
        
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            # print("!!!!!!!!!!!!!!!!!!!!!!")
            # print(pair_idx)
            # print("###################")
            #获取object和subject对应的word embedding:
            # for i in pair_idx:
            #     #object的word embedding
            #     obj_word_embedding=self.embed_vecs[i[0]+1]
            #     #subject的word embedding
            #     sub_word_embedding=self.embed_vecs[i[1]+1]
            #     pair_word_embeds.append(union_word_embeddings(obj_word_embedding,sub_word_embedding))
            
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_embeds.append(torch.cat((obj_embed_out[pair_idx[:, 0]], obj_embed_out[pair_idx[:, 1]]), dim=-1))
            pair_pos_embeds.append(torch.cat((pos_embed[pair_idx[:, 0]], pos_embed[pair_idx[:, 1]]), dim=-1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        pair_obj_embed = cat(pair_obj_embeds, dim=0)
        pair_pos_embed = cat(pair_pos_embeds, dim=0)
        #对位置信息进行处理
        #use union box and mask convolution
        if self.use_vision:
            if self.use_ctx_gate:
                ctx_gate = self.post_cat(prod_rep)
                if self.union_single_not_match:
                    visual_rep = ctx_gate * self.up_dim(union_features)
                else:
                    visual_rep = ctx_gate * union_features
            else:
                if self.union_single_not_match:
                    visual_rep = self.up_dim(union_features)
                else:
                    visual_rep = union_features
        else:
            visual_rep = prod_rep

        edge_feats = self.edge_context_layer(visual_rep, obj_feats, pair_obj_embed, pair_pos_embed, num_rels, num_objs,
                                             logger)

        if self.cat_pair:
            rel_pairs = prod_rep.view(prod_rep.size(0), 2, self.hidden_dim)
            rel_sub = rel_pairs[:, 0].contiguous().view(-1, self.hidden_dim)
            rel_obj = rel_pairs[:, 1].contiguous().view(-1, self.hidden_dim)
            # print("!!!!!!!!!!!!!!!!!!!!!!")
            # print(rel_sub.size())
            # print(rel_obj.size())
            # print("###################")
            rel_feats = torch.cat((rel_sub, edge_feats, rel_obj), dim=-1)
            # print("!!!!!!!!!!!!!!!!!!!!!!")
            # print(rel_feats.size())
            # print("###################")
            rel_feats = self.lin_rel(rel_feats)
        else:
            rel_feats = edge_feats

        return rel_feats, pair_pred, obj_dists


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        
            
        device=roi_features.device
        #rel_feats, pair_pred, obj_dists = self.feature_generate(roi_features, proposals, num_objs, rel_pair_idxs, union_features, num_rels, logger)
        rel_feats, pair_pred, obj_dists= self.feature_generate(roi_features, proposals, num_objs, rel_pair_idxs, union_features, num_rels, logger)
        #print("!!!!!!!!!!!!!!!!!!!!!!")
        #batch=4时，4张图片pair总和为对应的union_features的列数
        boxes=[]
        #获取出四张图片的boxes坐标
        for proposal in proposals:
            boxes.append(proposal.bbox.data.cpu().numpy())
        #记录是第几张图片
        flag_picture=0
        batch_union_location=[]
        union_location=[]
        IOU=float()
        obj_inf=[]
        sub_inf=[]
        union_inf=[]
        union_union=[]
        union_coin=[]
        IOU_list=[]
        #rel_pair_idxs为四张图片
        # for i in rel_pair_idxs:
        #     # a=[]
        #     #i为每张图片，j为 单个关系
        #     for j in i:
        #         pair_word=j.data.cpu().numpy()
        #         obj_id=pair_word[0]
        #         sub_id=pair_word[1]
        #         obj_proposal=boxes[flag_picture][obj_id]
        #         sub_proposal=boxes[flag_picture][sub_id]
        #         #获取proposals里面的位置信息
        #         #获取每张图片中单个物体的坐标
        #         obj_x1=obj_proposal[0]
        #         obj_y1=obj_proposal[1]
        #         obj_x2=obj_proposal[2]
        #         obj_y2=obj_proposal[3]
        #         sub_x1=sub_proposal[0]
        #         sub_y1=sub_proposal[1]
        #         sub_x2=sub_proposal[2]
        #         sub_y2=sub_proposal[3]
        #         #先获取对象对中联合区域面积组成list，再获取与对应物体相关的联合区域并与其本身算重合面积。通过IOU计算权重
        #         union_inf=calc_union_location(obj_x1,obj_y1,obj_x2,obj_y2,sub_x1,sub_y1,sub_x2,sub_y2)
        #         obj_inf.append([obj_id,obj_x1,obj_y1,obj_x2,obj_y2])
        #         sub_inf.append([sub_id,sub_x1,sub_y1,sub_x2,sub_y2])
        #         union_location.append([obj_inf,sub_inf,union_inf])
        #         #boxes 信息[左下角x,左下角y,右上角x,右上角y]
        #         obj_inf=[]
        #         sub_inf=[]
        #         union_inf=[]
        #     #针对一张图片
        #     # l = len(union_location)
        #     # # for m in union_location:
        #     # #     for n in union_location:
        #     # for i in range(l):
        #     #     for j in range(i, l):
        #     #         #获取与对应物体相关的联合区域并与其本身算重合面积。通过IOU计算权重
        #     #         # obj=obj和obj=sub
        #     #         # m[0][0]是object
        #     #         # m[1][0]是subject
        #     #         #m[2]是union
        #     #         # print(m[0][0])
        #     #         # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #     #         m = union_location[i]
        #     #         n = union_location[j]
        #     #         if (m[0][0][0]==n[0][0][0] and (not m[0][0][0] == n[1][0][0])) or (m[0][0][0]==n[1][0][0] and (not m[0][0][0] == n[0][0][0])):
        #     #             union_union=calc_coin_location(m[2][0],m[2][1],m[2][2],m[2][3],n[2][0],n[2][1],n[2][2],n[2][3])
        #     #             union_coin=calc_coin_location(union_union[0],union_union[1],union_union[2],union_union[3],m[0][0][1],m[0][0][2],m[0][0][3],m[0][0][4])
        #     #             if sum(union_coin)>0.:
        #     #                 if sum(union_union)>0.:
        #     #                     IOU=((union_coin[2]-union_coin[0])*(union_coin[3]-union_coin[1])/(m[0][0][3]-m[0][0][1])*(m[0][0][4]-m[0][0][2]))
        #     #                     print(str((union_coin[2]-union_coin[0])*(union_coin[3]-union_coin[1])))
        #     #                     print(str((m[0][0][3]-m[0][0][1])*(m[0][0][4]-m[0][0][2])))
        #     #                     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #     #             if IOU>0.:
        #     #                 IOU_list.append([obj_id,sub_id,IOU])
        #     #             continue
        #     #         #sub=obj
        #     #         elif (m[1][0][0]==n[0][0][0] and (not m[1][0][0] == n[1][0][0])) or (m[1][0][0]==n[1][0][0] and (not m[1][0][0] == n[0][0][0])):
        #     #             union_union=calc_coin_location(m[2][0],m[2][1],m[2][2],m[2][3],n[2][0],n[2][1],n[2][2],n[2][3])
        #     #             union_coin=calc_coin_location(union_union[0],union_union[1],union_union[2],union_union[3],m[1][0][1],m[1][0][2],m[1][0][3],m[1][0][4])
        #     #             if sum(union_coin)>0.:
        #     #                 if sum(union_coin)>0.:
        #     #                     IOU=((union_coin[2]-union_coin[0])*(union_coin[3]-union_coin[1])/(m[1][0][3]-m[1][0][1])*(m[1][0][4]-m[1][0][2]))
        #     #                     print(str((union_coin[2]-union_coin[0])*(union_coin[3]-union_coin[1])))
        #     #                     print(str((m[1][0][3]-m[1][0][1])*(m[1][0][4]-m[1][0][2])))
        #     #                     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #     #             if IOU>0.:    
        #     #                 IOU_list.append([obj_id,sub_id,IOU])
        #     #             continue
        #     # print(IOU_list)
        #     #以上想法目前不采用
            
        #     # batch_union_location.append(union_location)

        #     #计算两个重合区域的IOU 只有当IOU比值大于0.5才进行下一步操作
        #     l = len(union_location)
        #     for m in range(l):
        #         for n in range(m+1, l):
        #             if sum(union_location[m][2])>0. and sum(union_location[n][2])>0.:
        #                 union_union=calc_coin_location(union_location[m][2][0],union_location[m][2][1],union_location[m][2][2],union_location[m][2][3],union_location[n][2][0],union_location[n][2][1],union_location[n][2][2],union_location[n][2][3])
                    
        #                 a=(union_union[2]-union_union[0])*(union_union[3]-union_union[1])
        #                 b=(union_location[m][2][2]-union_location[m][2][0])*(union_location[m][2][3]-union_location[m][2][1])
        #                 IOU_score=a/b
        #                 if IOU_score>0.8 and IOU_score<1.0:
        #                     IOU_list.append([flag_picture,IOU_score,union_location[m][0][0][0],union_location[m][1][0][0],union_location[n][0][0][0],union_location[n][1][0][0]])                            
        #     l_score = len(IOU_list)
        #     for m in IOU_list:
        #         for n in IOU_list:
        #             if m[4]==n[5] and m[5]==n[4]:
        #                 IOU_list.remove(n)
                    
        #     flag_picture=flag_picture+1
        # print(IOU_list)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # IOU_list=[]
        # boxes=[]
        # batch_union_location=[]
        # flag_picture=0
        


        pre=0
        #object和subject对应的word embedding:
        #pair_word_embeds为每张图片的pairs的word embeddings
        #pair_union_feature为每张图片对应的pair的union features
        #union_feature为四张图片（batch=4）的对应的pair的union features
        pair_word_embed = []
        pair_word_embeds = []
        pair_union_feature=[]
        for i in rel_pair_idxs:
            #每张图片
            for j in i:
                #每张图片里面的每个pair
                pair_word=j.data.cpu().numpy()
                obj_id=pair_word[0]
                sub_id=pair_word[1]
                obj_word_embedding=self.embed_vecs[obj_id+1]
                sub_word_embedding=self.embed_vecs[sub_id+1]
                obj_word_embedding = obj_word_embedding.unsqueeze(-1)
                sub_word_embedding = sub_word_embedding.unsqueeze(0)
                #矩阵乘
                pair_word_embed=torch.matmul(obj_word_embedding,sub_word_embedding).unsqueeze(0).unsqueeze(0)
                # print(pair_word_embed.size())
                pair_word_embed = self.conv_block(pair_word_embed.cuda()).detach().view(4096).cpu().numpy()
                #pair_word_embed=self.relu(self.conv1(pair_word_embed))
                #将每个pair按顺序加入到对应图像的word embeddings中
                pair_word_embeds.append(list(pair_word_embed))
            # pair_union_feature=union_features[pre,pre+len(i)]
            # pre=pre+len(i)
        # print(len(pair_word_embeds))
        pair_word_embeds=torch.tensor(pair_word_embeds).to(device)
        # print(pair_word_embeds.size())
        # print(union_features.size())
        # print(torch.cat([pair_word_embeds, union_features], dim=1).size())
        union_features=self.word_feature(torch.cat([pair_word_embeds, union_features], dim=1))

        rel_dists = self.rel_compress(rel_feats)
        
        # print("--------------------")
        # print(rel_dists)
        # print(rel_dists.shape)

        # use frequence bias
        if self.use_bias:
            freq_dist = self.freq_bias.index_with_labels(pair_pred)
            rel_dists = rel_dists + freq_dist

        add_losses = {}
        if self.training and self.use_focal_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            focal_loss = self.focal_loss(rel_dists, labels)
            add_losses.update(dict(loss_focal=focal_loss))
        if self.training and self.use_reweight_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            reweight_loss = self.reweight_loss(rel_dists, labels)
            add_losses.update(dict(loss_reweight=reweight_loss))

        if self.training and self.use_class_balanced_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            class_balanced_loss = self.class_balanced_loss(rel_dists, labels)
            add_losses.update(dict(loss_class_balanced=class_balanced_loss))

        if self.training and self.use_nbdt_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            nbdt_loss = self.nbdt_loss(rel_dists, labels, self.cfg.MODEL.ROI_RELATION_HEAD.LOSS.NBDT.MODE)
            add_losses.update(dict(nbdt_loss=nbdt_loss))
        
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # if self.training:
        #     torch.save(rel_dists, "./rel_dists/step_" + str(self.i) + ".pth")
        #     self.i += 1

        # print(add_losses)

        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.use_class_balanced_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS
        self.use_nbdt_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        if self.use_class_balanced_loss:
            self.class_balanced_loss = ClassBalancedLoss(self.num_rel_cls,
                                                         config.MODEL.ROI_RELATION_HEAD.LOSS.CB_FACTOR,
                                                         config.MODEL.ROI_RELATION_HEAD.REL_SAMPLES,
                                                         loss_type=config.MODEL.ROI_RELATION_HEAD.LOSS.CB_TYPE)

        if self.use_nbdt_loss:
            criterion = nn.CrossEntropyLoss()
            self.nbdt_loss = NBDTLoss(config, criterion)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())


        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training and self.use_class_balanced_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            class_balanced_loss = self.class_balanced_loss(rel_dists, labels)
            add_losses.update(dict(loss_class_balanced=class_balanced_loss))

        if self.training and self.use_nbdt_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            nbdt_loss = self.nbdt_loss(rel_dists, labels)
            add_losses.update(dict(nbdt_loss=nbdt_loss))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        self.use_class_balanced_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS
        self.use_nbdt_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

        if self.use_class_balanced_loss:
            self.class_balanced_loss = ClassBalancedLoss(self.num_rel_cls,
                                                         config.MODEL.ROI_RELATION_HEAD.LOSS.CB_FACTOR,
                                                         config.MODEL.ROI_RELATION_HEAD.REL_SAMPLES,
                                                         loss_type=config.MODEL.ROI_RELATION_HEAD.LOSS.CB_TYPE)

        if self.use_nbdt_loss:
            criterion = nn.CrossEntropyLoss()
            self.nbdt_loss = NBDTLoss(config, criterion)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists


        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training and self.use_class_balanced_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            class_balanced_loss = self.class_balanced_loss(rel_dists, labels)
            add_losses.update(dict(loss_class_balanced=class_balanced_loss))

        if self.training and self.use_nbdt_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            nbdt_loss = self.nbdt_loss(rel_dists, labels)
            add_losses.update(dict(nbdt_loss=nbdt_loss))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VTransePredictor")
class VTransePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VTransePredictor, self).__init__()
        self.cfg = config
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.use_class_balanced_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS
        self.use_nbdt_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.hidden_dim, self.pooling_dim),
                                       nn.ReLU(inplace=True)
                                       ])
        layer_init(self.spt_emb[0], xavier=True)
        layer_init(self.spt_emb[2], xavier=True)

        if self.use_class_balanced_loss:
            self.class_balanced_loss = ClassBalancedLoss(self.num_rel_cls,
                                                         config.MODEL.ROI_RELATION_HEAD.LOSS.CB_FACTOR,
                                                         config.MODEL.ROI_RELATION_HEAD.REL_SAMPLES,
                                                         loss_type=config.MODEL.ROI_RELATION_HEAD.LOSS.CB_TYPE)

        if self.use_nbdt_loss:
            criterion = nn.CrossEntropyLoss()
            self.nbdt_loss = NBDTLoss(config, criterion)

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger,
                              ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)

        post_ctx_rep = ctx_rep

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
            add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()),
                                                          rel_labels)
            if self.use_class_balanced_loss:
                assert rel_dists.shape[0] == rel_labels.shape[0]
                class_balanced_loss = self.class_balanced_loss(rel_dists, rel_labels)
                add_losses.update(dict(loss_class_balanced=class_balanced_loss))

            if self.use_nbdt_loss:
                assert rel_dists.shape[0] == rel_labels.shape[0]
                nbdt_loss = self.nbdt_loss(rel_dists, rel_labels)
                add_losses.update(dict(nbdt_loss=nbdt_loss))


        return obj_dist_list, rel_dist_list, add_losses

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        union_dists = vis_dists + ctx_dists + frq_dists

        return union_dists

@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        self.use_class_balanced_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS
        self.use_nbdt_loss = config.MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        if self.use_class_balanced_loss:
            self.class_balanced_loss = ClassBalancedLoss(self.num_rel_cls,
                                                         config.MODEL.ROI_RELATION_HEAD.LOSS.CB_FACTOR,
                                                         config.MODEL.ROI_RELATION_HEAD.REL_SAMPLES,
                                                         loss_type=config.MODEL.ROI_RELATION_HEAD.LOSS.CB_TYPE)

        if self.use_nbdt_loss:
            criterion = nn.CrossEntropyLoss()
            self.nbdt_loss = NBDTLoss(config, criterion)

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss

        if self.training and self.use_class_balanced_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            class_balanced_loss = self.class_balanced_loss(rel_dists, labels)
            add_losses.update(dict(loss_class_balanced=class_balanced_loss))

        if self.training and self.use_nbdt_loss:
            labels = cat(rel_labels, dim=0)
            assert rel_dists.shape[0] == labels.shape[0]
            nbdt_loss = self.nbdt_loss(rel_dists, labels)
            add_losses.update(dict(nbdt_loss=nbdt_loss))

        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
