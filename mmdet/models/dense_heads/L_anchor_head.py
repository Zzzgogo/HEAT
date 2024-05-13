import pdb
import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmdet.utils.functions import *

@HEADS.register_module()
class L_AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(type='AnchorGenerator', scales=[8, 16, 32], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=(.0, .0, .0, .0),target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(type='CrossEntropyLoss', last_activation='sigmoid', loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layers='Conv2d', std=0.01)):
        super(L_AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.last_activation = loss_cls.get('last_activation')
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'EDL_Loss', 'EDL_Loss_2','EDL_Loss_3','EDL_Loss_BCE', 'FocalLoss', 'GHMC', 'QualityFocalLoss',
            'EDL_FocalLoss', 'EDL_BetaFocalLoss', 'EDL_FocalLoss_Dummy', 'EDL_Softmax_FocalLoss',
            'EDL_Softmax_FocalLoss_Dummy', 'EDL_Softmax_SL_FocalLoss', 'SSL_EDL_Softmax_FocalLoss',
        ]
        if self.last_activation == 'sigmoid':
            self.cls_out_channels = num_classes
        elif self.last_activation == 'relu':
            self.cls_out_channels = num_classes
        elif self.last_activation == 'softmax':
            self.cls_out_channels = num_classes + 1
        elif self.last_activation == 'EDL_BG':
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def forward_single(self, x):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'): #针对(576,1028)的输入图片，featmap_sizes：[72,128],[36,64],[18,32],[9,16],[5,8]
        """Get anchors according to feature map sizes.              
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors
        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)  #batch大小

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        #将每个特征图上每个grid的9个anchor映射到原图中
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device) #grid_anchors--->anchor_generator.py
        anchor_list = [multi_level_anchors for _ in range(num_imgs)] #为每张图片生成anchor列表

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device) #valid_flags-->anchor_generator.py，检验生成的anchor是否有效
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta,
                            label_channels=1, unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border) #允许anchor超出图片边界的范围(这里没有限制)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels) #-->max_iou_assigner.py ，#划分bbox的正负样本，并为正样本匹配gtbox与标签
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes) #--->sampling_result.py，对assign_result的分配结果进一步做处理，直接得到正负样本的索引号与坐标，并且得到正样本匹配到的gtbox的坐标以及标签

        num_valid_anchors = anchors.shape[0] #所有有效框的数量（总anchor数）
        bbox_targets = torch.zeros_like(anchors) #(..,4)
        bbox_weights = torch.zeros_like(anchors) #(..,4)
        labels = anchors.new_full((num_valid_anchors, ), self.num_classes, dtype=torch.long) #全为14
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float) #全为0

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes) #encode-->delta_xywh_bbox_coder.py，计算用于将anchor转换为gtbox框所需的回归变换参数
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds] #为正样本匹配标签，其他的标签都为14

            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0 #正样本的位置为1，其他为0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0 #再将负样本的位置也置为1

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)

    def get_D_targets(self, cls_scores, isSup):
        rst = []
        for cls_score in cls_scores:
            B,_,H,W = cls_score.shape
            if isSup:
                rst.append(cls_score.new_full((B, self.num_anchors, H, W), fill_value=1))
            else:
                rst.append(cls_score.new_full((B, self.num_anchors, H, W), fill_value=0))
        return rst

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None,
                    gt_labels_list=None, label_channels=1, unmap_outputs=True, return_sampling_results=False):

        num_imgs = len(img_metas) #一个batch中的图片数量
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels 每个特征图上的anchor总数
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i])) #将一张图片里所有anchor拼接在一起（5个特征层的anchor拼接在一起）
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i])) #将一张图片里所有valid_flag拼接在一起（。。。）

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
                 #multi_apply:对多输入采用某个函数，将结果按不同类型打包起来(返回result，得到一个batch中所有图片的正负样本以及匹配到的gtbox和标签等信息)
        results = multi_apply(self._get_targets_single, concat_anchor_list, concat_valid_flag_list, gt_bboxes_list,
            gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7] #一个batch中图片的anchor所匹配到的标签（未匹配到的为14），标签权重（正负样本为1，其他为0），正样本anchor的回归标签，
                                                                            #位置权重（正样本为1，其他为0），正样本索引，负样本索引，sampling_results_list的所有信息（包括一个batch中所有图片的）
        rest_results = list(results[7:])  # user-added return values 这里没有用到
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list]) #一个batch里所有图片的正样本总数
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list]) #一个batch里所有图片的负样本总数
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors) #将一个batch中的图片产生的anchor的标签按不同的特征层分开（共5个特征层）
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'd_score'))
    def loss_single(self, cls_score, bbox_pred, d_score, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, D_labels_list, num_total_samples):

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        # pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        # neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)
        # num_pos_samples = pos_inds.size(0)
        # num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        # if num_neg_samples > neg_inds.size(0):
        #     num_neg_samples = neg_inds.size(0)
        # topk_loss_cls_neg, topk_idces = loss_cls[neg_inds,0].topk(num_neg_samples)
        # loss_cls_pos = loss_cls[pos_inds].sum()
        # loss_cls_neg = topk_loss_cls_neg.sum()
        # loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'd_scores'))
    def loss(self, cls_scores, bbox_preds, D_scores, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, **kwargs):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] #得到5个特征图的大小
        assert len(featmap_sizes) == self.anchor_generator.num_levels 
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device) #anchor_list：每张图每个特征层生成的anchor(坐标映射为原图)，valid_flag_list：生成的anchor是否有效
        if self.last_activation == 'sigmoid' or self.last_activation == 'relu':
            label_channels = self.cls_out_channels
        elif self.last_activation == 'softmax' or self.last_activation == 'EDL_BG':
            label_channels = 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas,
                                           gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels,
                                           label_channels=label_channels) #得到一个batch中每张图片在不同特征层（5个）上anchor的标签，标签权重（区分正负样本），回归标签，回归框权重（区分正样本），所有图片正样本总数以及负样本总数
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)#等于所有图片的所有正样本数
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]] #每个特征层的anchor总数（共5层）
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i])) #将一个batch中每张图片在不同特征层上生成的anchor坐标各自拼接在一块
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors) #将所有图片的anchor分配到不同特征层上（与anchor_list不同，anchor_list会按图片进行分，而all_anchor_list会先分为5个特征层，每个特征层包含了所有图片在该特征层的anchor坐标，即(batchsize ,..,4))
        head_info = ['cls_scores','bbox_preds','all_anchor_list','labels_list','label_weights_list',
                     'bbox_targets_list','bbox_weights_list','num_total_samples']
        head_out = (head_info, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list
                      ,bbox_targets_list, bbox_weights_list, num_total_samples)
        losses_cls, losses_bbox, losses_noR = multi_apply(self.loss_single,    #损失 loss_single--->Lambda_L2.py
            cls_scores, bbox_preds, all_anchor_list, labels_list, #cls_scores, bbox_preds, all_anchor_list, labels_list,label_weights_list, bbox_targets_list, bbox_weights_list 都是5个特征层的形式，每个特征层的相关输出shape都是[bachsize,....]
            label_weights_list, bbox_targets_list, bbox_weights_list, [0,1,2,3,4],
            num_total_samples=num_total_samples, featmap_sizes = featmap_sizes, **kwargs)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_noR=losses_noR), head_out #返回每个特征层上的损失值，loss_cls和loss_bbox在每个特征层上是一个值，而loss_noR在每个特征层上是一个向量（向量长度=特征层上的anchor总数*batchsize）

    @force_fp32(apply_to=('L_scores'))
    def loss_L(self, L_scores, head_out, losses, **kwargs): #losses = loss_noR
        label_weights_list = head_out[5] #得到label_weights_list(正负样本)
        bbox_weights_list = head_out[7]  #得到bbox_weights_list(正样本)
        losses_L, _ = multi_apply(self.loss_single_L, L_scores, losses, label_weights_list, bbox_weights_list, **kwargs) #loss_single_L-->Lambda_L2.py
        return dict(loss_L=losses_L) #返回5个特征层上的loss_L
    
    @force_fp32(apply_to=('R_scores'))
    def loss_R(self, R_scores, head_out,losses, **kwargs):
        all_anchor_list = head_out[3]
        label_weights_list = head_out[5]
        bbox_weights_list = head_out[7]
        bbox_targets_list = head_out[6]
        # num_total_samples = torch.full((1,),head_out[8])
        # num = num_total_samples.expand(5, 1)
        losses_R, _ = multi_apply(self.loss_single_R, R_scores,losses, bbox_targets_list, label_weights_list, bbox_weights_list,all_anchor_list, **kwargs)
        return dict(loss_R=losses_R)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, rescale=False, with_nms=True, **kwargs):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores) #5

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)#grid_anchors-->anchor_generator.py,得到每个特征层上的anchor坐标(映射到原图上)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export(): #false
            assert len(img_metas) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])]
        scale_factors = [img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])]

        if with_nms: 
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale, **kwargs)#挑选样本时,-->Lambda_L2.py
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale, with_nms, **kwargs)#验证时，-->Lambda_L2.py
        return result_list #得到预测框和预测类别（nms后，一张图片有最多100个框）

    def _get_bboxes(self, mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, img_shapes, scale_factors, cfg, rescale=False,
                    with_nms=True, **kwargs):
        """Transform outputs for a batch item into bbox predictions.
        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=mlvl_cls_scores[0].device, dtype=torch.long)

        mlvl_bboxes, mlvl_scores = [], []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
            if self.last_activation == 'sigmoid':
                scores = cls_score.sigmoid()
            elif self.last_activation == 'relu' or self.last_activation == 'EDL_BG':
                alphas = cls_score.relu() + 1
                S = alphas.sum(dim=2, keepdim=True) + 1e-20
                Smax, _ = S.max(dim=1, keepdim=True)
                gamma = 1
                scores = alphas / ((1-gamma)*Smax + gamma*S)
            elif self.last_activation == 'softmax':
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.last_activation == 'sigmoid' or self.last_activation == 'relu':
                    max_scores, _ = scores.max(-1)
                elif self.last_activation == 'softmax' or self.last_activation == 'EDL_BG':
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if self.last_activation == 'softmax':
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.last_activation == 'relu' or self.last_activation == 'sigmoid':
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label, idces = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                                                            cfg.max_per_img, return_inds=True)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [tuple(mlvl_bs) for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)]

        return det_results
        # cScore = mlvl_scores[:, :-1]
        # flatten_score = cScore.contiguous().view(-1)
        # nmsIdx = (flatten_score > cfg.score_thr).nonzero()[idces]  # [100,1]
        # bbIdx = (nmsIdx // 20)[:, 0]
        # cIdx = (nmsIdx % 20)[:, 0]
        # nms_scores = cScore[bbIdx]
        # entropy = (-nms_scores * nms_scores.log()).sum(dim=1)
        #
        # return det_results, entropy

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)