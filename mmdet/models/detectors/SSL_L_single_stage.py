import warnings
import pdb
import torch

from mmdet.core import bbox2result, bbox2tupleresult
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .SSL_Lambda import SSLBase_L_Detector
from mmdet.utils.functions import *

@DETECTORS.register_module()
class SSL_L_SingleStageDetector(SSLBase_L_Detector):
    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(SSL_L_SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):                     # c2             c3            c4              c5
        x = self.backbone(img)  # ---> resnet.py; x :[2,256,144,256];[2,512,72,128];[2,1024,36,64];[2,2048,18,32]
        if self.with_neck:                                # P3            P4            P5            P6           P7
            x = self.neck(x)   #---->fpn.py(forward)； x: [2,256,72,128];[2,256,36,64];[2,256,18,32];[2,256,9,16];[2,256,5,8]
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        super(SSL_L_SingleStageDetector, self).forward_train(img, img_metas) #forward_train--->SSL_lambda.py
        x = self.extract_feat(img)
                                          #---->Lambda_L1.py/Lambda_L2.py/...
        losses, head_out = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs)#losses: {loss_cls, loss_bbox, loss_noR},
                                                                                                                       #head_info = ['cls_scores','bbox_preds','all_anchor_list','labels_list','label_weights_list','bbox_targets_list','bbox_weights_list','num_total_samples']
        # There is loss_noR in losses #
        # visualize(img[0], 'visualization/img.jpg')
        # for i, (feat, loss) in enumerate(zip(x, losses['loss_noR'])):
        #     _,_,H,W = feat.shape
        #     tmp = loss.reshape(2,H,W,9)[0].sum(dim=-1)
        #     visualize(tmp, f'visualization/loss_{i}.jpg', size=(H*(2**i),W*(2**i)), color=False)
        feat_out = [i.detach() for i in x] #feat_out与x数值一样，但两者不关联，对feat_out的计算与操作将不会影响x
        return losses, head_out, feat_out

    def forward_train_L(self, loss, head_out, feat_out, **kwargs,):
        losses = self.bbox_head.forward_train_L(loss, head_out, feat_out, **kwargs)#-->Lambda_L2.py(根据选择所定)
        return losses
    
    def forward_train_R(self,loss,head_out, feat_out, **kwargs,):
        losses = self.bbox_head.forward_train_R(loss,head_out, feat_out, **kwargs) #Lambda_L2.py
        return losses
    
    def forward_train_ll(self, loss, head_out, feat_out, **kwargs,):
        losses = self.bbox_head.forward_train_ll(loss, head_out, feat_out, **kwargs)#-->Lambda_L2.py(根据选择所定)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        feat = self.extract_feat(img) #五个特征层
        if kwargs['isEval']:
            _results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)#simple_test-->Lambda_L2.py，得到nms后的预测框和预测类别
            if kwargs['isUnc']:
                results_list, entropy_list = _results_list[0], _results_list[1]
                bbox_results = [
                    bbox2tupleresult(det_bboxes, det_labels, entropy_list, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in results_list
                ]
            elif not kwargs['isUnc']:
                results_list = _results_list
                bbox_results = [ #bbox2result-->transforms.py
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in results_list
                ] #将预测框按类别聚集
            return bbox_results 
        else:
            # if 'justOut' in kwargs and kwargs['justOut']:
            #     outs, L_scores = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
            #     return outs, L_scores
            results_list, *uncertainties = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)#simple_test-->Lambda_L2.py
            if self.test_cfg.uncertainty_pool == 'Entropy_NoNMS':
                return (results_list, *uncertainties)
            elif self.test_cfg.uncertainty_pool == 'Entropy_ALL':
                return (results_list, *uncertainties)
            elif self.test_cfg.uncertainty_pool == 'Entropy_NMS':
                return (results_list, *uncertainties)
            bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                            for det_bboxes, det_labels in results_list]
            return bbox_results, uncertainties

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.
        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.
        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels