from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import argparse
import csv
import os
import shutil
import cv2
import numpy as np
import time

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.ops import RoIPool, nms

import _init_paths
import models
from config import cfg as c
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

from datasets.data_processing import img_preprocessing
from models.backbone import Backbone
from models.head import Head
from oim.labeled_matching_layer import LabeledMatchingLayer
from oim.unlabeled_matching_layer import UnlabeledMatchingLayer
from rpn.proposal_target_layer import ProposalTargetLayer
from rpn.rpn_layer import RPN
from utils.boxes import bbox_transform_inv, clip_boxes
from utils.config import cfg
from utils.utils import smooth_l1_loss

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person'
]

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, c.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(c.MODEL.IMAGE_SIZE[0]), int(c.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            c,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--c', type=str, default='demo/inference-config.yaml')
    #parser.add_argument('--image',type=str)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

#* Euclidean distance between points
def dist_feat(points):
    if type(points) != torch.Tensor:
        points = torch.Tensor(points)
    if points.dim() == 3:
        points.squeeze_()
    assert points.dim() == 2
    feat = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            feat.append(torch.dist(points[i], points[j]).item())
    return torch.Tensor(feat)

class Network(nn.Module):
    """
    Person search network.

    Paper: Joint Detection and Identification Feature Learning for Person Search
           Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang
    """

    def __init__(self, check):
        super(Network, self).__init__()
        rpn_depth = 1024  # Depth of the feature map fed into RPN
        num_classes = 2  # Background and foreground
        self.backbone = Backbone()
        self.head = Head()
        self.rpn = RPN(rpn_depth)
        self.roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.cls_score = nn.Linear(2048, num_classes)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feature = nn.Linear(2048, 256)
        self.proposal_target_layer = ProposalTargetLayer(num_classes)
        self.labeled_matching_layer = LabeledMatchingLayer()
        self.unlabeled_matching_layer = UnlabeledMatchingLayer()
        self.check = check
        self.freeze_blocks()

    def forward(self, img, img_info, gt_boxes, real_img, name, fl, probe_roi=None):
        #print("\n               INPUT")
        #print("===============================================")
        #print("1. img :", img.cpu().numpy().shape)
        #print("2. img_info :", img_info.cpu().numpy())
        #print("3. gt_boxes :", gt_boxes[0].cpu().numpy())
        #print("4. real_img :", real_img.cpu().numpy().shape)
        #print("===============================================")
        
        """
        Args:
            img (Tensor): Single image data.
            img_info (Tensor): (height, width, scale)
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.
            probe_roi (Tensor): Take probe_roi as proposal instead of using RPN.

        Returns:
            proposals (Tensor): Region proposals produced by RPN in (0, x1, y1, x2, y2) format.
            probs (Tensor): Classification probability of these proposals.
            proposal_deltas (Tensor): Proposal regression deltas.
            features (Tensor): Extracted features of these proposals.
            rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox and loss_oim (Tensor): Training losses.
        """
        assert img.size(0) == 1, "Single batch only."
        
        args = parse_args()
        update_config(c, args)
        
        pose_model = eval('models.'+c.MODEL.NAME+'.get_pose_net')(
        c, is_train=False
        )
        
        if c.TEST.MODEL_FILE:
            #print('=> loading model from {}'.format(c.TEST.MODEL_FILE))
            pose_model.load_state_dict(torch.load(c.TEST.MODEL_FILE), strict=False)

        
        pose_model = torch.nn.DataParallel(pose_model, device_ids=c.GPUS)
        pose_model.to(CTX)
        pose_model.eval()

        num_box = len(gt_boxes)
        keys = []
        ids = []
        
        for i in range(num_box):
            #print(i)
            
            gt = gt_boxes[i].cpu().numpy()
            re_img = real_img[0].cpu().numpy()
            re_img = cv2.resize(re_img, (img_info[1], img_info[0]))
            
            if fl == True:
                re_img = cv2.flip(re_img, 1)
            
            x1, y1, x2, y2, id = int(gt[0]), int(gt[1]), int(gt[2]),int(gt[3]), int(gt[-1])
            
            img_ = re_img.copy()
            img_ = cv2.rectangle(img_, (x1,y1), (x2,y2), (255,0,0), 3)
            cv2.putText(img_, f'{id}', (x1-5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            ids.append(id)
            image_bgr = img_

            # estimate on the image
            image = image_bgr[:, :, [2, 1, 0]] # rgb
            
            input = []
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb/255.).permute(2,0,1).float().to(CTX)
            input.append(img_tensor)
            #gb = gt_boxes[num].numpy()
            
            # object detection box
            box = [(x1,y1), (x2,y2)]
            
            # pose estimation
            center, scale = box_to_center_scale(box, c.MODEL.IMAGE_SIZE[0], c.MODEL.IMAGE_SIZE[1])
            image_pose = image.copy() if c.DATASET.COLOR_RGB else image_bgr.copy()
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            #print(pose_preds)
            print(pose_preds.shape)
            dist = dist_feat(pose_preds)
            #print(dist)
            print(type(dist))
            print(len(dist)); exit()
            
            #exit()
            #print(pose_preds)
            if len(pose_preds)>=1:
                keys.append(pose_preds)
                for kpt in pose_preds:
                    draw_pose(kpt,image_bgr) # draw the poses
            os.makedirs(f"key_results/{id}", exist_ok = True)
            save_path = f'key_results/{id}/{name}_{i}.jpg'
            cv2.imwrite(save_path,image_bgr)
            print('the result image has been saved as {}'.format(save_path))
        #print("# of keypoints : ",len(keys))
        #print("ids : ", ids)
        
        #! gt box cropped --> people
        #! each id --> ids
        # Extract basic feature from image data
        base_feat = self.backbone(img)

        if probe_roi is None:
            # Feed basic feature map to RPN to obtain rois
            proposals, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feat, img_info, gt_boxes)
        else:
            # Take given probe_roi as proposal if probe_roi is not None
            proposals, rpn_loss_cls, rpn_loss_bbox = probe_roi, 0, 0

        if self.training:
            # Sample some proposals and assign them ground-truth targets
            (
                proposals,
                cls_labels,
                pid_labels,
                gt_proposal_deltas,
                proposal_inside_ws,
                proposal_outside_ws,
            ) = self.proposal_target_layer(proposals, gt_boxes)
        else:
            cls_labels, pid_labels, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws = [
                None
            ] * 5
        
        #print("\n* ROI Pooling      : input = proposals, feature map")
        #print("===============================================")
        #print("* applied NMS (#proposals : 2000 --> 128)")
        #print("Proposals shape : ",proposals.shape)
        # RoI pooling based on region proposals
        pooled_feat = self.roi_pool(base_feat, proposals)
        #print("\n* proposals --> projection on feature map")
        #print("Feature map shape : ",base_feat.shape)
        #print("\n* roi pooling result : #proposal x 1024 x 14 x 14")
        #print("Pooled proposals shape : ",pooled_feat.shape)
        #print("===============================================")
        # Extract the features of proposals
        proposal_feat = self.head(pooled_feat).squeeze(2).squeeze(2)

        #print("==> Output shape : ", proposal_feat.shape)
        #print("\n\n* Get probs, deltas and features")
        scores = self.cls_score(proposal_feat)
        probs = F.softmax(scores, dim=1)
        proposal_deltas = self.bbox_pred(proposal_feat)
        features = F.normalize(self.feature(proposal_feat))
        #print("Probs shape : ", probs.shape)
        #print("deltas shape : ", proposal_deltas.shape)
        #print("features shape : ", features.shape)

        if self.training:
            loss_cls = F.cross_entropy(scores, cls_labels)
            loss_bbox = smooth_l1_loss(
                proposal_deltas, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws
            )

            # OIM loss
            labeled_matching_scores = self.labeled_matching_layer(features, pid_labels)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(features, pid_labels)
            unlabeled_matching_scores *= 10
            matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            pid_labels = pid_labels.clone()
            pid_labels[pid_labels == -2] = -1
            loss_oim = F.cross_entropy(matching_scores, pid_labels, ignore_index=-1)
        else:
            loss_cls, loss_bbox, loss_oim = 0, 0, 0
        #print("\n* Calculate Loss(cls, bbox)")
        #print("loss_cls : ", loss_cls.data.cpu().numpy())
        #print("loss_bbox : ", loss_bbox.data.cpu().numpy())
        #print("\n* Calculate Loss(OIM)")
        #print("labeled matching score shape: ", labeled_matching_scores.shape)
        #print("unlabeled matching score shape: ", unlabeled_matching_scores.shape)
        #print("==> loss_oim : ",loss_oim.data.cpu().numpy())
        #print("\n")
        if self.check == True : exit()
        return (
            proposals,
            probs,
            proposal_deltas,
            features,
            rpn_loss_cls,
            rpn_loss_bbox,
            loss_cls,
            loss_bbox,
            loss_oim,
        )

    def freeze_blocks(self):
        """
        The reason why we freeze all BNs in the backbone: The batch size is 1
        in the backbone, so BN is not stable.

        Reference: https://github.com/ShuangLI59/person_search/issues/87
        """
        for p in self.backbone.SpatialConvolution_0.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # Frozen all bn layers in backbone
        self.backbone.apply(set_bn_fix)

    def train(self, mode=True):
        """
        It's not enough to just freeze all BNs in backbone.
        Setting them to eval mode is also needed.
        """
        nn.Module.train(self, mode)

        if mode:
            # Set all bn layers in backbone to eval mode
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)

    def inference(self, img, probe_roi=None, threshold=0.75):
        """
        End to end inference. Specific behavior depends on probe_roi.
        If probe_roi is None, detect persons in the image and extract their features.
        Otherwise, extract the feature of the probe RoI in the image.

        Args:
            img (np.ndarray[H, W, C]): Image of BGR order.
            probe_roi (np.ndarray[4]): The RoI to be extracting feature.
            threshold (float): The threshold used to remove those bounding boxes with low scores.

        Returns:
            detections (Tensor[N, 5]): Detected person bounding boxes in
                                       (x1, y1, x2, y2, score) format.
            features (Tensor[N, 256]): Features of these bounding boxes.
        """
        device = self.cls_score.weight.device
        processed_img, scale = img_preprocessing(img)
        # [C, H, W] -> [N, C, H, W]
        processed_img = torch.from_numpy(processed_img).unsqueeze(0).to(device)
        # img_info: (height, width, scale)
        img_info = torch.Tensor([processed_img.shape[2], processed_img.shape[3], scale]).to(device)
        if probe_roi is not None:
            probe_roi = torch.from_numpy(probe_roi).float().view(1, 4)
            probe_roi *= scale
            # Add an extra 0, which means the probe_roi is from the first image in the batch
            probe_roi = torch.cat((torch.zeros(1, 1), probe_roi.float()), dim=1).to(device)

        with torch.no_grad():
            proposals, probs, proposal_deltas, features, _, _, _, _, _ = self.forward(
                processed_img, img_info, None, probe_roi
            )

        if probe_roi is not None:
            return features

        # Unscale proposals back to raw image space
        proposals = proposals[:, 1:5] / scale
        # Unnormalize proposal deltas
        num_classes = proposal_deltas.shape[1] // 4
        stds = torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(num_classes).to(device)
        means = torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(num_classes).to(device)
        proposal_deltas = proposal_deltas * stds + means
        # Apply proposal regression deltas
        boxes = bbox_transform_inv(proposals, proposal_deltas)
        boxes = clip_boxes(boxes, img.shape)

        # Remove those boxes with scores below the threshold
        j = 1  # Only consider foreground class
        keep = torch.nonzero(probs[:, j] > threshold)[:, 0]
        boxes = boxes[keep, j * 4 : (j + 1) * 4]
        probs = probs[keep, j]
        features = features[keep]

        # Remove redundant boxes with NMS
        detections = torch.cat((boxes, probs.unsqueeze(1)), dim=1)
        keep = nms(boxes, probs, cfg.TEST.NMS)
        detections = detections[keep]
        features = features[keep]
        
        return detections, features
