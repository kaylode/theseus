import os, sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from .fpn import getFPN
from .utils import *


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=80, backbone = 'FPN50', pretrained = False, input_size = (300,300), device = None):
        super(RetinaNet, self).__init__()

        self.n_classes = num_classes
        self.device = device if device is not None else torch.device("cpu")
        self.anchor_areas = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 512 * 512]  # p3->p7
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh()
        self.input_size = input_size

        self.fpn = getFPN(backbone)
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        
        self.priors_cxcy = self._get_anchor_boxes(torch.Tensor(input_size))

        
        if self.pretrained:
            self.load_state_dict(backbone)
        self.freeze_bn()
        
    def forward(self, inputs):
        x = inputs
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        prediction = {}
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4, H, W] -> [N, H, W, 9*4] -> [N, H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N, 9*80, H, W] -> [N, H, W, 9*80] -> [N, H*W*9, 80]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        
        return loc_preds, cls_preds
    
    def _get_anchor_wh(self):
        """
        Compute anchor width and height for each feature map.
        :return: anchor_wh: (tensor) anchor wh, (size) [#fm, #anchors_per_cell, 2]
        """
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h=ar
                w = math.sqrt(s * ar)
                h = w / ar
                for sr in self.scale_ratios:
                    anchor_w = w * sr
                    anchor_h = h * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)
    
    
    def _get_anchor_boxes(self, input_size):
        """
        Compute anchor boxes for each feature map. Format: (cx,cy,w,h)
        :param input_size: the size of input image
        :return: boxes: (list) anchor boxes for each feature map. Each of size [#anchors, 4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        """
        
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]
        # num_anchors_per_level = [int(fs[0]) * int(fs[1]) * 9 for fs in fm_sizes]

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5  # plus 0.5 to each cell's center, [fm_w*fm_h, 2]
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(
                fm_h, fm_w, 9, 2)  # convert the center to original image, and expand to all anchors
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat([xy, wh], 3)  # [x, y, w, h]
            
            # Normalize coordinate
            i_h, i_w = input_size
            box[:,:,:,0] = box[:,:,:,0]*1.0 / i_w
            box[:,:,:,1] = box[:,:,:,1]*1.0 / i_h
            box[:,:,:,2] = box[:,:,:,2]*1.0 / i_w
            box[:,:,:,3] = box[:,:,:,3]*1.0 / i_h
      
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0).to(self.device)
    

    def detect2(self, batch_loc_preds, batch_cls_preds, min_score=0.02, nms_thresh = 0.7):
        """
        Decode outputs back to bounding box locations and class labels.
        :param loc_preds: (tensor) predicted locations, sized [#anchors, 4]
        :param cls_preds: (tensor) predicted class labels, sized [#anchors, #classes]
        :param input_size: (int/tuple) the input size of original image
        :return:
            boxes: (tensor) decode box locations, sized [#obj, 4]
            labels: (tensor) class labels for each box, sized [#obj,].
        """
        boxes_keep = []
        labels_keep = []
        scores_keep = []
        
        for loc_preds, cls_preds in zip(batch_loc_preds, batch_cls_preds):
            boxes = change_box_order(
                gcxgcy_to_cxcy(loc_preds, self.priors_cxcy),order = 'cxcy2xyxy')

            scores, labels = cls_preds.max(dim=1)
            
            ids = scores > min_score
            ids = ids.nonzero().squeeze()

            new_boxes = boxes.clone()
            new_scores = scores.clone()
        
            keep = box_nms(new_boxes, new_scores, nms_thresh)
            boxes_keep.append(boxes[ids][keep])
            labels_keep.append(labels[ids][keep])
            scores_keep.append(scores[ids][keep])
        return {
            'boxes': boxes_keep,
            'labels': labels_keep,
            'scores': scores_keep}


    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """
        Freeze BatchNorm layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def load_state_dict(self, name):
        if name=='FPN50':   
            
            try:
                d = torch.load('weights/pretrained/resnet50-19c8e357.pth')
            except:
                print('Pretrained weights not found')
                return None

            print('Loading pretrained {}...'.format(name))
            dd = self.fpn.state_dict()
            for k in d.keys():
                if not k.startswith('fc'):  # skip fc layers
                    dd[k] = d[k]
         
            for m in self.fpn.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            for m in self.cls_head.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

            for m in self.loc_head.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

            pi = 0.01
            init.constant_(self.cls_head[-1].bias, -math.log((1 - pi) / pi))

            self.fpn.load_state_dict(dd)
            
            print('Loaded pretrained model!')

    
    def detect(self, predicted_locs, predicted_scores, min_score=0.04, max_overlap=0.5, top_k = 200, gpu = True):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        print(predicted_locs)
        print(predicted_scores)
        # Detect object on cpu 
        if not gpu:
            self.device = torch.device("cpu")
            predicted_locs = predicted_locs.cpu()
            predicted_scores = predicted_scores.cpu()

        batch_size = predicted_locs.shape[0]
        n_priors = self.priors_cxcy.shape[0]
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = change_box_order(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy),order = 'cxcy2xyxy')  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)
            #print(max_scores)
            #print(best_label)
            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    condition = overlap[box] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(self.device)
                    #print(torch.tensor(condition,dtype=torch.uint8))
                    suppress = torch.max(suppress, condition)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return {
            'boxes': all_images_boxes,
            'labels': all_images_labels,
            'scores': all_images_scores}  # lists of length batch_size
