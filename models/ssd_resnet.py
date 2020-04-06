from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from models.l2norm import L2Norm
from torchvision import transforms
import itertools
import os
import numpy as np
from utils import *

# from torchvideotransforms import video_transforms, volume_transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AuxiliaryConvolutions(nn.Module):
    """ Auxiliary Convolottion following the structure of SSD auxiliary convolutions
    """
    def __init__(self, n_classes):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary convolutions added to base network
        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=(1, 1), padding=0)
        self.conv1_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, padding=1)
        
        self.conv2_1 = nn.Conv2d(512, 128, kernel_size=(1, 1), padding=0)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=(1, 1), padding=0)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=0)  

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=(1, 1), padding=0)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=0) 

        self.pool4 = nn.AdaptiveAvgPool2d(n_classes * 10)
        self.fc1 = nn.Linear(512, n_classes)
        # self.fc2 = nn.linear

    def forward(self, conv_resnet3d_feats):
        x = F.relu(self.conv1_1(conv_resnet3d_feats))
        x = F.relu(self.conv1_2(x))
        conv_aux_1_feats = x

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        conv_aux_2_feats = x

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        conv_aux_3_feats = x

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        conv_aux_4_feats = x

        return conv_aux_1_feats, conv_aux_2_feats, conv_aux_3_feats, conv_aux_4_feats


class PredictionConvolution(nn.Module):
    """
    Convolutions used to map location prediction to feature space
    """
    def __init__(self, n_classes):
        """
        :param n_clip_bboxes: number of bounding boxes per clip
        :param n_classes: number of actions
        """
        super(PredictionConvolution, self).__init__()

        self.n_classes = n_classes

        # Define number of priors per grid location in feature map
        n_boxes = {
            'conv_resnet_3': 4,
            "conv_resnet_4": 6,
            "conv_aux_1": 6,
            "conv_aux_2": 6,
            "conv_aux_3": 4,
            "conv_aux_4": 4
        }

        # Localization Prediction convolutions
        self.loc_conv_resnet_3 = nn.Conv2d(256, n_boxes['conv_resnet_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv_resnet_4 = nn.Conv2d(512, n_boxes['conv_resnet_4'] * 4, kernel_size=3, padding=1)
        self.loc_conv_aux_1 = nn.Conv2d(512, n_boxes['conv_aux_1'] * 4, kernel_size=3, padding=1)
        self.loc_conv_aux_2 = nn.Conv2d(256, n_boxes['conv_aux_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv_aux_3 = nn.Conv2d(256, n_boxes['conv_aux_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv_aux_4 = nn.Conv2d(256, n_boxes['conv_aux_4'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv_resnet_3 = nn.Conv2d(256, n_boxes['conv_resnet_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv_resnet_4 = nn.Conv2d(512, n_boxes['conv_resnet_4'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv_aux_1 = nn.Conv2d(512, n_boxes['conv_aux_1'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv_aux_2 = nn.Conv2d(256, n_boxes['conv_aux_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv_aux_3 = nn.Conv2d(256, n_boxes['conv_aux_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv_aux_4 = nn.Conv2d(256, n_boxes['conv_aux_4'] * n_classes, kernel_size=3, padding=1)


    def forward(self, conv_resnet_3_feats, conv_resnet_4_feats, conv_aux_1_feats, conv_aux_2_feats, conv_aux_3_feats, conv_aux_4_feats):
        
        # 
        batch_size = conv_resnet_3_feats.size(0)

        # -------------------------- Localization Prediction ------------------------- #        
        l_conv_res_3 = self.loc_conv_resnet_3(conv_resnet_3_feats)
        l_conv_res_3 = l_conv_res_3.permute(0, 2, 3, 1).contiguous()
        l_conv_res_3 = l_conv_res_3.view(batch_size, -1, 4)

        l_conv_res_4 = self.loc_conv_resnet_4(conv_resnet_4_feats)
        l_conv_res_4 = l_conv_res_4.permute(0, 2, 3, 1).contiguous()
        l_conv_res_4 = l_conv_res_4.view(batch_size, -1, 4)

        l_conv_aux_1 = self.loc_conv_aux_1(conv_aux_1_feats)
        l_conv_aux_1 = l_conv_aux_1.permute(0, 2, 3, 1).contiguous()
        l_conv_aux_1 = l_conv_aux_1.view(batch_size, -1, 4)
        
        l_conv_aux_2 = self.loc_conv_aux_2(conv_aux_2_feats)
        l_conv_aux_2 = l_conv_aux_2.permute(0, 2, 3, 1).contiguous()
        l_conv_aux_2 = l_conv_aux_2.view(batch_size, -1, 4)
        
        l_conv_aux_3 = self.loc_conv_aux_3(conv_aux_3_feats)
        l_conv_aux_3 = l_conv_aux_3.permute(0, 2, 3, 1).contiguous()
        l_conv_aux_3 = l_conv_aux_3.view(batch_size, -1, 4)
        
        l_conv_aux_4 = self.loc_conv_aux_4(conv_aux_4_feats)
        l_conv_aux_4 = l_conv_aux_4.permute(0, 2, 3, 1).contiguous()
        l_conv_aux_4 = l_conv_aux_4.view(batch_size, -1, 4)

        prediction_feats = torch.cat([l_conv_res_3, l_conv_res_4, l_conv_aux_1, l_conv_aux_2, l_conv_aux_3, l_conv_aux_4], dim=1) 

        
        # ----------------------------- Class Prediction ----------------------------- #        
        # Class prediction convolutions (predict classes in localization boxes)
        c_conv_res_3 = self.cl_conv_resnet_3(conv_resnet_3_feats)
        c_conv_res_3 = c_conv_res_3.permute(0, 2, 3, 1).contiguous()
        c_conv_res_3 = c_conv_res_3.view(batch_size, -1, self.n_classes)


        c_conv_res_4 = self.cl_conv_resnet_4(conv_resnet_4_feats)
        c_conv_res_4 = c_conv_res_4.permute(0, 2, 3, 1).contiguous()
        c_conv_res_4 = c_conv_res_4.view(batch_size, -1, self.n_classes)


        c_conv_aux_1 = self.cl_conv_aux_1(conv_aux_1_feats)
        c_conv_aux_1 = c_conv_aux_1.permute(0, 2, 3, 1).contiguous()
        c_conv_aux_1 = c_conv_aux_1.view(batch_size, -1, self.n_classes)

        c_conv_aux_2 = self.cl_conv_aux_2(conv_aux_2_feats)
        c_conv_aux_2 = c_conv_aux_2.permute(0, 2, 3, 1).contiguous()
        c_conv_aux_2 = c_conv_aux_2.view(batch_size, -1, self.n_classes)

        c_conv_aux_3 = self.cl_conv_aux_3(conv_aux_3_feats)
        c_conv_aux_3 = c_conv_aux_3.permute(0, 2, 3, 1).contiguous()
        c_conv_aux_3 = c_conv_aux_3.view(batch_size, -1, self.n_classes)

        c_conv_aux_4 = self.cl_conv_aux_4(conv_aux_4_feats)
        c_conv_aux_4 = c_conv_aux_4.permute(0, 2, 3, 1).contiguous()
        c_conv_aux_4 = c_conv_aux_4.view(batch_size, -1, self.n_classes)


        classes_scores = torch.cat([c_conv_res_3, c_conv_res_4, c_conv_aux_1, 
                        c_conv_aux_2, c_conv_aux_3, c_conv_aux_4], dim=1)


        return prediction_feats, classes_scores


class SSDResnet(nn.Module):
    """Single Shot Resnet3D network
    Utilizes of a Resnet3D an auxiliary network and predictive region

    Args:
        clip_len: number of frames per clip
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, n_classes):
        super(SSDResnet, self).__init__()
        
        self.n_classes = n_classes
        
        # ------------------------------ Generate Priors ----------------------------- #
        self.priors = self.generate_priors()

        # ---------------------------- Resnet3D Base Class --------------------------- #
        self.bmdl = models.resnet18(True)
        self.bmdl.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(256, 20)

        # -------------------------- Auxiliary Convolutions -------------------------- # 
        self.aux_convs = AuxiliaryConvolutions(n_classes)

        # -------------------------- Prediction Convolutions ------------------------- #
        self.pred_convs = PredictionConvolution(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 256, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)
        
        phase = 'test'
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        feature_maps = {}
        loc = list()
        conf = list()

        # apply resnet3d up to resnet block 3 & 4
        x = self.bmdl.conv1(x)
        x = self.bmdl.bn1(x)
        x = self.bmdl.relu(x)
        x = self.bmdl.maxpool(x)

        x = self.bmdl.layer1(x)
        x = self.bmdl.layer2(x)

        x = self.bmdl.layer3(x)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        s = x / norm
        conv_res_3 = s * self.rescale_factors

        x = self.bmdl.layer4(x)
        # s = self.L2Norm(x) # may have to rescale this alyer
        conv_res_4 = x

        # Apply auxiliary convolution layers
        conv_aux_1, conv_aux_2, conv_aux_3, conv_aux_4 = self.aux_convs(x)

        # Perform convolution to extract predicted values
        locs, cls_scores = self.pred_convs(conv_res_3, conv_res_4, conv_aux_1, 
                                                conv_aux_2, conv_aux_3, conv_aux_4)

        return locs, cls_scores

    def generate_priors(self):
        """Generates Priors for individual frames
        Structure follows SSD prior
        """
        fmap_dims = {'conv_res_3': 38,
                     'conv_res_4': 19,
                     'conv_aux_1': 10,
                     'conv_aux_2': 5,
                     'conv_aux_3': 3,
                     'conv_aux_4': 1}

        obj_scales = {'conv_res_3': 0.1,
                      'conv_res_4': 0.2,
                      'conv_aux_1': 0.375,
                      'conv_aux_2': 0.55,
                      'conv_aux_3': 0.725,
                      'conv_aux_4': 0.9}

        aspect_ratios = {'conv_res_3': [1., 2., 0.5],
                         'conv_res_4': [1., 2., 3., 0.5, .333],
                         'conv_aux_1': [1., 2., 3., 0.5, .333],
                         'conv_aux_2': [1., 2., 3., 0.5, .333],
                         'conv_aux_3': [1., 2., 0.5],
                         'conv_aux_4': [1., 2., 0.5]}
        
        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes = prior_boxes.view(8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
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
        batch_size = predicted_locs.size(0)
        n_priors = self.priors.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

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
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    condition = overlap[box] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(device)
                    suppress = torch.max(suppress, condition)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

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

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss, self.alpha * loc_loss


if __name__ == "__main__":
    from data.coco import *
    from utils import *
    from models.augmentations import *

    transforms = SSDAugmentation()

    dataset = COCODetection(root="/media/ynki9/DATA/dev2/coco", transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                                num_workers=1)
    model = SSDResnet(n_classes=len(COCO_CLASSES)).to(device)
    criterion = MultiBoxLoss(model.priors).to(device)

    img, gt = next(iter(loader))

    img = img.to(device)
    gt = gt.type(torch.FloatTensor)
    bboxes = gt[:, :, 0:4].to(device)
    labels = gt[:, :, 4:].squeeze(2).to(device)
    
    pred_locs, pred_scores = model(img)

    loss = criterion(pred_locs, pred_scores, bboxes, labels)


