import os
import sys
import time
import csv
import torch.utils.data
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from models.ssd_resnet import SSDResnet,  MultiBoxLoss
from helpers import AverageMeter
from data.coco import *
from utils import *
from models.augmentations import *
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def run(suppress=None):
    parser = argparse.ArgumentParser(description="Evaluates SSD model")
    parser.add_argument('--image', dest='img_path', required=True,
                        help="Input Image path", metavar="FILE")

    model_checkpoint_path = "./weights/checkpoint_ynet3d_v1-2-12.pth.tar"

    transforms = SSDAugmentation()
    target = np.zeros((12, 5))
    args = parser.parse_args()
    timg = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    img, boxes, labels = transforms(timg, target[:, :4],
                                                target[:, 4])
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.unsqueeze(0)
    checkpoint = torch.load(model_checkpoint_path)
    model = checkpoint['model']
    model.to(device)

    img = img.to(device)
    
    pred_locs, pred_scores = model(img)

    det_boxes, det_labels, d_scores = model.detect_objects(pred_locs, pred_scores, min_score=0.01, max_overlap=0.45, top_k=6)

    # Move detections to the CPU
    # det_boxes = cxcy_to_xy(det_boxes[0])
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [timg.shape[1], timg.shape[0], timg.shape[1], timg.shape[0]]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return img

    # Annotate
    annotated_image = Image.fromarray(cv2.cvtColor(timg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()#("arial.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image
        


    print(pred_locs)


if __name__ == "__main__":
    run().show()