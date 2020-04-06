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

# ------------------------------- Model Params ------------------------------- #
clip_len = 16 # Number of frames to use per clip
dataset_root = "/media/ynki9/DATA/dev2/coco"
weights_root = "./weights"
model_version = 1
sub_version = 2
local_data = None
model_checkpoint_path = None

# ---------------------------------------------------------------------------- #
#                                Learning Params                               #
# ---------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
start_epoch = 0
end_epoch = 24
worker_count = 2
learning_rate = 1e-4
print_freq = 200
best_loss = 100
load_checkpoint = True


def main():
    global start_epoch
    global end_epoch, best_loss, model_checkpoint_path

    print(f"Initializing training for model version [{model_version}].")
    print(f"Utilizing COCO Dataset with clip length [{clip_len}] frames.")

    transforms = SSDAugmentation()

    # Training Dataset
    print("Loading Training Dataset.")
    train_dataset = COCODetection(root=dataset_root, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=worker_count)

    # Validation Dataset
    print("Loading Validation Dataset")
    # validation_dataset = COCODetection(root=dataset_root, transform=transforms)
    # val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
    #                                             num_workers=worker_count, sampler=validation_sampler)

    # initialize model
    print("Generating model version {}.".format(model_version))
    model = SSDResnet(n_classes=len(COCO_CLASSES)).to(device)

    # Initialize Criterion and Back Propagation
    criterion =  MultiBoxLoss(model.priors).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    model_checkpoint_path = "{}/checkpoint_ynet3d_v{}-{}-10.pth.tar".format(weights_root, model_version, sub_version)
    if load_checkpoint and os.path.exists(model_checkpoint_path): # Load saved model parameters          
        print("Loading checkpoint: {}".format(model_checkpoint_path))
        checkpoint = torch.load(model_checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Open Metric Writer
    metric_file_path = "{}/metrics-{}-{}.csv".format(weights_root, model_version, sub_version)
    print("Using Metric File: {}".format(metric_file_path))

    # Perform Training
    print("Performing Training: Epoch {}-{}".format(start_epoch, end_epoch))
    for epoch in range(start_epoch, end_epoch):
        print(f"# --------------------------------- EPOCH {epoch} --------------------------------- #")

        # perform training
        train_loss = train(train_loader=train_loader, model=model, epoch=epoch, 
                optimizer=optimizer, criterion=criterion, metric_file_path=metric_file_path)

        # Validate every epoch
        # val_loss, accuracy = validate(val_loader=val_loader,
        #                     model=model,
        #                     criterion=val_criterion)
                            
        # Did validation loss improve?
        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, train_loss, best_loss, is_best, accuracy)




def train(train_loader, model, epoch, optimizer, criterion, metric_file_path=None):
    model.train()      
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Losses
    losses = AverageMeter()
    localization_losses = AverageMeter()
    classification_losses = AverageMeter()

    start = time.time()    

    for i,  (img, gt) in enumerate(train_loader):
        data_time.update(time.time() - start)

        img = img.to(device)
        gt = gt.type(torch.FloatTensor)
        bboxes = gt[:, :, 0:4].to(device)
        labels = gt[:, :, 4:].squeeze(2).to(device)

        # Forward prop.
        optimizer.zero_grad()
        pred_locs, pred_scores = model(img)

        # Loss
        cls_loss, loc_loss = criterion(pred_locs, pred_scores, bboxes, labels)
        loss = cls_loss + loc_loss

        # Backward prop.
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item())
        localization_losses.update(loc_loss.item())
        classification_losses.update(cls_loss.item())

        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            # Record Status
            if metric_file_path != None:
                with open(metric_file_path, 'w+') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    if csv_writer != None:
                        metrics = [epoch, i, len(train_loader), batch_time.avg, losses.val, classification_losses.val, 
                                localization_losses.val]
                        csv_writer.writerow(metrics)                

            # Print Status
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f})\n'                  
                  'Classification Loss {classification_loss.val:.4f} ({classification_loss.avg:.4f})\t'
                  'Localization Loss {multibox_loss.val:.4f} ({multibox_loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  classification_loss=classification_losses,
                                                                  multibox_loss=localization_losses))
        
        del img, bboxes, labels, gt  # clear memory
    return losses.val


def validate(val_loader, model, criterion):    
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    multibox_losses = AverageMeter()
    classification_losses = AverageMeter()

    start = time.time()
    correct = 0
    total = 0

    # begin validation
    with torch.no_grad():
        for i, (frames, labels, bboxes) in enumerate(val_loader):
            data_time.update(time.time() - start)

            frames_flip = model.horizontal_flip(frames).to(device)
            frames = frames.to(device)
            labels = labels.to(device) 
            bboxes = bboxes.to(device)

            # Forward prop.        
            x_locs, xf_locs, x_action_label, xf_action_label = model(frames, frames_flip)
            # predictions = torch.argmax(raw_pred, dim=1)

            # Loss
            cls_loss, loc_loss, = criterion(x_locs, x_action_label, bboxes, labels)  # scalar
            loss = cls_loss + loc_loss

            # accuracy 
            _, accuracy_pred = torch.max(x_action_label.data, 1)
            total += labels.size(0)
            correct += labels.squeeze(1).eq(accuracy_pred).sum().item()

            losses.update(loss.item(), frames.size(0))
            multibox_losses.update(loc_loss.item(), frames.size(0))
            classification_losses.update(cls_loss.item(), frames.size(0))

            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\n'                  
                      'Classification Loss {classification_loss.val:.4f} ({classification_loss.avg:.4f})\t'
                      'MultiBox Loss {multibox_loss.val:.4f} ({multibox_loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses, 
                                                                      classification_loss=classification_losses,
                                                                      multibox_loss=multibox_losses))

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    return losses.avg, (100 * correct / total)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best, accuracy):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model,
             'optimizer': optimizer,
             'accuracy': accuracy}
    filename = model_checkpoint_path
    torch.save(state, filename)
    # Store Best Checkpoint
    if is_best:
        torch.save(state, "{}/BEST_imp{}-{}-{}.pth.tar".format(weights_root, model_version, sub_version, epoch))


if __name__ == "__main__":
    print(f"Executing on device: {device}")
    if len(sys.argv) > 1:        
        datastore_version = int(sys.argv[1]) # select model        
        dataset_root = f"E:\\dev2\\UCF-10{datastore_version}"
    if len(sys.argv) > 2:
        image_per_class = int(sys.argv[2])
    main()