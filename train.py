from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil

from eval import evaluate_model


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
# parser.add_argument('--start_iter', default=0, type=int,
#                     help='Resume training at this iter')
parser.add_argument("--epochs", default=100, type=int,
                    help="Number of training epochs")
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')

parser.add_argument("--split-file", default=None, type=str, 
                    help="Path to training split file")
parser.add_argument("--save-dir", default="checkpoints", type=str,
                    help="Directory to save checkpoints and logs")
parser.add_argument("--experiment-name", default=None, type=str)
parser.add_argument("--save-freq", type=int, default=9999, 
                    help="Save checkpoint every N epochs")

args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# if not os.path.exists(args.save_folder):
#     os.mkdir(args.save_folder)
save_dir = Path(args.save_dir)
weights_dir = save_dir

weights_dir.mkdir(parents=True, exist_ok=True)


def should_validate(epoch_num, total_epochs):
    """Epoch-based validation schedule (epoch_num is 1-indexed):

        1   - 150 : never
        151 - 159 : never (gap before the first explicit checkpoint)
        160 - 239 : every 10 epochs, starting at 160
        240 - end : every 5 epochs, starting at 240
        final epoch (== total_epochs): always, regardless of the above
    """
    if epoch_num >= total_epochs:
        return True
    if epoch_num < 160:
        return False
    if epoch_num < 240:
        return (epoch_num - 160) % 10 == 0
    return (epoch_num - 240) % 5 == 0


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               split_file=args.split_file,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        
    train_loader = data.DataLoader(dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate, drop_last=False,
                                persistent_workers=args.num_workers > 0,
                                pin_memory=True, generator=torch.Generator(device=device))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ckpt = torch.load(args.resume, map_location=device)
        ssd_net.load_state_dict(ckpt["model"])
    else:
        # vgg_weights = torch.load(args.save_folder + args.basenet)
        vgg_weights = torch.load(Path(args.save_folder) / args.basenet, map_location="cpu")
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    net = ssd_net.to(device)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
    
    net = ssd_net

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    start_epoch = 0

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    
    print("=" * 60)
    print("Dataset       :", dataset.name)
    print("Training imgs :", len(dataset))
    print("Epochs        :", args.epochs)
    print("Batch size    :", args.batch_size)
    print("Learning rate :", args.lr)
    print("Save dir      :", save_dir)

    # print("=" * 60)
    # net.train()
    # # loss counters
    # loc_loss = 0
    # conf_loss = 0
    # epoch = 0
    # print('Loading the dataset...')

    # epoch_size = len(dataset) // args.batch_size
    # print('Training SSD on:', dataset.name)
    # print('Using the specified args:')
    # print(args)

    # step_index = 0
    print("\nStarting training...\n")
    best_mAP = 0.0
    for epoch in range(start_epoch, args.epochs):
        net.train()

        lr = adjust_learning_rate(optimizer, epoch + 1)

        epoch_loc_loss = 0.0
        epoch_conf_loss = 0.0
        epoch_total_loss = 0.0

        if args.visdom:
            vis_title = 'SSD.PyTorch on ' + dataset.name
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
            epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = [ann.to(device, non_blocking=True) for ann in targets]

            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            epoch_loc_loss += loss_l.item()
            epoch_conf_loss += loss_c.item()
            epoch_total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                loc=f"{loss_l.item():.3f}",
                conf=f"{loss_c.item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )
        
        num_batches = len(train_loader)
        epoch_loc_loss /= num_batches
        epoch_conf_loss /= num_batches
        epoch_total_loss /= num_batches

        print()
        print(
            f"Epoch {epoch+1:03d} | "
            f"Loc {epoch_loc_loss:.4f} | "
            f"Conf {epoch_conf_loss:.4f} | "
            f"Total {epoch_total_loss:.4f}"
        )
        mAP = None
        aps = None
        is_best = False

        if should_validate(epoch + 1, args.epochs):
            metrics = evaluate_model(
                net=net,
                dataset_root=args.dataset_root,
                device=device,
                save_dir=os.path.join(args.save_dir, "eval")
            )

            mAP = metrics["mAP"]
            aps = metrics["aps"]

            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP
        

        log_epoch(
            epoch=epoch + 1, lr=lr, 
            loc_loss=epoch_loc_loss, conf_loss=epoch_conf_loss,
            mAP=mAP, aps=aps or {}
        )

        save_checkpoint(
            model=net, optimizer=optimizer, 
            epoch=epoch, best=is_best, best_mAP=best_mAP, 
            save_dir=weights_dir
        )

    save_checkpoint(
        model=net,
        optimizer=optimizer,
        epoch=args.epochs,
        best=False,
        best_mAP=best_mAP,
        save_dir=weights_dir,
        filename="checkpoint_last.pth",
    )


# def adjust_learning_rate(optimizer, gamma, step):
#     """Sets the learning rate to the initial LR decayed by 10 at every
#         specified step
#     # Adapted from PyTorch Imagenet example:
#     # https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     lr = args.lr * (gamma ** (step))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optmizer, epoch):
    """
    Epoch-based step learning rate schedule.

    Epochs:
        1 - 239 : 1e-3
        240-300 : 1e-4
    """
    lr = args.lr

    if epoch >= 240:
        lr *= 0.1

    for param_group in optmizer.param_groups:
        param_group["lr"] = lr

    return lr

def log_epoch(epoch, lr, loc_loss, conf_loss, aps, mAP):
    history_csv = save_dir / "epoch_history.csv"

    row = pd.DataFrame([{
        "epoch": epoch,
        "lr": lr,
        "loc_loss": loc_loss,
        "conf_loss": conf_loss,
        "total_loss": loc_loss + conf_loss,
        "mAP": mAP,
        **aps,
    }])

    if history_csv.exists():
        history = pd.read_csv(history_csv)
        history = history[history["epoch"] != epoch]
        history = pd.concat([history, row], ignore_index=True)
    else:
        history = row
    
    history = history.sort_values("epoch").reset_index(drop=True)

    history.to_csv(history_csv, index=False)

def save_checkpoint(model, optimizer, epoch, save_dir, 
                    best=False, best_mAP=None, filename="checkpoint_latest.pth"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle DataParallel
    state_dict = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    state = {
        "epoch": epoch,
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "best_mAP": best_mAP,
    }

    latest_ckpt = save_dir / filename
    torch.save(state, latest_ckpt)

    if best:
        shutil.copy2(latest_ckpt, save_dir / "checkpoint_best.pth")

    # Periodic checkpoints
    if epoch % args.save_freq == 0:
        shutil.copy2(
            latest_ckpt,
            save_dir / f"checkpoint_epoch_{epoch:03d}.pth",
        )


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()