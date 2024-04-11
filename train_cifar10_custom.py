# -*- coding: utf-8 -*-
'''

Train Custom Dataset with PyTorch and Vision Transformers!

# train
python train_cifar10_custom.py --net vit_small --n_epochs 200 --custom
python train_cifar10_custom.py --net vit_tiny --n_epochs 200 --custom
python train_cifar10_custom.py --net vit_nano --n_epochs 200 --custom

# export onnx
python train_cifar10_custom.py \
--net vit_nano \
--ckpt_path checkpoint/vit_nano-4-ckpt-99-202.t7 \
--export_onnx \
--custom

# optimization
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 1
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 2
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 3
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 4
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 5
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 6
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 7
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 8
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 9
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 10
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 11
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 12
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 13
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 14
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 15
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 16
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 17
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 18
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 19
sit4onnx -if vit_nano_Nx3x32x32.onnx -b 20
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

from torch.utils.data import Dataset
from PIL import Image

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10-Custom Training')
parser.add_argument('--custom', action='store_true', help='custom dataset training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--export_onnx', action='store_true', help='export onnx')
parser.add_argument('--ckpt_path', default='')
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): データセットのディレクトリへのパス。
            transform (callable, optional): サンプルに適用されるオプショナルな変換。
        """
        self.root_dir = root
        self.transform = transform
        self.images = []
        self.labels = []

        # ディレクトリ内の画像を読み込む
        for label, cls_dir in enumerate(sorted(os.listdir(root))):
            cls_dir_path = os.path.join(root, cls_dir)
            if os.path.isdir(cls_dir_path):
                for img_name in os.listdir(cls_dir_path):
                    self.images.append(os.path.join(cls_dir_path, img_name))
                    self.labels.append(label)  # フォルダ名に基づいてラベルを割り当てる

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # PIL形式で画像を読み込む
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def list_directories(path):
    directories = []  # フォルダ名を保持するためのリスト
    # 指定したパスの内容を走査
    for item in os.listdir(path):
        # フルパスを組み立て
        full_path = os.path.join(path, item)
        # アイテムがディレクトリの場合のみリストに追加
        if os.path.isdir(full_path):
            directories.append(item)
    return sorted(directories)

class FinalModel(nn.Module):
    def __init__(self, model):
        super(FinalModel, self).__init__()
        self.base_model = model

    def forward(self, x):
        eff_out: torch.Tensor = self.base_model(x)
        return torch.argmax(eff_out, axis=1).to(torch.bool)

# take in args
custom: bool = args.custom
export_onnx: bool = args.export_onnx
ckpt_path: str = args.ckpt_path
usewandb = not args.nowandb and not args.export_onnx
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="cifar10-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:
    N = 2
    M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
n_classes = 10
if not custom:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_classes = 10
else:
    trainset = CustomDataset(root='./data/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
    testset = CustomDataset(root='./data/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    classes = tuple(list_directories('./data/train'))
    n_classes = len(classes)

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=n_classes)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = n_classes
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_nano":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 2,
    heads = 4,
    mlp_dim = 128,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=n_classes,
                downscaling_factors=(2,2,2,1))

if export_onnx:
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model'])
    net = FinalModel(net)
    net.cuda()
    net.eval()
    import onnx
    from onnxsim import simplify
    RESOLUTION = [
        [32,32],
    ]
    MODEL = f'vit_nano'
    for H, W in RESOLUTION:
        onnx_file = f"{MODEL}_1x3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cuda()
        torch.onnx.export(
            net,
            args=(x),
            f=onnx_file,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

    onnx_file = f"{MODEL}_Nx3x{H}x{W}.onnx"
    x = torch.randn(1, 3, 32, 32).cuda()
    torch.onnx.export(
        net,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch'},
            'output' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    exit(0)

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    scheduler.step(epoch-1) # step cosine scheduling

    list_loss.append(val_loss)
    list_acc.append(acc)

    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print(list_loss)

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))

