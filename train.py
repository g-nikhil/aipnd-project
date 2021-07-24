#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# AIPND-Final Project
#
# PROGRAMMER:   Nikhil G.
# DATE MODIFIED: 07/24/2021
##
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from collections import OrderedDict
import json
if torch.cuda.is_available():
    from workspace_utils import active_session, keep_awake
from PIL import Image
import numpy as np
import argparse
import sys
from os import path

supported_arch_list = ['vgg16', 'resnet18', 'alexnet']
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
outputs = len(cat_to_name)

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments
    parser.add_argument('--epochs', dest = 'epochs', type = int, default = 5, help = 'Number of Epochs for tranining')
    parser.add_argument('--learning_rate', dest = 'lr', type = float, default = 0.001, help = 'Learning Rate')
    parser.add_argument('--arch', dest = 'arch', type = str, default = 'vgg16', help = 'CNN Model Architecture to use', choices = supported_arch_list)
    parser.add_argument('--data_dir', dest = 'data_dir', type = str, default = 'flowers', help = 'Folder with test/train/valid image folders')
    parser.add_argument('--dropout', dest = 'dropout', type=float, default = 0.2, help = 'Hidden Layer drop out probability')
    parser.add_argument('--gpu', dest='gpu', type=bool, default = False, help = 'Is Run on GPU')
    parser.add_argument('--hidden_units', dest= 'hidden_units', type=int, nargs=2, metavar=('hu1', 'hu2'), help='comma separated list of hidden units')
    
    return parser.parse_args()
# get_input_args

def check_input_args(in_args):
    if in_args.hu_divisor <= 1:
        print('--Input Error: hidden_unit_divisor must be greater than 1--');
        sys.exit()
    if in_args.epochs <= 0:
        print('--Input Error: epochs must be greater than 0--');
        sys.exit()
    if in_args.learning_rate <= 0 or in_args.learning_rate> 1:
        print('--Input Error: learning_rate must be between 0 and 1--');
        sys.exit()  
    if in_args.dropout <= 0 or in_args.learning_rate> 1:
        print('--Input Error: dropout must be between 0 and 1--');
        sys.exit()
    if not path.exists(in_args.data_dir):
        print('--Input Error: data_dir doesnot exist--');
        sys.exit()
    if in_args.arch not in supported_arch_list:
        print('--Input Error: given arch is not supported--');
        sys.exit()
# check_input_args

def get_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    normalize_transform = transforms.Normalize(mean, std)

    data_transforms = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.3),
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize_transform
                        ])

    test_transforms = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize_transform
                        ]) 

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    return (train_dataset, valid_dataset, test_dataset), (train_dataloader, valid_dataloader, test_dataloader)
# get_dataloaders

def get_device(gpu = False):
     return 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
# get_device

def build_network(in_args):
    
    if in_args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        inputs = model.fc.in_features
    elif in_args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        inputs = model.classifier[1].in_features
    else:
        model = models.vgg16(pretrained=True)
        inputs = model.classifier[0].in_features
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    model_dict = OrderedDict([
                          ('fc0', nn.Linear(inputs, in_args.hidden_units[0])),
                          ('relu0', nn.ReLU()),
                          ('dropout0', nn.Dropout(p=in_args.dropout))
                    ])   
    for i in range(1, len(in_args.hidden_units)-1):        
        model_dict['fc'+i] = nn.Linear(in_args.hidden_units[i], in_args.hidden_units[i+1]))
        model_dict['relu'+i] = nn.ReLU()
        model_dict['dropout'+i] = nn.Dropout(p=in_args.dropout)
    #    
    model_dict['output'] = nn.Linear(in_args.hidden_units[-1], outputs)    
    model_dict['logsoftmax'] = nn.LogSoftmax(dim=1)
    
    # Build Classifier Network
    classifier = nn.Sequential(model_dict)
    # replace model classifier with our own
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(vgg_model.classifier.parameters(), lr=0.001)
    
    return (model, criterion, optimizer)
# build_network



# Call to main function to run the program
if __name__ == "__main__":
    main()
