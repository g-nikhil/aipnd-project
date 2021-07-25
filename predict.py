import torch
from torchvision import models

import json
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

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments
    parser.add_argument('--arch', dest = 'arch', type = str, default = 'vgg16', help = 'CNN Model Architecture to use', choices = supported_arch_list)
    parser.add_argument('--image_path', dest = 'image_path', type = str, default = 'flowers/test/28/image_05270.jpg', help = 'path to Sample Image')
    parser.add_argument('--save_path', dest = 'save_path', type = str, default = 'checkpoint.pth', help = 'location to save checkpoint')
    parser.add_argument('--gpu', dest='gpu', type=bool, default = False, help = 'Is Run on GPU')
    parser.add_argument('--topk', dest='topk', type=int, default = 5, help = 'Number of top categories to return')
    return parser.parse_args()
# get_input_args

def check_input_args(in_args):
    if not path.exists(in_args.image_path):
        print('--Input Error: image_path doesnot exist--');
        sys.exit()
    if not path.exists(in_args.save_path):
        print('--Input Error: save_path does not exist--');
        sys.exit()
    if in_args.arch not in supported_arch_list:
        print('--Input Error: given arch is not supported--');
        sys.exit()
# check_input_args

def load_checkpoint(filepath, gpu, arch):
    if gpu and torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
# load_checkpoint

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    x, y = image.size
    
    x = x if x <= y else y # shorter side
    y = y if x <= y else x # longer side
    
    # reduce x -> 256 px then y -> ? 
    new_x, new_y = 256, 256*y//x    
    resized_img = image.resize((new_x, new_y))
    
    (left, upper) = ((new_x - 224)/2, (new_y - 224)/2)
    edges = ( left, upper, left + 224, upper + 224) # (left, upper, right, lower)
    crop_img = resized_img.crop(edges)

    np_img = np.array(crop_img)
    np_img = np_img/255
    normalized_img = (np_img - mean) / std
    
    # (dim_x, dim_y, dim_color) --> (dim_color, dim_x, dim_y)
    color_normalized_img = normalized_img.transpose((2,0,1))
    
    return color_normalized_img
# process_image

def get_device(gpu = False):
    if gpu and torch.cuda.is_available:
        print('*** Running on GPU ***')
        return 'cuda'
    else:
        print('*** Running on CPU ***')
        return 'cpu'
# get_device

def predict( model, image_path, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    np_img = process_image(Image.open(image_path))
    # pytorch expects a float tensor with one dimension
    img = torch.FloatTensor(np_img).unsqueeze(0)
    
    model.eval()
    device = get_device(gpu)
    model, img = model.to(device), img.to(device)
    
    logps = model.forward(img)
    top_ps, top_idx = torch.topk(logps, topk)
    ps = torch.exp(top_ps)
    idx_to_class = {model.class_to_idx[k]:k for k in model.class_to_idx}
    category = [ idx_to_class[idx] for idx in top_idx.numpy()[0]]

    return ps.detach().numpy()[0], category
# predict

def main():
    in_args = get_input_args()
    check_input_args(in_args)
    model = load_checkpoint(in_args.save_path, in_args.gpu, in_args.arch)
    ps, category = predict(model, in_args.image_path, in_args.topk, in_args.gpu)
    for op in zip(ps,  [cat_to_name[c] for c in category ]):
        print(op)
# main

# Call to main function to run the program
if __name__ == "__main__":
    main()