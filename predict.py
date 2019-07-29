# Imports here
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable
import seaborn as sns

def load_checkpoint(filename):
    # Load the saved file
    checkpoint = torch.load(filename)

    # Download pretrained model
    model = models.vgg16(pretrained=True);

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])

    return model

def process_image(image):
    img_pil = Image.open(image)

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = adjustments(img_pil)

    return img_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, cat_to_name, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    '''
    #import json
    #with open('cat_to_name.json', 'r') as f:
     #   cat_to_name = json.load(f)
    #evaluate model in cpu
    if gpu:
        model.to('cpu')
    else:
        model.to('cpu')
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities
    prob_log = model.forward(torch_image)

    # linear scale
    prob_linear = torch.exp(prob_log)

    # Find the top 5
    top_probs, top_labels = prob_linear.topk(topk)

    # Get all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_flowers

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Image Classification Prediction')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('--image_path', type=str, default = "flowers/test/10/image_07090.jpg", help='path of image')
    parser.add_argument('--saved_model' , type=str, default='my_train_checkpoint_cmd.pth', help='path of your saved model')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
    args = parser.parse_args()

    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)
    #load checkpoint
    model = load_checkpoint(args.saved_model)
    print("Model loaded")
    #flower_img = imshow(process_image("flowers/test/10/image_07090.jpg"))

    # TODO: Display an image along with the top 5 classes
    # Define image path
    image_path = args.image_path
    print("image path")
    # plot setting
    #plt.figure(figsize = (6,10))
    #ax = plt.subplot(2,1,1)
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    print("processing image")
    # Plot
    #img = process_image(image_path)
    #imshow(img, ax, title = title_);

    print("Starting prediction function")
    # Prediction
    gpu = args.gpu
    probs, flowers = predict(image_path, model, cat_to_name, gpu, topk = args.topk)

    print(flowers)
    print(probs)
    print("Provided image is ... " +
          flowers[0] + " with " + str(probs[0]) + " probability.")

    # Plot bar chart
    #plt.subplot(2,1,2)
    #sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    #plt.show()

    print("End of Image Classification Prediction Model")

if __name__ == "__main__":
    main()
