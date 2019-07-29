# Imports
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

def data_preparation(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transformation for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = dict()
    train_data  = image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data  = image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data  = image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)


    #Using the image datasets and the transforms, define the dataloaders
    batch_size = 50
    dataloaders = dict()
    trainloader  = dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    validloader  = dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
    testloader  = dataloaders['test']  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)

    return train_data, validation_data, test_data, trainloader, validloader, testloader

# Validation function
def validation(model, testloader, criterion):
    accuracy = 0
    loss = 0

    for i, (inputs, labels) in enumerate(testloader):

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        loss = loss + criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy = accuracy + equality.type(torch.FloatTensor).mean()

    return loss, accuracy

def model_train(args):
    # Step 1 Load the data
    train_data, validation_data, test_data, trainloader, validloader, testloader = data_preparation(args)

    #Step 2 load pretrained models
    arch = args.arch
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        arch = 'vgg16'
        model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 3 Define untrained models
    hidden_units = args.hidden_units
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, hidden_units, bias=True)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    # GPU
    if args.gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    #loss
    criterion = nn.NLLLoss()
    learning_rate = args.lr
    # Optimizer should defined after moving model to GPU
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Step 4 Train the models
    print("Training process initializing .....\n")
    steps=0
    print_at = 50
    epochs=args.epochs
    for e in range(epochs):
        run_loss = 0

        for i, (inputs, labels) in enumerate(trainloader):
            steps = steps +  1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss = run_loss + loss.item()

            if steps % print_at == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {} | ".format(e+1),
                      "Training Loss: {:.4f} | ".format(run_loss/print_at),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

                run_loss = 0
                model.train()

    print("\nTraining process is now complete!!")

    # Step 5 model testing
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct +  (predicted == labels).sum().item()

    print("Accuracy achieved by the network on test images is {:.4f} %".format(100 * correct / total))

    # Step 6 save checkpoint

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
                'model_state': model.state_dict(),
                'criterion_state': criterion.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'classifier': classifier,
                'class_to_idx': model.class_to_idx,
                'epochs': epochs,
                'arch': arch,
                'lr': learning_rate
               }

    torch.save(checkpoint, args.saved_model)

    print("Model successfully trained......END")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Image Classifcation')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: vgg16, vgg13]')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--saved_model' , type=str, default='my_train_checkpoint_cmd.pth', help='path of your saved model')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model_train(args)

if __name__ == "__main__":
    main()
