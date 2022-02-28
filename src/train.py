from __future__ import division

import copy
import gc
import os
import sys
import time
import matplotlib.pyplot as plt

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from random import random


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
# Detect if we have a GPU available
# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101
print("Using " + str(device))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if random()>0.8: 
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def train_model(model, dataloaders, criterion, optimizer, output_path, num_epochs=25, is_inception=False, saveChkpts=False):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    plt.ion()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, num_epochs + 1, 10.0))
    plt.show()

    directory = os.path.join(output_path, model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # pylint: disable=E1101
                    _, preds = torch.max(outputs, 1)
                    # pylint: enable=E1101

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # pylint: disable=E1101
                running_corrects += torch.sum(preds == labels.data)
                # pylint: enable=E1101

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if saveChkpts:
                    model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model_wts, os.path.join(output_path, model_name, 'chkpt_' + str(epoch)))
                if epoch in [30, 40, 50]:
                    model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(output_path, model_name, model_name + '_best_until_' + str(epoch) + '_' + '{:4f}'.format(best_acc)))
            else:
                train_acc_history.append(epoch_acc)

        ohist = [h.cpu().numpy() for h in train_acc_history]
        phist = [h.cpu().numpy() for h in val_acc_history]
        line_up, = plt.plot(range(0, len(ohist)), ohist, label="Train")
        line_down, = plt.plot(range(0, len(phist)), phist, label="Validation")
        plt.legend(handles=[line_up, line_down])
        plt.draw()
        plt.pause(0.0005)
        print()

    torch.save(best_model_wts, os.path.join(output_path, model_name, model_name + '_' + '{:4f}'.format(best_acc)))

    plt.savefig(os.path.join(output_path, model_name, model_name+'_'+str(learning_rate)+'_training.png'))
    plt.clf()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history


def set_parameter_requires_grad(model, feature_extracting, den):

    if feature_extracting and not (den == 1):
        if isinstance(model, models.ResNet) or isinstance(model, models.Inception3):
            tam = len(list(model.parameters()))
            cont = 0
            for param in model.parameters():
                if cont <= tam/den:
                    param.requires_grad = False
                cont = cont+1
        else:
            tam = len(list(model.features.parameters()))
            cont = 0
            for param in model.features.parameters():
                if cont <= tam/den:
                    param.requires_grad = False
                cont = cont+1
    elif feature_extracting and den == 1:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained, den):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, den)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, den)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, den)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, den)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, den)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, den)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def test_model(model, dataloader, classes):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    prob = np.array([])
    pred = np.array([])
    labs = np.array([])
    first = True
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # pylint: disable=E1101
            _, predicted = torch.max(outputs.data, 1)
            # pylint: enable=E1101
            sm = torch.nn.Softmax()
            probabilities = sm(outputs)
            #print(probabilities)  # Converted to probabilities
            if(first):
                prob = probabilities.cpu().numpy()
                labs = labels.cpu().numpy()
                pred = predicted.cpu().numpy()
                first = False
            else:
                prob = np.concatenate((prob, probabilities.cpu().numpy()), axis=0)
                labs = np.concatenate((labs, labels.cpu().numpy()), axis=0)
                pred = np.concatenate((pred, predicted.cpu().numpy()), axis=0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(0, len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    res = np.concatenate(([labs], [pred]), axis=0)
    res = np.concatenate((res.transpose(), prob), axis=1)
    print('Accuracy of the network on the test images: (%d/%d) : %.2f %%' % (correct, total, 100 * correct / total))
    for i in range(num_classes):
        print('%5s : (%d/%d) : %.2f %%' % (classes[i], class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))

    return res


def run_tests(data_dir, classes, num_classes, model_name, batch_size, num_epochs, learning_rate, output_path, feature_extractor, use_pretrained, use_data_augmentation, den=1):

    orig_stdout = sys.stdout
    directory = os.path.join(output_path, model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(os.path.join(output_path, model_name, model_name+'_'+str(learning_rate)+'_results.txt'), 'w')
    sys.stdout = f

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extractor, use_pretrained, den)
    print(model_ft)

    if use_data_augmentation:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                AddGaussianNoise(0.1, 0.08),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    test_data_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=4)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameteFrs
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extractor:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, _, _ = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, output_path, num_epochs=num_epochs, is_inception=(model_name == "inception"))
    model_ft.eval()
    print('\nBEST MODEL ACC:')
    confidences = test_model(model_ft, test_data_loader, classes)

    scipy.io.savemat(os.path.join(output_path, model_name, model_name+'_'+str(learning_rate)+'_confidences.mat'), dict(confidences=confidences))
    print('Confidences in confidences.mat with columns: Label, Pred, ClassConfs...')

    sys.stdout = orig_stdout
    f.close()

    with open(os.path.join(output_path, model_name, model_name+'_'+str(learning_rate)+'_results.txt'), 'r') as fin:
        print(fin.read())









# Models to choose from
modelos = ['resnet', 'alexnet', 'squeezenet', 'inception', 'densenet', 'vgg']
batch_size = 32
num_epochs = 60
use_pretrained = True
feature_extractor = False
learning_rate = 0.001


for dataset_name in ["Diatoms", "Glomeruli", "Pollen"]:
    data_dir = "../datasets/"+dataset_name+"/dataset"
    classes = sorted(os.listdir(os.path.join(data_dir,os.listdir(data_dir)[0])))
    num_classes = len(classes)
    print(classes)
    print(len(classes))

    output_path_base = "../models/"+dataset_name+"/NoiseDA"
    use_data_augmentation = True

    for run in range(5):

        output_path0 = output_path_base + '/' + str(run) + '/'

        # Tests
        output_path = output_path0 + str(learning_rate)
        for model_name in modelos:
            run_tests(data_dir, classes, num_classes, model_name, batch_size, num_epochs, learning_rate, output_path, feature_extractor, use_pretrained, use_data_augmentation)

        gc.collect()
