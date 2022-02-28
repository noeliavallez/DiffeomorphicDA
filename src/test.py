from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import scipy.io
import gc
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
# Detect if we have a GPU available
#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.rc('font', size=6)          
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i+0.4, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def initialize_model(model_name, num_classes, feature_extract, use_pretrained, den):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
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


def test_model(model, dataloader):

    with torch.no_grad():

        all_preds = torch.tensor([])
        all_preds = all_preds.to(device)

        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            all_preds = torch.cat(
                (all_preds, preds)
                ,dim=0
            )

    return all_preds


def predict_test(data_dir, model_name, model_file, classes, output_path, batch):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load model
    model_ft, input_size = initialize_model(model_name, len(classes), False, True, 1)

    state_dict = torch.load(model_file, map_location='cpu')
    model_ft.load_state_dict(state_dict)
    model_ft.to(device)
    model_ft.eval()

    # Create dataset
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_set = datasets.ImageFolder(data_dir, data_transforms)
    test_data_loader = torch.utils.data.DataLoader(image_set, batch_size=batch, shuffle=False, num_workers=1)

    confidences = test_model(model_ft, test_data_loader)

    stacked = torch.stack(
    (
        torch.tensor(image_set.targets, dtype=torch.int64).to(device)
        ,confidences.argmax(dim=1)
    )
        ,dim=1
    )

    cmt = torch.zeros(len(classes),len(classes), dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    
    cm = confusion_matrix(image_set.targets, confidences.argmax(dim=1).to('cpu'))

    return cmt, cm, image_set.classes


def save_confmat(cm, cm_path, classes):

    fig = plt.figure(figsize=(15,9))
    fig.tight_layout()
    plot_confusion_matrix(cm, classes)
    plt.savefig(cm_path+'.png', bbox_inches='tight')
    #plt.show()

    scipy.io.savemat(cm_path+'.mat', dict(confidences=cm))










results_folder = "../results"
models_folder = "../models"
datasets_folder = "../datasets"
if not os.path.exists(results_folder+'/ALL'):
    os.makedirs(results_folder+'/ALL')

dataset_names = ["Glomeruli", "Diatoms", "Pollen"]
da_methods = ["ProposedDA", "NoiseDA", 'GAN_DA', "woDA", "TraditionalDA"]
modelos = [ 'alexnet', 'densenet', 'inception', 'resnet', 'squeezenet', 'vgg']

batch_size = 32
cols = ['Fold', 'Model', "TestAcc", "ValAcc"]


for dataset_name in dataset_names:

    for da_method in da_methods:

        data_dir = datasets_folder + "/"+dataset_name+"/dataset"

        classes = sorted(os.listdir(os.path.join(data_dir,os.listdir(data_dir)[0])))
        num_classes = len(classes)
        #print(classes)
        #print(len(classes))

        output_path_base =  results_folder + "/" + dataset_name + "/" + da_method
        modeldir = models_folder + "/" + dataset_name + "/" + da_method

        df = pd.DataFrame(columns = cols)

        for learning_rate in [0.001]:
            output_path = output_path_base + '/' + str(learning_rate)
            for run in range(5):
                for model_name in modelos:

                    model_path = modeldir + '/' + str(run) + '/' + str(learning_rate) + '/' + model_name
                    for f in os.listdir(model_path):
                        if model_name in f and len([i for i, ltr in enumerate(f) if ltr == '_']) == 1:
                            model_file = f
                    model_file = model_path + "/" + model_file
                    print(model_file)
                    
                    cmt, cm, cl = predict_test(data_dir+"/test", model_name, model_file, classes, output_path, batch_size)
                    save_confmat(cm, output_path+'/'+str(run)+'_test_'+model_name+'_confmat', cl)

                    cmt2, cm2, cl2 = predict_test(data_dir+"/val", model_name, model_file, classes, output_path, batch_size)
                    save_confmat(cm2, output_path+'/'+str(run)+'_val_'+model_name+'_confmat', cl2)

                    t = torch.sum(cmt).item()
                    tp = 0
                    for i in range(num_classes):
                        tp += cmt[i,i].item()
                    acc = tp/t
                    #print('Test ConfMat:')
                    #print(cmt)
                    #print('Test:', acc)

                    t2 = torch.sum(cmt2).item()
                    tp2 = 0
                    for i in range(num_classes):
                        tp2 += cmt2[i,i].item()
                    acc2 = tp2/t2
                    #print('Val ConfMat:')
                    #print(cmt2)
                    #print('Val:', acc2)

                    print('Val:', acc2, 'Test:', acc)

                    df.loc[df.shape[0]] = [run, model_name, acc, acc2]

                    torch.cuda.empty_cache()
                    gc.collect()

        df.to_excel(results_folder + '/ALL/' + dataset_name + '_' + da_method +'.xlsx', index=False)

