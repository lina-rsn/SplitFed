#============================================================================
# SplitfedV1 (SFLV1) learning: VGG16 on CIFAR-10
# Adapted from a SplitfedV1 ResNet18/HAM10000 implementation to use CIFAR-10 and VGG16
#
# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP
#
# This program is Version1: Single program simulation
# ============================================================================
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.models as models 
import math
import os.path
from pandas import DataFrame
from pathlib import Path

import random
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

#===================================================================
program = "SFLV1 pretrained VGG16 on CIFAR10"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

if torch.cuda.is_available():
    device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Is the current version of PyTorch built with MPS activated?",torch.backends.mps.is_built())
else:
    device = 'cpu'


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

#===================================================================
# No. of users
num_users = 5
epochs = 200
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.00001
num_classes = 10   # CIFAR-10 has 10 classes (HAM10000 had 7)

WD = "."

INTERMEDIATE_DIR = os.path.join(WD, "intermediate_vgg16")
LABEL_DIR = os.path.join(WD, "labels_vgg16")
Path(INTERMEDIATE_DIR).mkdir(parents=True, exist_ok=True)
Path(LABEL_DIR).mkdir(parents=True, exist_ok=True)
SAVE_INTERMEDIATE = True
IMAGE_DIR = os.path.join(WD, "images_vgg16") 
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)


#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class VGG16_client_side(nn.Module):
    def __init__(self):
        super(VGG16_client_side, self).__init__()
        # here load the pretrained on imagenet vgg16
        # cut the model on the cut layer
        vgg_model = models.vgg16(pretrained = True)
        features = list(vgg_model.features.children())
        self.model = nn.Sequential(*features[:5]) # includes Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d 

    def forward(self, x):
        return self.model(x)


net_glob_client = VGG16_client_side()
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)

net_glob_client.to(device)
print(net_glob_client)

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================


class VGG16_server_side(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_server_side, self).__init__()
        model = models.vgg16(pretrained = True)
        features = list(model.features.children())
        self.features = nn.Sequential(*features[5:])

        '''self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )'''
        self.classifier = nn.Sequential( # smaller classifier for smaller image size
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

net_glob_server = VGG16_server_side(num_classes)


if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs

net_glob_server.to(device)
print(net_glob_server)

#===================================================================================
# For Server Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []


criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr

    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    #---------forward prop-------------
    fx_server = net_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))

        # copy the last trained model in the batch
        w_server = net_server.state_dict()

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:

            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            w_locals_server.append(copy.deepcopy(w_server))

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)
                #print(idx_collect)

        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display

            w_glob_server = FedAvg(w_locals_server)

            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [net_glob_server for i in range(num_users)]

            w_locals_server = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        #---------forward prop-------------
        fx_server = net(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)


        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())


        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                l_epoch_check = False

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if federation is happened----------
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")

                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")

    return

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 256, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 256, shuffle = True)


    def train(self, net, global_epoch):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr)

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)

                # saving the intermediate representations to a directory
                if SAVE_INTERMEDIATE:
                    if (global_epoch+1) % 50 == 0:
                        save_name = f"{global_epoch}_{self.idx}_{batch_idx}.pt"
                        torch.save(client_fx.detach().cpu(), os.path.join(INTERMEDIATE_DIR, save_name))
                        torch.save(labels.detach().cpu(), os.path.join(LABEL_DIR, save_name))
                        torch.save(images.detach().cpu(), os.path.join(IMAGE_DIR, save_name))

                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)

                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()


            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                evaluate_server(fx, labels, self.idx, len_batch, ell)

            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID CIFAR-10 splits will be created based on this
def dataset_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

#=============================================================================
#                         Data loading
#=============================================================================
# CIFAR-10 is downloaded/loaded directly via torchvision 
data_dir = 'data'

#=============================================================================
#                         Data preprocessing
#=============================================================================
# CIFAR-10 images are natively 32x32 (HAM10000 images were resized to 64x64),
# and CIFAR-10's own channel statistics are used instead of the ImageNet
# stats that made sense for the dermatoscopic images.
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       
    transforms.RandomHorizontalFlip(),
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_transforms = transforms.Compose([
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# torchvision.datasets.CIFAR10 already gives us the official train/test split
# and downloads the data automatically, replacing the manual
# train_test_split() over a HAM10000 metadata dataframe.
dataset_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
dataset_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)

#----------------------------------------------------------------
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)


#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []

    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device),global_epoch=iter)
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)


    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)

    # Update client-side global model
    net_glob_client.load_state_dict(w_glob_client)

#===================================================================================

print("Training and Evaluation completed!")

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})
file_name = program+".xlsx"
df.to_excel(file_name, sheet_name= "v1_test", index = False)

#=============================================================================
#                         Program Completed
#=============================================================================