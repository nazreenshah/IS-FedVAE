# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
#import plotly.express as px
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from numpy import savetxt, loadtxt
import shutil

import pickle as pkl
import matplotlib.pyplot as plt

import os

import gc
import sys

from pl_bolts.models.autoencoders import VAE

import wandb
import random


# sys.path.append(os.path.abspath('util/'))
from DirichletData import clientDataset, load_data

# %%
# random_seed = 0
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# %%
# set number of clients 
NUM_CLIENTS=10

# set the dirichlet parameter
dirichlet_alpha = 0.001

# set the learning parameters
lr = 1e-2 #1e-3,1e-1,1e-2
num_rounds = 100
batch_size = 256
num_epochs = 1
verbose = False

# set the device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

dtype = torch.float32
torch.set_default_dtype(dtype)

# Linear Probe vs Semi-Supervised Learning
SSL = False
CL_TAG = "SSL" if SSL else "LP"

#pretraining
pretrain=True
PRETRAIN_TAG = f"pretrain={pretrain}"

# loading and saving model
LOAD_CLIENT = True
LOAD_SERVER = True
SAVE_MODEL = True


# personalised vs generalised FL hyperparameter
beta = 0.5


dataset_name = "CIFAR100"
if dataset_name=="CIFAR100":
    class_numbers=100
else :
    class_numbers=10
    
    
res_size = 64        # ResNet image size



# setting path for the results
path = f'results_{dataset_name}/clients={NUM_CLIENTS}/alpha={dirichlet_alpha}/epochs={num_epochs}/rounds={num_rounds}/beta={beta}/{CL_TAG}/{PRETRAIN_TAG}/'

if os.path.exists(path + 'images/'):
    shutil.rmtree(path + 'images/')
if os.path.exists(path + 'models/'):
    shutil.rmtree(path + 'models/')

os.makedirs(path + 'images/')
os.makedirs(path + 'models/')

# loading client train and test datasets
client_trainloaders = []
client_testloaders = []

for i in range(NUM_CLIENTS):
    trainloader, testloader = load_data(n=100, client_id=i, n_clients=NUM_CLIENTS, alpha=dirichlet_alpha, dataset_name=dataset_name, batch_size=batch_size, res_size=res_size)
    client_trainloaders.append(trainloader)
    client_testloaders.append(testloader)
num_batches = min([len(client_trainloaders[i]) for i in range(NUM_CLIENTS)])

print(device)
    
# loading pretrained client vae models 
models_client = [torch.load("pretrained_vae_models/pretrain_"+str(dataset_name)+"_VAE.pth",map_location=device) for i in range(NUM_CLIENTS)]


models_client[0]


# define optimizer for the backbone client vae model pretraining
optimizers = [torch.optim.Adam(models_client[i].parameters(), lr=lr) for i in range(NUM_CLIENTS)]


#classifier model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, class_numbers)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)



classifier_train_losses =[[0] * num_rounds for i in range(NUM_CLIENTS)]
classifier_test_losses = [[0] * num_rounds for i in range(NUM_CLIENTS)]

classifier_train_acc = [[0] * num_rounds for i in range(NUM_CLIENTS)]
classifier_test_acc = [[0] * num_rounds for i in range(NUM_CLIENTS)]

#classifier hyperparameters
learning_rate = 0.1 #0.1,0.01
momentum = 0.9 #0.9,0.5

#loading client classifier models
classifier_networks = [Net().to(device) for i in range(NUM_CLIENTS)]


#define optimizers for the classifier model
classifier_optimizers = []
for i in range(NUM_CLIENTS):
    if SSL:
        params_to_optimize = [
            {'params': models_client[i].encoder.parameters()},
            {'params': models_client[i].fc_mu.parameters()},
            {'params': models_client[i].fc_var.parameters()},
            {'params': classifier_networks[i].parameters()}
        ]
    else:
        params_to_optimize = [
            {'params': classifier_networks[i].parameters()}
        ]
    classifier_optimizers.append(optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum))



def load_dataset(n, batch_size=64, dataset_name="MNIST"):
    '''
    Each client VAE take n% global dataset
    '''
    data_dir = 'data'
    # Load dataset
    if(dataset_name=="MNIST"):
        transform =transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(res_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.MNIST(f"{data_dir}/dataset/MNIST", train=True, download=False, transform=transform)
        testset = torchvision.datasets.MNIST(f"{data_dir}/dataset/MNIST", train=False, download=False, transform=transform)
    elif(dataset_name=="CIFAR10"):
        transform =transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(res_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(f"{data_dir}/dataset/CIFAR10", train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(f"{data_dir}/dataset/CIFAR10", train=False, download=False, transform=transform)
    elif(dataset_name=="CIFAR100"):
        transform =transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(res_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(f"{data_dir}/dataset/CIFAR100", train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR100(f"{data_dir}/dataset/CIFAR100", train=False, download=False, transform=transform)
    else:
        raise Exception("Dataset not recognized")

    
    values = np.arange(len(trainset))
    train_size = len(trainset) * n // 100
    samples = np.random.choice(values, size=train_size, replace=False)
    trainset = torch.utils.data.Subset(trainset, samples)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    values = np.arange(len(testset))
    test_size = len(testset) * n // 100
    samples = np.random.choice(values, size=test_size, replace=False)
    testset = torch.utils.data.Subset(testset, samples)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    return trainset, trainloader, testset, testloader


train_client_loss = []

for i in range(NUM_CLIENTS):
    train_client_loss.append([])


# Function to generate embeddings at clients and save them.
def client_encoder_encoding():
    print("Client_Encoding")

    for current_client in range(NUM_CLIENTS):
        print('Client %d/%d' % (current_client + 1, NUM_CLIENTS))
        for i,data in enumerate(client_trainloaders[current_client]):
            X=data[0].to(device)

            model = models_client[current_client]        

            x = model.encoder(X)

            encoded_mu = model.fc_mu(x).detach().cpu()
            encoded_logvar = model.fc_var(x)
            encoded_var = torch.exp(encoded_logvar).detach().cpu()

            savetxt(f'cache/client{current_client}_data{i}_encoded_mu.csv', np.array(encoded_mu), delimiter=',')
            savetxt(f'cache/client{current_client}_data{i}_encoded_var.csv', np.array(encoded_var), delimiter=',')
            np.save(f'cache/client{current_client}_data{i}_X', np.array(X.detach().cpu()))

            del encoded_mu
            del encoded_logvar
            del encoded_var
            del X
        
        
#Function to train the client classifiers
def classifier_training(round):
    for current_client in range(NUM_CLIENTS):
        model = models_client[current_client].train()
        classifier_network = classifier_networks[current_client].train()
        correct = 0
        total = 0
        train_loss_per_epoch = []
        train_loss_per_round = []
        criterion = nn.CrossEntropyLoss()
        print('Client %d/%d' % (current_client + 1, NUM_CLIENTS)) 
        for epoch in range(num_epochs): 
          
            print(f'epoch no:{epoch+1}/{num_epochs}')   
            for batch_idx, (data, target) in enumerate(trainloader):
                data = data.to(device)
                target = target.to(device)

                x = model.encoder(data)

                
                mu = model.fc_mu(x)
                logvar = model.fc_var(x)

                _, _, z = model.sample(mu, logvar) 
                output = classifier_network(z)

                classifier_optimizers[current_client].zero_grad()

                loss = F.nll_loss(output, target)
                loss.backward()
                
                classifier_optimizers[current_client].step()

                train_loss_per_epoch.append(loss.item())
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                total += len(target)

            train_loss_per_round.append(np.mean(train_loss_per_epoch))
        train_loss = np.max(train_loss_per_round)
        classifier_train_losses[current_client][round] = train_loss
        # wandb.log({"Client_classifier_train_loss"+str(current_client):train_loss})
            
        acc = 100. * correct / total
        classifier_train_acc[current_client][round] = acc
        # wandb.log({"Client_classifier_train_accuracy"+str(current_client):acc})

    print(f'train_loss: {train_loss}')
    print(f'train_acc: {acc}')


# Function to test the client classifiers.
def classifier_testing(round):
    for current_client in range(NUM_CLIENTS):
        model = models_client[current_client].eval()
        classifier_network = classifier_networks[current_client].eval()
        print('Client %d/%d' % (current_client + 1, NUM_CLIENTS)) 
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            test_loss_per_round = []
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(device)
                target = target.to(device)

                x = model.encoder(data)

                mu = model.fc_mu(x)
                logvar = model.fc_var(x)

                _, _, z = model.sample(mu, logvar)
                output = classifier_network(z)
            
                
                test_loss_per_round.append(F.nll_loss(output, target).item())
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                total += len(target)

            test_loss = np.mean(test_loss_per_round)
            classifier_test_losses[current_client][round] = test_loss
            # wandb.log({"Client_classifier_test_loss"+str(current_client):test_loss})

            acc = 100. * correct / total
            classifier_test_acc[current_client][round] = acc
            # wandb.log({"Client_classifier_test_accuracy"+str(current_client):acc})
    print(f'test_loss: {test_loss}')
    print(f'test_acc: {acc}')

#Functions to combine the latent distributions from clients
def combine(mus, vars):
    mu = np.sum(mus, axis=0) / NUM_CLIENTS
    var = 0
    for i in range(len(mus)):
        var += mus[i] ** 2
    var /= NUM_CLIENTS
    var += np.sum(vars, axis=0) / NUM_CLIENTS - mu ** 2
    return torch.tensor(mu), torch.tensor(var)  

def combine_saved_client_encodings():
    combined_latent_space = []
    for i in range(num_batches):
        mus = [loadtxt(f'cache/client{current_client}_data{i}_encoded_mu.csv', delimiter=',') for current_client in range(NUM_CLIENTS)]
        vars = [loadtxt(f'cache/client{current_client}_data{i}_encoded_var.csv', delimiter=',') for current_client in range(NUM_CLIENTS)]
        mu, var = combine(mus, vars)
        combined_latent_space.append((mu, var))
    return combined_latent_space

#Function tosample from the distribution
def model_sample_var(mu, var):
    std = var**0.5
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return p, q, z


#Function to train the client VAE models
def saved_decoded_client_function(combined_latent_space, current_client, round): 
    train_loss_per_epoch=[]
    train_recon_per_epoch=[]
    train_kl1_per_epoch=[]
    
    model = models_client[current_client].train()

    print("Client Training")
    
    for epoch in range(num_epochs): 
          
        print(f'epoch no:{epoch+1}/{num_epochs}')   
        train_loss_per_batch=[]
        train_recon_per_batch=[]
        train_kl1_per_batch=[]
        
        
        for i in range(num_batches):
            mu_client = loadtxt(f'cache/client{current_client}_data{i}_encoded_mu.csv', delimiter=',')
            var_client = loadtxt(f'cache/client{current_client}_data{i}_encoded_var.csv', delimiter=',')

            mu_client = torch.tensor(mu_client).to(device, dtype=dtype)
            var_client = torch.tensor(var_client).to(device, dtype=dtype)
            
            mu_server, var_server = combined_latent_space[i]
            
            mu_server = torch.tensor(mu_server).to(device, dtype=dtype)
            var_server = torch.tensor(var_server).to(device, dtype=dtype)
            
            n01, dist_client, z_k = model_sample_var(mu_client, var_client) 
            n01, dist_server, z = model_sample_var(mu_server, var_server)
    
            z_eff = beta * z_k + (1-beta) * z
            x_hat = model.decoder(z_eff)
            
            x = np.load(f'cache/client{current_client}_data{i}_X.npy')
            x = torch.tensor(x).to(device, dtype=dtype)
            x.requires_grad = True   

            optimizers[current_client].zero_grad()  
            
            #Calculate the importance sampling weight 
            factor = (((var_client / var_server) ** 0.5) * torch.exp((-(z-mu_server)**2/(2 * var_server)) + (-(z_k-mu_client)**2/(2 * var_client)))).mean()
  
            recon_loss = factor*F.mse_loss(x_hat, x, reduction="mean")
            kl_1 = model.kl_coeff * torch.distributions.kl_divergence(dist_server, n01).mean()
           
            loss = recon_loss + kl_1 

            torch.autograd.set_detect_anomaly = True
            loss.backward(retain_graph=True)

            optimizers[current_client].step() 
            loss=loss.cpu().detach().numpy()
            train_loss_per_batch.append(loss)
            train_recon_per_batch.append(recon_loss.item())
            train_kl1_per_batch.append(kl_1.item())
            

        train_loss_per_epoch.append(np.mean(train_loss_per_batch))
        train_recon_per_epoch.append(np.mean(train_recon_per_batch))
        train_kl1_per_epoch.append(np.mean(train_kl1_per_batch))
        
    save_image(x.cpu(), path + f"images/client{current_client+1}_input.jpg")
    save_image(x_hat.cpu(), path + f"images/client{current_client+1}_output.jpg")
    

    if current_client == 0 and round%10 == 0:
        save_image(x.cpu(), path + f"images/client{current_client+1}_input_round{round}.jpg")
        save_image(x_hat.cpu(), path + f"images/client{current_client+1}_output_round{round}.jpg")

    return np.mean(train_loss_per_epoch), np.mean(train_recon_per_epoch), np.mean(train_kl1_per_epoch)

# wandb.init(
   
# )

# Main 
for round in range(num_rounds):
    print('Round %d/%d' % (round + 1, num_rounds))
    if os.path.exists('cache'):
        shutil.rmtree('cache')
    os.makedirs('cache')

    client_encoder_encoding()
    torch.cuda.empty_cache()

    combined_encoded_client = combine_saved_client_encodings()
    torch.cuda.empty_cache()
 
    for current_client in range(NUM_CLIENTS):
        print('Client %d/%d' % (current_client + 1, NUM_CLIENTS))                                                                                                                                                    
        train_loss_avg,train_recon_avg,train_kl_avg=saved_decoded_client_function(combined_encoded_client, current_client, round)
        # wandb.log({"Client_VAE_Loss"+str(current_client): train_loss_avg})
        # wandb.log({"Client_reconstruction_Loss"+str(current_client): train_recon_avg})
        # wandb.log({"Client_KL_Loss"+str(current_client): train_kl_avg})
        print(f"train_loss_client{current_client+1}: {train_loss_avg}")       
        train_client_loss[current_client].append(train_loss_avg)
        torch.cuda.empty_cache()
    print("Classifier Testing")
    classifier_testing(round)
    torch.cuda.empty_cache()
    print("Classifier Training")
    classifier_training(round)
    torch.cuda.empty_cache()
    

    shutil.rmtree('cache')
    
    if round%5==0:
        model_path = path + 'models/'
        if SAVE_MODEL:
            for current_client in range(NUM_CLIENTS):
                torch.save(models_client[current_client], model_path + f'client{current_client+1}_VAE.pth')
                torch.save(classifier_networks[current_client], model_path + f'client{current_client+1}_classifier.pth')
        plots = {}

        for i in range(NUM_CLIENTS):
            plots[f'FL_train_loss_client{i}'] = train_client_loss[i]
            plots[f'classifier_train_loss_client{i}'] = classifier_train_losses[i]
            plots[f'classifier_test_loss_client{i}'] = classifier_test_losses[i]
            plots[f'classifier_train_acc_client{i}'] = torch.tensor(classifier_train_acc[i]).detach().cpu()
            plots[f'classifier_test_acc_client{i}'] = torch.tensor(classifier_test_acc[i]).detach().cpu()
            
        plots_1=pd.DataFrame(dict([(key, pd.Series(value)) for key, value in plots.items()]))
        plots_1.replace(np.nan, 0, inplace=True)
        plots_1.to_csv(path + f'plots.csv')
            

# wandb.finish()

plots = {}

for i in range(NUM_CLIENTS):
    plots[f'FL_train_loss_client{i}'] = train_client_loss[i]
    plots[f'classifier_train_loss_client{i}'] = classifier_train_losses[i]
    plots[f'classifier_test_loss_client{i}'] = classifier_test_losses[i]
    plots[f'classifier_train_acc_client{i}'] = torch.tensor(classifier_train_acc[i]).detach().cpu()
    plots[f'classifier_test_acc_client{i}'] = torch.tensor(classifier_test_acc[i]).detach().cpu()
    

pd.DataFrame(plots).to_csv(path + f'plots.csv')

model_path = path + 'models/'
if SAVE_MODEL:
    for current_client in range(NUM_CLIENTS):
        torch.save(models_client[current_client], model_path + f'client{current_client+1}_VAE.pth')
        torch.save(classifier_networks[current_client], model_path + f'client{current_client+1}_classifier.pth')
    




