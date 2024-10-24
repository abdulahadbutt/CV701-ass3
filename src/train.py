import torch 
import torchvision
from dataset import ImageWoof
from model import CNN 
import glob 
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from dvclive import Live
import os 
import yaml 
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter



def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu'  
):
    with tqdm(data_loader, unit='batch') as data:
        batch_loss_list = []
        for batch in data:
            data.set_description(f"Epoch {epoch_index}")

            # ? Feeding to CNN
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            
            # ? Getting Loss
            batch_loss = criterion(outputs, labels)
            batch_loss_list.append(batch_loss.item())
            batch_loss.backward()

            # ? Gradient Descent
            optimizer.step()


            data.set_postfix(
                batch_loss=batch_loss.item()
            )

    
    return {
        'epoch_idx': epoch_index,
        'batch_losses': batch_loss_list,
        'epoch_loss': np.mean(batch_loss_list)
    }



def train(
    model: torch.nn.Module,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:torch.device,
    live: Live
):

    

    model.train()

    epoch_statistics_list = []
    for epoch in range(epochs):
        epoch_statistics = train_one_epoch(
            model, optimizer, data_loader, epoch, criterion, device
        )
        
        epoch_statistics_list.append(epoch_statistics)
        live.log_metric('train/loss', epoch_statistics['epoch_loss'], plot=True)
        live.next_step()


    return epoch_statistics


def save_checkpoint(
        model: torch.nn.Module, 
        epoch: int, 
        optimizer: torch.optim, 
        f1_score: int, 
        path: str):
    
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer,
            "f1_score": f1_score,
        },
        path,
    )



params = yaml.safe_load(open('params.yaml'))
IMG_SIZE = params['IMG_SIZE']
IMG_SIZE = int(IMG_SIZE)
ROOT_DIR = params['ROOT_DIR']
BATCH_SIZE = params['BATCH_SIZE']
LEARNING_RATE = params['LEARNING_RATE']
EPOCHS = params['EPOCHS']
OPTIMIZER = params['OPTIMIZER']

MAX_PARAMS = params['MAX_PARAMS']
MAX_EPOCHS = params['MAX_EPOCHS']
assert EPOCHS <= MAX_EPOCHS, "Too many epochs listed"

torch.manual_seed(1)

live = Live('metrics', dvcyaml=False, save_dvc_exp=True)

os.makedirs('models', exist_ok=True)

train_dataset = ImageWoof(
    ROOT_DIR, 
    IMG_SIZE,
    train=True,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
assert model.num_of_params() <= MAX_PARAMS, "Too many network parameters"
model.to(device)


if OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    print('Invalid optimizer listed')
    exit()



criterion = torch.nn.CrossEntropyLoss()
loss_statistics = train(
    model, optimizer, train_dataloader, EPOCHS, criterion, device, live
)


save_checkpoint(model, EPOCHS, optimizer, '00', 'models/last.pth')
