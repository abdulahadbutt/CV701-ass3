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
    model.train()
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


def test_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu' 
):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as data:
            batch_loss_list = []
            for batch in data:
                data.set_description(f"Testing after epoch{epoch_index}")

                # ? Feeding to CNN
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                
                # ? Getting Loss
                batch_loss = criterion(outputs, labels)
                batch_loss_list.append(batch_loss.item())

                data.set_postfix(
                    batch_loss=batch_loss.item()
                )

                _, predictions = outputs.max(1)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)


    
    return {
        'epoch_idx': epoch_index,
        'batch_losses': batch_loss_list,
        'epoch_loss': np.mean(batch_loss_list),
        'accuracy': (num_correct / num_samples).item()
    }



def train(
    model: torch.nn.Module,
    optimizer: torch.optim,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:torch.device,
    live: Live,
    test_dataloader: torch.utils.data.DataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):

    
    best_acc = 0
    train_statistics_list = []
    for epoch in range(epochs):
        # * Training Code
        train_epoch_statistics = train_one_epoch(
            model, optimizer, train_dataloader, epoch, criterion, device
        )
        train_statistics_list.append(train_epoch_statistics)
        live.log_metric('train/loss', train_epoch_statistics['epoch_loss'], plot=True)
        scheduler.step()
        
        # * Testing Code
        test_epoch_statistics = test_one_epoch(
            model, optimizer, test_dataloader, epoch, criterion, device
        )
        latest_test_acc = test_epoch_statistics['accuracy']
        if latest_test_acc > best_acc:
            print(f'UPDATING BEST ACC [{best_acc}] -> [{latest_test_acc}]')
            best_acc = latest_test_acc
            save_checkpoint(model, epoch, optimizer, best_acc, 'models/best_model.pth')
        print(latest_test_acc, type(latest_test_acc))
        live.log_metric('test/loss', test_epoch_statistics['epoch_loss'], plot=True)
        live.log_metric('test/accuracy', test_epoch_statistics['accuracy'], plot=True)



        live.next_step()

    return train_statistics_list


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
SCHEDULER = params['SCHEDULER']

MAX_PARAMS = params['MAX_PARAMS']
MAX_EPOCHS = params['MAX_EPOCHS']
assert EPOCHS <= MAX_EPOCHS, "Too many epochs listed"

torch.manual_seed(1)

live = Live('metrics', dvcyaml=False, save_dvc_exp=True)

os.makedirs('models', exist_ok=True)
os.makedirs('metrics/', exist_ok=True)

train_dataset = ImageWoof(
    ROOT_DIR, 
    IMG_SIZE,
    train=True,
)

test_dataset = ImageWoof(
    ROOT_DIR,
    IMG_SIZE,
    train=False
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
assert model.num_of_params() <= MAX_PARAMS, f"Too many network parameters {model.num_of_params()}"
model.to(device)


if OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
else:
    print('Invalid optimizer listed')
    exit()

if SCHEDULER == 'constant':
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
elif SCHEDULER == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
elif SCHEDULER == 'linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
else:
    scheduler = None 

criterion = torch.nn.CrossEntropyLoss()
loss_statistics = train(
    model, optimizer, train_dataloader, EPOCHS, criterion, device, live, test_dataloader, scheduler
)


save_checkpoint(model, EPOCHS, optimizer, '00', 'models/last.pth')
