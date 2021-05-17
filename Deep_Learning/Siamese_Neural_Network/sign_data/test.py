import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import numpy as np

#load dataset
train_dir = "/content/drive/My Drive/data/sign_data/train"
test_dir = "/content/drive/My Drive/data/sign_data/test"
train_csv = "/content/drive/My Drive/data/sign_data/train_data.csv"
test_csv =  "/content/drive/My Drive/data/sign_data/test_data.csv"

transforms = transforms.Compose([transforms.Resize((100,100)),
                                 transforms.ToTensor()])

class SiameseData(Dataset):
  def __init__(self,train_csv=None,train_dir=None,transform=None):
    self.train_df = pd.read_csv(train_csv)
    self.train_df.coumns = ['image1','image2','image3']
    self.train_dir =  train_dir
    self.transform = transform

  def __getitem__(self,index):
    image1_path = os.path.join(self,train_dir,self.train_df.iat[index,0])
    image2_path = os.path.join(self.train_dir,self.train_df.iat[index,1])
    img1 = io.imread(image1_path)
    img2 = io.imread(image2_path)
    img1 = img1.convert("L")
    img2 = img2.convert("L")
    #apply image transormations
    if self.transform is not None:
      img1 = self.transform(img1) 
      img2 = self.transform(img2)

    return img1,img2,torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
  def __len__(self):
    return len(self.train_df)

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
            
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance((output1, output2),keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

#create a siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
  
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

#set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = SiameseNetwork()
model = model.to(device)

criterion = ContrastiveLoss()

train_ds = SiameseData(train_csv,train_dir,transform=transforms)
test_ds = SiameseData(test_csv,test_dir, transform=transforms)

train_dl = DataLoader(train_ds,shuffle=True,num_workers=8,pin_memory=True,batch_size=32)
test_dl = DataLoader(test_ds,shuffle=False,num_workers=8,pin_memory=True,batch_size=32) 

#train the model
def train(epochs,max_lr,model,train_dl,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    losses = []

    optimizer = opt_func(model.parameters(),max_lr)
    #one cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(1,epochs):
        for batch_idx, (data,targets) in enumerate(train_dl):
            img0, img1 , label = data
            #img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            data.to(device)
            optimizer.zero_grad()
            #forward
            output1,output2 = model(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            losses.append(loss.item())
            #adam step
            optimizer.step()
        print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')  
    return model

epochs = 5
max_lr = 0.01
opt_func = torch.optim.Adam

history = train(epochs,max_lr,model,train_dl,opt_func)