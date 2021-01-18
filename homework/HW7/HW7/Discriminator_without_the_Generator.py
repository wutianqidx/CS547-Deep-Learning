import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__() #Specifying the model parameters 
        # input is 3x32x32
        
        #8 convolution layers with ReLU activation 
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        
        self.conv_layer2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 2, padding = 1) 
        self.conv_layer3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 2, padding = 1) 
        self.conv_layer5 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1)     
        self.conv_layer7 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer8 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 2, padding = 1)  
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        
        #3 Fully connected layers with ReLU activation 
        self.fc1 = nn.Linear(in_features=14*14, out_features= 1) 
        self.fc10 = nn.Linear(in_features=14*14, out_features= 10)
        
        # layer_norm 
        self.layernorm_1 = nn.LayerNorm([196,32,32])
        self.layernorm_2 = nn.LayerNorm([196,16,16])
        self.layernorm_3 = nn.LayerNorm([196,16,16])
        self.layernorm_4 = nn.LayerNorm([196,8,8])
        self.layernorm_5 = nn.LayerNorm([196,8,8])
        self.layernorm_6 = nn.LayerNorm([196,8,8])
        self.layernorm_7 = nn.LayerNorm([196,8,8])
        self.layernorm_8 = nn.LayerNorm([196,4,4])
        self.pool = nn.MaxPool2d(4, stride=4) #Max pooling 

    def forward(self, x, extract_features=0): #Specifying the NN architecture 
        x = F.leaky_relu(self.conv_layer1(x)) #Convolution layers with leaky_relu activation
        x = self.layernorm_1(x)  
        x = F.leaky_relu(self.conv_layer2(x))
        x = self.layernorm_2(x)
        x = F.leaky_relu(self.conv_layer3(x))
        x = self.layernorm_3(x)
        x = F.leaky_relu(self.conv_layer4(x))
        x = self.layernorm_4(x)
        if(extract_features==4):
            h = F.max_pool2d(x,8,8)
            h = h.view(-1, 14*14)
            return h
        x = F.leaky_relu(self.conv_layer5(x))
        x = self.layernorm_5(x)
        x = F.leaky_relu(self.conv_layer6(x))
        x = self.layernorm_6(x)
        x = F.leaky_relu(self.conv_layer7(x))
        x = self.layernorm_7(x)
        x = F.leaky_relu(self.conv_layer8(x))
        x = self.layernorm_8(x)
        if(extract_features==8):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, 14*14)
            return h
        x = self.pool(x)
        x = x.view(-1,14*14)
        x_1 = self.fc1(x)
        x_10 = self.fc10(x)
        return x_1,x_10

def train_test():
    batch_size = 128  #Batch size
    #Defining the data augmentation transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=0.1*torch.randn(1),
                contrast=0.1*torch.randn(1),
                saturation=0.1*torch.randn(1),
                hue=0.1*torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    #Parameters
    model = discriminator()
    #print ('\nModel Architecture is:\n', model)
    model.cuda() #Sending the model to the GPU
    model.train()
    criterion = nn.CrossEntropyLoss()
    LR = 0.0001 #Learning rate
    optimizer = optim.Adam(model.parameters(), lr=LR) #ADAM optimizer 
    train_accuracy = []
    '''
    Train Mode
    Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
    -> Backpropagation -> Optimizer updating the parameters -> Prediction 
    '''
    for epoch in range(100):  # loop over the dataset multiple times
        #Defining the learning rate based on the no. of epochs
        start_time = time.time()
        if(epoch==50):
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR/10.0
        if(epoch==75):
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR/100.0

        running_loss = 0.0
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

            if(Y_train_batch.shape[0] < batch_size):
                continue

            X_train_batch = Variable(X_train_batch).cuda()
            Y_train_batch = Variable(Y_train_batch).cuda()
            optimizer.zero_grad()
            _,output = model(X_train_batch)
            loss = criterion(output, Y_train_batch)
            
            loss.backward()
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000

            optimizer.step()
            prediction = output.data.max(1)[1] #Label Prediction 
            accuracy = (float(prediction.eq(Y_train_batch.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
            train_accuracy.append(accuracy)
        accuracy_epoch = np.mean(train_accuracy)
        print('In epoch ', epoch,' the accuracy of the training set =', accuracy_epoch, 'time = ', time.time()-start_time)
      
    correct = 0
    total = 0
    test_accuracy = []
    model.eval()
    for batch in testloader:
        data, target = batch
        data, target  = Variable(data).cuda(), Variable(target).cuda()
        _,output = model(data)  #Forward propagation     
        prediction = output.data.max(1)[1] #Label Prediction
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0 #Computing the test accuracy
        test_accuracy.append(accuracy)
    accuracy_test2 = np.mean(test_accuracy)
    print('Accuracy on the test set = ', accuracy_test2)
    torch.save(model,'cifar10.model')

if __name__ == "__main__":
    train_test()
