import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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
        
        #3 Fully connected layers with ReLU activation 
        self.fc1 = nn.Linear(in_features=14*14, out_features= 1) 
        self.fc10 = nn.Linear(in_features=14*14, out_features= 10)
        
        self.layernorm_1 = nn.LayerNorm([196,32,32])
        self.layernorm_2 = nn.LayerNorm([196,16,16]) #Batch Normalization
        self.layernorm_3 = nn.LayerNorm([196,16,16])
        self.layernorm_4 = nn.LayerNorm([196,8,8])
        self.layernorm_5 = nn.LayerNorm([196,8,8])
        self.layernorm_6 = nn.LayerNorm([196,8,8])
        self.layernorm_7 = nn.LayerNorm([196,8,8])
        self.layernorm_8 = nn.LayerNorm([196,4,4])
        self.pool = nn.MaxPool2d(4, stride=4) #Max pooling 

    def forward(self, x, extract_features=0): #Specifying the NN architecture 
        x = F.leaky_relu(self.conv_layer1(x)) #Convolution layers with relu activation
        x = self.layernorm_1(x) #Batch normalization 
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

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__() #Specifying the model parameters 
        self.conv_layer1 = nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride = 2, padding = 1) 
        self.conv_layer2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer5 = nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride = 2, padding = 1) 
        self.conv_layer6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1)     
        self.conv_layer7 = nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride = 2, padding = 1) 
        self.conv_layer8 = nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride = 1, padding = 1)  
        
        # Fully connected layers with ReLU activation 
        self.fc1 = nn.Linear(in_features=100, out_features= 196*4*4) 
        
        # Batch_norm
        self.batchnorm_0 = nn.BatchNorm2d(196) 
        self.batchnorm_1 = nn.BatchNorm2d(196)
        self.batchnorm_2 = nn.BatchNorm2d(196) 
        self.batchnorm_3 = nn.BatchNorm2d(196)
        self.batchnorm_4 = nn.BatchNorm2d(196)
        self.batchnorm_5 = nn.BatchNorm2d(196)
        self.batchnorm_6 = nn.BatchNorm2d(196)
        self.batchnorm_7 = nn.BatchNorm2d(196)
        self.pool = nn.MaxPool2d(4, stride=4) #Max pooling 

    def forward(self, x): #Specifying the NN architecture 
        x = F.relu(self.fc1(x))
        x = x.view(-1,196,4,4)
        #x = self.batchnorm_0(x)
        x = F.relu(self.conv_layer1(x)) #Convolution layers with relu activation
        x = self.batchnorm_1(x) #Batch normalization 
        x = F.relu(self.conv_layer2(x))
        x = self.batchnorm_2(x)
        x = F.relu(self.conv_layer3(x))
        x = self.batchnorm_3(x)
        x = F.relu(self.conv_layer4(x))
        x = self.batchnorm_4(x)
        x = F.relu(self.conv_layer5(x))
        x = self.batchnorm_5(x)
        x = F.relu(self.conv_layer6(x))
        x = self.batchnorm_6(x)
        x = F.relu(self.conv_layer7(x))
        x = self.batchnorm_7(x)
        x = F.tanh(self.conv_layer8(x))
        return x

batch_size = 128
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

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

    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if not os.path.exists('output'):
        os.makedirs('output')

    #Parameters
    aD =  discriminator()
    aD.cuda()

    aG = generator()
    aG.cuda()

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))
    criterion = nn.CrossEntropyLoss()

    LR = 0.0001 #Learning rate
    train_accuracy = []

    np.random.seed(352)
    n_z = 100
    n_classes = 10
    label = np.asarray(list(range(10))*10)
    noise = np.random.normal(0,1,(100,n_z))
    label_onehot = np.zeros((100,n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = Variable(save_noise).cuda()

    start_time = time.time()
    gen_train = 1
    num_epochs = 200
    # Train the model
    for epoch in range(0,num_epochs):

        # training
        aG.train()
        aD.train()
        # before epoch training loop starts
        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []
        loss5 = []
        acc1 = []
      
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

            if(Y_train_batch.shape[0] < batch_size):
                continue
                # train G
            if((batch_idx%gen_train)==0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()

                label = np.random.randint(0,n_classes,batch_size)
                noise = np.random.normal(0,1,(batch_size,n_z))
                label_onehot = np.zeros((batch_size,n_classes))
                label_onehot[np.arange(batch_size), label] = 1
                noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = Variable(noise).cuda()
                fake_label = Variable(torch.from_numpy(label)).cuda()

                fake_data = aG(noise)
                gen_source, gen_class  = aD(fake_data)

                gen_source = gen_source.mean()
                gen_class = criterion(gen_class, fake_label)

                gen_cost = -gen_source + gen_class
                gen_cost.backward()
                for group in optimizer_g.param_groups:
                    for p in group['params']:
                        state = optimizer_g.state[p]
                        if('step' in state and state['step']>=1024):
                            state['step'] = 1000
                optimizer_g.step()
            
            # train D  
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()

            # train discriminator with input from generator
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
            with torch.no_grad():
                fake_data = aG(noise)

            disc_fake_source, disc_fake_class = aD(fake_data)

            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)

            # train discriminator with input from the discriminator
            real_data = Variable(X_train_batch).cuda()
            real_label = Variable(Y_train_batch).cuda()

            disc_real_source, disc_real_class = aD(real_data)

            prediction = disc_real_class.data.max(1)[1]
            accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)

            gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()
            for group in optimizer_d.param_groups:
                for p in group['params']:
                    state = optimizer_d.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000
            optimizer_d.step()
          
            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            # within the training loop
            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if((batch_idx%50)==0):
                print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                            "%.2f" % np.mean(loss2), 
                            "%.2f" % np.mean(loss3), 
                            "%.2f" % np.mean(loss4), 
                            "%.2f" % np.mean(loss5), 
                            "%.2f" % np.mean(acc1))

        # Test the model
        aD.eval()
        with torch.no_grad():
            test_accu = []
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
                X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

                with torch.no_grad():
                    _, output = aD(X_test_batch)

                prediction = output.data.max(1)[1] # first column has actual prob.
                accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
                test_accu.append(accuracy)
                accuracy_test = np.mean(test_accu)
        print('Testing',accuracy_test, time.time()-start_time)

        ### save output
        with torch.no_grad():
            aG.eval()
            samples = aG(save_noise)
            samples = samples.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0,2,3,1)
            aG.train()

        fig = plot(samples)
        plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)

        if(((epoch+1)%1)==0):
            torch.save(aG,'tempG.model')
            torch.save(aD,'tempD.model')

    torch.save(aG,'generator.model')
    torch.save(aD,'discriminator.model')

if __name__ == "__main__":
    train_test()
