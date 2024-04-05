import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

class CustomDataset(Dataset):
    def __init__(self, imagesDirectory, masksDirectory, transform=None):
        self.imagesDirectory = imagesDirectory
        self.masksDirectory = masksDirectory
        self.transform = transform
        self.imagesPath = [os.path.join(imagesDirectory, filename) for filename in os.listdir(imagesDirectory)]

        print(os.listdir(imagesDirectory))

    def __len__(self):
        return len(self.imagesPath)

    def __getitem__(self, idx):
        imageName = self.imagesPath[idx]
        image = Image.open(imageName)
        maskName = os.path.join(self.masksDirectory, os.path.basename(imageName))
        mask = Image.open(maskName)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class ConvolutionBlock(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.convOne = nn.Conv2d(inC, outC, kernel_size=3, padding=1)
        self.batchNormOne = nn.BatchNorm2d(outC)
        self.convTwo = nn.Conv2d(outC, outC, kernel_size=3, padding=1)
        self.batchNormTwo = nn.BatchNorm2d(outC)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.convOne(inputs)
        x = self.batchNormOne(x)
        x = self.relu(x)
        x = self.convTwo(x)
        x = self.batchNormTwo(x)
        x = self.relu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.conv = ConvolutionBlock(inC, outC)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.up = nn.ConvTranspose2d(inC, outC, kernel_size=2, stride=2, padding=0)
        self.conv = ConvolutionBlock(outC+outC, outC)


    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoderOne = EncoderBlock(3, 64)
        self.encoderTwo = EncoderBlock(64, 128)
        self.encoderThree = EncoderBlock(128, 256)
        self.encoderFour = EncoderBlock(256, 512)

        self.bottleNeck = ConvolutionBlock(512, 1024)

        self.decoderOne = DecoderBlock(1024, 512)
        self.decoderTwo = DecoderBlock(512, 256)
        self.decoderThree = DecoderBlock(256, 128)
        self.decoderFour = DecoderBlock(128, 64)

        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ Encoder """
        sConvOne, poolOne = self.encoderOne(inputs)
        sConvTwo, poolTwo = self.encoderTwo(poolOne)
        sConvThree, poolThree = self.encoderThree(poolTwo)
        sConvFour, poolFour = self.encoderFour(poolThree)

        """ Bottleneck """
        bottleNeck = self.bottleNeck(poolFour)

        """ Decoder """
        decoderOne = self.decoderOne(bottleNeck, sConvFour)
        decoderTwo = self.decoderTwo(decoderOne, sConvThree)
        decoderThree = self.decoderThree(decoderTwo, sConvTwo)
        decoderFour = self.decoderFour(decoderThree, sConvOne)
        
        """ Classifier """
        outputs = self.outputs(decoderFour)
        outputs = torch.sigmoid(outputs)

        return outputs
    
imagesDirectory = './images'
masksDirectory = './masks'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CustomDataset(imagesDirectory, masksDirectory, transform=transform)
train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

def train(epoch):
    model.train()
    for epoch in range(1, epoch + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

    filePath = './modelweights/torchmodel.pth'
    torch.save(model.state_dict(), filePath)

def loadandtest():
    filePath = './modelweights/torchmodel.pth'
    model.load_state_dict(torch.load(filePath))
    model.eval()

    test_dataset = CustomDataset(imagesDirectory, masksDirectory, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    n_examples = 2

    fig, axs = plt.subplots(n_examples, 3, figsize=(14, n_examples*7), constrained_layout=True)
    for ax, ele in zip(axs, test_loader):
        images, masks = ele
        images, masks = images.to(device), masks.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            outputs = torch.where(outputs > 0.45, 255, 0)  # Assuming it's a segmentation model
            
        for i in range(images.size(0)):
            ax[0].set_title('Original image')
            ax[0].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
            
            ax[1].set_title('Original mask')
            ax[1].imshow(np.transpose(masks[i].cpu().numpy(), (1,2,0)), cmap='gray')
            
            ax[2].set_title('Predicted mask')
            ax[2].imshow(np.transpose(outputs[i].cpu().numpy(), (1,2,0)), cmap='gray')
        
    plt.show()

    

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train(6)
        elif sys.argv[1] == 'test':
            loadandtest()
        else:
            print("Usage: python model.py train/test")
    else:
        print("Usage: python model.py train/test")