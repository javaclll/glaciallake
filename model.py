import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys

class CustomDataset(Dataset):
    def __init__(self, imagesDirectory, masksDirectory, transform=None, transformImages=None, transformMasks=None):
        self.imagesDirectory = imagesDirectory
        self.masksDirectory = masksDirectory
        self.transformImages = transformImages
        self.transform = transform
        self.transformMasks = transformMasks
        self.imagesPath = [os.path.join(imagesDirectory, filename) for filename in os.listdir(imagesDirectory)]

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
        else:
            if self.transformImages:
                image = self.transformImages(image)

            if self.transformMasks:
                mask = self.transformMasks(mask)


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


class Resize(object):
    def __init__(self, maxsize = 1500):
        assert isinstance(maxsize, int)
        self.maxsize = maxsize

    def __call__(self, data):
        data = np.array(data)
        height, width, _ = data.shape
        scale = (max(height, width) // self.maxsize) + 1
        target_height, target_width = height // scale, width // scale
        data = torch.permute(torch.tensor(data), (2, 0, 1))
    
        if scale != 1:
            data = data.float()
            data = torch.nn.functional.interpolate(data, size=(target_height, target_width), mode='nearest')

        return data
    
class ScaleImages(object):
    def __init__(self, threshold=128):
        assert isinstance(threshold, int)
        self.threshold = threshold

    def __call__(self, data):
        data = data.float() / 255.0
        return data

class ScaleMasks(object):
    def __init__(self, threshold=128):
        assert isinstance(threshold, int)
        self.threshold = threshold

    def __call__(self, data):
        data = np.array(data)
        data = ((data > self.threshold) * 1).astype(np.float32)
        data = torch.tensor(data)
        return data


class PaddingImages(object):
    def __init__(self, padMultiplier=16, offset=0):
        assert isinstance(padMultiplier, int)
        assert isinstance(offset, int)

        self.padMultiplier = padMultiplier
        self.offset = offset

    def __call__(self, data):
        data = np.array(data)
        _, height, width = data.shape
        targetHeight = height + (-height) % self.padMultiplier
        targetWidth = width + (-width) % self.padMultiplier
        padTop = (targetHeight - height) // 2
        padBottom = targetHeight - height - padTop
        padLeft = (targetWidth - width) // 2
        padRight = targetWidth - width - padLeft
        data = torch.tensor(data)

        data = torch.nn.functional.pad(data, (padLeft, padRight, padTop, padBottom))

        return data
    
class PaddingMasks(object):
    def __init__(self, padMultiplier=16, offset=0):
        assert isinstance(padMultiplier, int)
        assert isinstance(offset, int)

        self.padMultiplier = padMultiplier
        self.offset = offset

    def __call__(self, data):
        data = np.array(data)
        height, width = data.shape[-2:]
        targetHeight = height + (-height) % self.padMultiplier
        targetWidth = width + (-width) % self.padMultiplier
        padTop = (targetHeight - height) // 2
        padBottom = targetHeight - height - padTop
        padLeft = (targetWidth - width) // 2
        padRight = targetWidth - width - padLeft

        data = torch.tensor(data)
        data = torch.nn.functional.pad(data.unsqueeze(0).float(), (padLeft, padRight, padTop, padBottom), value=0).squeeze(0).byte()
        return data
    
transformImages = transforms.Compose([
    Resize(1500),
    ScaleMasks(128),
    PaddingImages(16,0),
])

transformMasks = transforms.Compose([
    Resize(1500),
    ScaleMasks(128),
    PaddingMasks(16,0),
])

train_size = 0.8
val_size = 0.1
test_size = 0.1

dataset = CustomDataset(imagesDirectory, masksDirectory, transformImages=transformImages, transformMasks=transformMasks)

trainDataset, tvDataset = train_test_split(dataset, test_size=1-train_size, random_state=42)
testDataset, valDataset = train_test_split(tvDataset, test_size=val_size / (val_size + test_size), random_state=42)

trainLoader = DataLoader(trainDataset, batch_size=5, shuffle=True)
validationLoader = DataLoader(valDataset, batch_size=1, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

def train(epochs):
    for epoch in range(1, epochs + 1):
        model.train()
        trainLoss = 0.0
        with tqdm(trainLoader, unit="batch") as tepoch:
            for images, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                images, masks = images.to(device), masks.to(device)
                masks = masks.float()

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                trainLoss += loss.item() * images.size(0)
                tepoch.set_postfix(loss=loss.item())

        # Validation phase
        model.eval()
        valLoss = 0.0
        with torch.no_grad():
            for imagesVal, masksVal in validationLoader:
                imagesVal, masksVal = imagesVal.to(device), masksVal.to(device)
                masksVal = masksVal.float()
                outputs_val = model(imagesVal)
                loss_val = criterion(outputs_val, masksVal)
                valLoss += loss_val.item() * imagesVal.size(0)

        # Calculate average losses
        trainLoss /= len(trainDataset)
        valLoss /= len(valDataset)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")

        # Save the model after each epoch
        filePath = f'./modelweights/torchmodeltransforms{epoch}.pth'
        torch.save(model.state_dict(), filePath)


def loadandtest():
    filePath = './modelweights/torchmodeltransforms5.pth'
    model.load_state_dict(torch.load(filePath))
    model.eval()
    n_examples = 2

    fig, axs = plt.subplots(n_examples, 3, figsize=(14, n_examples*7), constrained_layout=True)
    for ax, ele in zip(axs, testLoader):
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
            train(5)
        elif sys.argv[1] == 'test':
            loadandtest()
        else:
            print("Usage: python model.py train/test")
    else:
        print("Usage: python model.py train/test")