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
import torchmetrics
import cv2

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

        self.encoderOne = EncoderBlock(3, 32)
        self.encoderTwo = EncoderBlock(32, 64)
        self.encoderThree = EncoderBlock(64, 128)
        self.encoderFour = EncoderBlock(128, 256)

        self.bottleNeck = ConvolutionBlock(256, 512)

        self.decoderOne = DecoderBlock(512, 256)
        self.decoderTwo = DecoderBlock(256, 128)
        self.decoderThree = DecoderBlock(128, 64)
        self.decoderFour = DecoderBlock(64, 32)

        self.outputs = nn.Conv2d(32, 3, kernel_size=1, padding=0)

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
        targetHeight, targetWidth = height // scale, width // scale
        data = torch.permute(torch.tensor(data), (2, 0, 1))
    
        if scale != 1:
            data = data.float()
            data = torch.nn.functional.interpolate(data, size=(targetHeight, targetWidth), mode='nearest')

        return data

class Scale(object):
    def __call__(self, data):
        data = data.float() / 255.0
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
    Scale(),
    PaddingImages(16,0),
])

transformMasks = transforms.Compose([
    Resize(1500),
    Scale(),
    PaddingMasks(16,0),
])

trainSize = 0.8
valSize = 0.1
testSize = 0.1

dataset = CustomDataset(imagesDirectory, masksDirectory, transformImages=transformImages, transformMasks=transformMasks)

trainDataset, tvDataset = train_test_split(dataset, test_size=1-trainSize, random_state=42)
testDataset, valDataset = train_test_split(tvDataset, test_size=valSize / (valSize + testSize), random_state=42)

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

        model.eval()
        valLoss = 0.0
        with torch.no_grad():
            for imagesVal, masksVal in validationLoader:
                imagesVal, masksVal = imagesVal.to(device), masksVal.to(device)
                masksVal = masksVal.float()
                outputs_val = model(imagesVal)
                loss_val = criterion(outputs_val, masksVal)
                valLoss += loss_val.item() * imagesVal.size(0)

        trainLoss /= len(trainDataset)
        valLoss /= len(valDataset)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")

        filePath = f'./modelweights/torchmodeltransforms{epoch}.pth'
        torch.save(model.state_dict(), filePath)


def loadandtest():
    filePath = './modelweights/torchmodeltransforms10.pth'
    model.load_state_dict(torch.load(filePath))
    model.eval()
    n_examples = 2

    fig, axs = plt.subplots(n_examples, 3, figsize=(14, n_examples*7), constrained_layout=True)
    for ax, ele in zip(axs, testLoader):
        images, masks = ele
        images, masks = images.to(device), masks.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            outputs = torch.where(outputs > 0.5, 255, 0)
            
        for i in range(images.size(0)):
            ax[0].set_title('Glacial Lake Image')
            ax[0].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))

            ax[1].set_title('Glacial Lake Mask')
            ax[1].imshow(np.transpose(masks[i].cpu().numpy(), (1, 2, 0)) * 255)
            
            ax[2].set_title('UNet Predicted Lake mask')
            ax[2].imshow(np.transpose(outputs[i].cpu().numpy(), (1,2,0)))

            grayMask = cv2.cvtColor(np.transpose(masks[i].cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2GRAY)

            contours, _ = cv2.findContours(grayMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            maskArea = 0
            centroidX = 0
            centroidY = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                maskArea += area

                centroid = cv2.moments(contour)

                if centroid["m00"] != 0:
                    cX = int(centroid["m10"] / centroid["m00"])
                    cY = int(centroid["m01"] / centroid["m00"])

                    centroidX += cX * area
                    centroidY += cY * area
                
                if area > 0:
                    centroidX = centroidX / area
                    centroidY = centroidY / area

            
            print("Centroid and Area for the Mask are ", centroidX, centroidY, maskArea)

    plt.show()



def calculateIOU():

    jaccardIndex = torchmetrics.JaccardIndex(task="multiclass", num_classes=2)

    for image, mask in testDataset:
        prediction = model(image.unsqueeze(0))[0]
        prediction = (prediction > 0.5).int()
        jaccardIndex.update(prediction.unsqueeze(0), mask.unsqueeze(0))

    meanJI = jaccardIndex.compute()
    print("Mean Jaccard Distance for Test Data is ", meanJI.item())

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train(10)
        elif sys.argv[1] == 'test':
            loadandtest()
            calculateIOU()
        else:
            print("Usage: python model.py train/test")
    else:
        print("Usage: python model.py train/test")