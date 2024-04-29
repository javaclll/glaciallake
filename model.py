import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import cv2
import csv
from metrics import calculateIOU, calculateMetrics, calcAreaandCentroid

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

        return image, mask, idx

class CustomTestDataSet(Dataset):
    def __init__(self, testDirectory, transformImages=None):
        self.testDirectory = testDirectory
        self.transformImages = transformImages
        self.testImagesPath = [os.path.join(testDirectory, filename) for filename in os.listdir(testDirectory)]

    def __len__(self):
        return len(self.testImagesPath)
    
    def __getitem__(self, idx):
        imageName = self.testImagesPath[idx]
        image = Image.open(imageName)
        if self.transformImages:
            image = self.transformImages(image)
        return image

class CustomCheckDataSet(Dataset):
    def __init__(self, testDirectory, checkName, transformImages=None):
        self.testDirectory = testDirectory
        self.checkName = checkName
        self.transformImages = transformImages
        self.testImagePath = os.path.join(testDirectory, checkName)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        imageName = self.testImagePath
        image = Image.open(imageName)
        if self.transformImages:
            image = self.transformImages(image)
        return image
            
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

        self.bottleNeck = ConvolutionBlock(256, 128)

        self.decoderOne = DecoderBlock(128, 64)
        self.decoderTwo = DecoderBlock(64, 32)
        self.decoderThree = DecoderBlock(32, 3)

        self.outputs = nn.Conv2d(3, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ Encoder """
        sConvOne, poolOne = self.encoderOne(inputs)
        sConvTwo, poolTwo = self.encoderTwo(poolOne)
        sConvThree, poolThree = self.encoderThree(poolTwo)

        """ Bottleneck """
        bottleNeck = self.bottleNeck(poolThree)

        """ Decoder """
        decoderOne = self.decoderOne(bottleNeck, sConvThree)
        decoderTwo = self.decoderTwo(decoderOne, sConvTwo)
        decoderThree = self.decoderThree(decoderTwo, sConvOne)
        
        """ Classifier """
        outputs = self.outputs(decoderThree)
        outputs = torch.sigmoid(outputs)

        return outputs

imagesDirectory = './images'
masksDirectory = './masks'
testDirectory = "./area"

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

def train(epochs, newTrain=(True, 0)):
    epochstart = 0

    if not newTrain[0]:
        filePath = f'./modelweights/torchmodelcentroid{newTrain[1]}.pth'
        model.load_state_dict(torch.load(filePath))
        epochstart = newTrain[1]

    for epoch in range(epochstart + 1, epochs + epochstart + 1):
        model.train()
        trainLoss = 0.0
        with tqdm(trainLoader, unit="batch") as tepoch:
            for images, masks, _ in tepoch:
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
            for imagesVal, masksVal, _ in validationLoader:
                imagesVal, masksVal = imagesVal.to(device), masksVal.to(device)
                masksVal = masksVal.float()
                outputs_val = model(imagesVal)
                loss_val = criterion(outputs_val, masksVal)
                valLoss += loss_val.item() * imagesVal.size(0)

        trainLoss /= len(trainDataset)
        valLoss /= len(valDataset)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")

        filePath = f'./modelweights/torchmodelcentroid{epoch}.pth'
        torch.save(model.state_dict(), filePath)

csvPath = "./glacialscales.csv"

def getScale(index):
    if index < 400:
        index = index % 100
        with open(csvPath, 'r') as file:
            reader = csv.DictReader(file)
            counter = 0
            
            for row in reader:
                counter += 1
                if counter == index:
                    return float(row['Scale'])
    else:
        index = index % 100
        with open(csvPath, 'r') as file:
            reader = csv.DictReader(file)
            counter = 0
            
            for row in reader:
                counter += 1
                if counter == index:
                    return float(row['Scale']) * 0.84

def loadandtest():
    filePath = './modelweights/torchmodelcentroid36.pth'
    model.load_state_dict(torch.load(filePath))
    model.eval()

    calculateIOU(testDataset, model)
    calculateMetrics(testDataset, model)

    nexamples = 2

    print(f"Visualizing the Images and the Predicted Masks for {nexamples} Test Data.")
    fig, axs = plt.subplots(nexamples, 3, figsize=(14, nexamples*7), constrained_layout=True)
    for ax, ele in zip(axs, testLoader):
        images, masks, idx = ele
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

            grayMask = cv2.cvtColor(np.float32(np.transpose(outputs[i].cpu().numpy(), (1, 2, 0))), cv2.COLOR_BGR2GRAY)
            grayMask = np.uint8(grayMask)
            calcArea, calcCentX, calcCentY = calcAreaandCentroid(grayMask)

            contours, _ = cv2.findContours(grayMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scale = getScale(int(idx))

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

            # TODO: Scale the centroidX and centroidY to the original image size, and then add it to the centeral cooridnates of the original image
            centroidX = centroidX 
            centroidY = centroidY

            #TODO: Scale the area to the original image size
            maskArea = maskArea * scale * scale
            calcArea = calcArea * scale * scale
    
            print(f"Centroid and Area for the Mask are {centroidX:.2f}, {centroidY:.2f}, {maskArea:.2f}")
            print(f"Calculated Centroid and Area for the Mask are {calcCentX:.2f}, {calcCentY:.2f}, {calcArea:.2f}")
 

    plt.show()


def checkImages(name = None):
    scale = 1
    if name:
        name += ".png"

        allScales = {
            "chola.png": 4.221832295,
            "digtsho.png": 4.221832295,
            "dudhpokhari.png": 4.221832295,
            "imja.png": 8.443075524,
            "lumdingcho.png": 8.443075524,
            "nupchu.png": 4.221832295,
            "tampokharisabai.png": 4.221832295,
            "thulagidona.png": 8.443075524,
            "tshorolpa.png": 16.88446932,
            "westcham.png": 8.443075524,
        }

        scale = allScales[name]

        checkdataset = CustomCheckDataSet(testDirectory, name, transformImages=transformImages)
        checkLoader = DataLoader(checkdataset, batch_size=1, shuffle=True)
        filePath = './modelweights/torchmodelcentroid5.pth'
        model.load_state_dict(torch.load(filePath))
        model.eval()
        
        # Show the Image
        nexamples = 1
        
        print(f"Visualizing the Images and the Predicted Masks for {nexamples} Test Data for {name} w/ scale {scale}.")
    else:
        checkdataset = CustomTestDataSet(testDirectory, transformImages=transformImages)
        checkLoader = DataLoader(checkdataset, batch_size=1, shuffle=True)
        filePath = './modelweights/torchmodelcentroid36.pth'
        model.load_state_dict(torch.load(filePath))
        model.eval()
        
        # SHow Random Tests Data
        nexamples = 2
        print(f"Visualizing the Images and the Predicted Masks for {nexamples} Test Data w/ scale {scale}.")


    fig, axs = plt.subplots(nexamples, 2, figsize=(14, nexamples*7), constrained_layout=True)
    for ax, ele in zip(axs, checkLoader):
        images = ele
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            outputs = torch.where(outputs > 0.5, 255, 0)
            
        for i in range(images.size(0)):
            # ax.set_title('Glacial Lake Image')
            # ax.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))

            ax.set_title('UNet Predicted Lake mask')
            ax.imshow(np.transpose(outputs[i].cpu().numpy(), (1,2,0)))

            grayMask = cv2.cvtColor(np.float32(np.transpose(outputs[i].cpu().numpy(), (1, 2, 0))), cv2.COLOR_BGR2GRAY)
            grayMask = np.uint8(grayMask)
            calcArea, calcCentX, calcCentY = calcAreaandCentroid(grayMask)

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

            # TODO: Scale the centroidX and centroidY to the original image size, and then add it to the centeral cooridnates of the original image
            centroidX = centroidX 
            centroidY = centroidY

            #TODO: Scale the area to the original image size
            maskArea = maskArea * scale * scale
            calcArea = calcArea * scale * scale
            print(scale)
            print(f"Centroid and Area for the Mask are {centroidX:.2f}, {centroidY:.2f}, {maskArea:.2f}")
            print(f"Calculated Centroid and Area for the Mask are {calcCentX:.2f}, {calcCentY:.2f}, {calcArea:.2f}")
    
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train(10)
        elif sys.argv[1] == 'test':
            loadandtest()
        elif sys.argv[1] == 'check':
                checkImages()
        else:
            print("Usage: python model.py train | test | check <scale> | append <epoch>")
    elif len(sys.argv) == 3:
        if sys.argv[1] == 'append':
            train(10, (False, int(sys.argv[2])))
        elif sys.argv[1] == 'check':
            checkImages(name=sys.argv[2])
        else:
            print("Usage: python model.py train | test | check <scale> | append <epoch>")
    else:
        print("Usage: python model.py train | test | check <scale> | append <epoch>")