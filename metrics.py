import torch, torchmetrics
import numpy as np

def calcAreaandCentroid(mask):
    area = np.sum(mask/255)
    yC, xC = np.mgrid[:mask.shape[0], :mask.shape[1]]
    centroidX = np.sum(xC * mask) / (area * 255)
    centroidY = np.sum(yC * mask) / (area * 255)
    return area, centroidX, centroidY

def calculateIOU(testDataset, model):
    jaccardIndex = torchmetrics.JaccardIndex(task="multiclass", num_classes=2)

    for image, mask, _ in testDataset:
        prediction = model(image.unsqueeze(0))[0]
        prediction = (prediction > 0.5).int()
        jaccardIndex.update(prediction.unsqueeze(0), mask.unsqueeze(0))
        print()

    meanJI = jaccardIndex.compute()
    print("Mean Jaccard Distance for Test Data is ", meanJI.item())

def calculateTrueAndFalses(predictions, targets):
    predictionsFlat = predictions.reshape(-1)
    targetsFlat = targets.reshape(-1)
    
    TP = torch.sum((predictionsFlat == 1) & (targetsFlat == 1)).float()
    TN = torch.sum((predictionsFlat == 0) & (targetsFlat == 0)).float()
    FP = torch.sum((predictionsFlat == 1) & (targetsFlat == 0)).float()
    FN = torch.sum((predictionsFlat == 0) & (targetsFlat == 1)).float()
    
    return TP, TN, FP, FN

def calculateMetrics(testDataset, model):
    precisionSum = 0
    recallSum = 0
    f1Sum = 0
    accuracySum = 0
    sumJI = 0
    noOfItems = len(testDataset)
    for image, mask, _ in testDataset:
        prediction = model(image.unsqueeze(0))[0]
        prediction = (prediction > 0.5).int()

        TP,TN,FP,FN = calculateTrueAndFalses(prediction, mask) 

        JI = TP/(TP+FP+FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if precision == 0 or recall == 0:
            accuracy = 0
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = (TP + TN) / (TP + TN + FP + FN)

        precisionSum += precision
        recallSum += recall
        f1Sum += f1
        accuracySum += accuracy
        sumJI += JI

    
    print(f"Precision: {precisionSum/noOfItems}, Recall: {recallSum/noOfItems}, F1 Score: {f1Sum/noOfItems} & Accuracy {accuracySum/noOfItems}, Mean JI {sumJI/noOfItems}")