import cv2
import os
import numpy as np

def generateAdditionalImages(image):
    rotatedRight = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    flippedHorizontal = cv2.flip(image, 1)
    rotateLeft = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rows, cols, _ = image.shape
    distortionMatrix = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
    distoredImage = cv2.warpAffine(image, distortionMatrix, (cols, rows))

    return rotatedRight, flippedHorizontal, rotateLeft, distoredImage

imagePath = "./data/images"
maskPath = "./data/masks"

imageFiles = os.listdir(imagePath)
maskFiles = os.listdir(maskPath)

for idx, file in enumerate(imageFiles):
    imagefilePath = os.path.join(imagePath, file)
    image = cv2.imread(imagefilePath)

    maskfilePath = os.path.join(maskPath, file)
    mask = cv2.imread(maskfilePath)

    imrotatedRight, imflippedHorizontal, imrotatedLeft, imdistortedImage = generateAdditionalImages(image)
    msrotatedRight, msflippedHorizontal, msrotatedLeft, msdistortedImage = generateAdditionalImages(mask)

    cv2.imwrite(os.path.join(imagePath, f"image_{idx + 101}.png"), imrotatedRight)
    cv2.imwrite(os.path.join(imagePath, f"image_{idx + 201}.png"), imrotatedLeft)
    cv2.imwrite(os.path.join(imagePath, f"image_{idx + 301}.png"), imflippedHorizontal)
    cv2.imwrite(os.path.join(imagePath, f"image_{idx + 401}.png"), imdistortedImage)

    cv2.imwrite(os.path.join(maskPath, f"image_{idx + 101}.png"), msrotatedRight)
    cv2.imwrite(os.path.join(maskPath, f"image_{idx + 201}.png"), msrotatedLeft)
    cv2.imwrite(os.path.join(maskPath, f"image_{idx + 301}.png"), msflippedHorizontal)
    cv2.imwrite(os.path.join(maskPath, f"image_{idx + 401}.png"), msdistortedImage)

print("Additional images generated successfully!")
