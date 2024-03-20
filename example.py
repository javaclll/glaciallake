import torch
import numpy
import math
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


exampleImage = cv2.imread('assets/example.png', flags=cv2.IMREAD_GRAYSCALE)
plt.imshow(exampleImage, cmap='gray')

def gaussianKernel(size, sigma = 1):
    size = int(size) // 2
    x, y = numpy.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * numpy.pi * sigma**2)
    g = numpy.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobelFilters():
    Kx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], numpy.float32)
    Ky = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], numpy.float32)

    Ix = ndimage.convolve(exampleImage, Kx)
    Iy = ndimage.convolve(exampleImage, Ky)

    G = numpy.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = numpy.arctan2(Iy, Ix)

    return (G, theta)

def nonMaxSuppression(image, angle):
    M, N = image.shape
    Z = numpy.zeros((M,N), dtype=numpy.int32)
    angle = angle * 180. / numpy.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                if (0<= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]
                
                if (image[i,j] >= q) and (image[i,j] >= r):
                    Z[i,j] = image[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    return Z


def threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = image.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = image.shape
    res = numpy.zeros((M,N), dtype=numpy.int32)

    weak = numpy.int32(10)
    strong = numpy.int32(255)

    strongI, strongJ = numpy.where(image >= highThreshold)
    zerosI, zerosJ = numpy.where(image < lowThreshold)
    weakI, weakJ = numpy.where((image <= highThreshold) & (image >= lowThreshold))

    res[strongI, strongJ] = strong
    res[weakI, weakJ] = weak
    res[zerosI, zerosJ] = 0

    return (res, weak, strong)

def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i,j] == weak):
                try:
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

if __name__ == "__main__":
    image = exampleImage
    gaussian = gaussianKernel(5, 1)
    image = ndimage.convolve(image, gaussian)

    G, theta = sobelFilters()
    G = nonMaxSuppression(G, theta)
    res, weak, strong = threshold(G)
    image = hysteresis(res, weak, strong)
    print("Image after Canny Edge Detection...")
    gaussian = gaussianKernel(5, 1)
    image = ndimage.convolve(image, gaussian)

    G, theta = sobelFilters()
    G = nonMaxSuppression(G, theta)
    res, weak, strong = threshold(G)
    image = hysteresis(res, weak, strong)
    plt.imshow(image, cmap='gray')
    plt.show()

