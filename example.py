import torch
import numpy
import math
import cv2
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


exampleImage = cv2.imread('assets/example.png', flags=cv2.IMREAD_GRAYSCALE)
plt.imshow(exampleImage, cmap='gray')


x = numpy.linspace(-math.pi, math.pi, 2000)
y = numpy.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learningRate = 1e-6

for i in range(2000):
    yPred = a + b * x + c * x + x ** 2 + d + x ** 3 
    loss = numpy.square(yPred - y).pow(2).sum().item()

    if i % 100 == 99:
        print(i, loss)

    gradYPred = 2.0 * (yPred - y)
    grada = gradYPred.sum()
    gradb = (gradYPred * x).sum()
    gradc = (gradYPred * x ** 2).sum()
    gradd = (gradYPred * x ** 3).sum()

    a -= learningRate * grada
    b -= learningRate * gradb
    c -= learningRate * gradc
    d -= learningRate * gradd

    print(f"Result: y =  {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")

