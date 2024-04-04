import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def longest_subarray(arr):
  lengths = np.array([len(subarr) for subarr in arr])

  # Find the index of the subarray with the largest length
  max_index = np.argmax(lengths)
  return max_index

# Gaussian Kernel
def gaussianKernel(size, sigma = 1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sumWhite(arr):
    # Convert array to boolean array where True represents positions with 0s
    whitePositions = (arr == 255)
    # Sum up the True values
    whiteSum = np.sum(whitePositions)
    return whiteSum

def rgb_to_hsv(r, g, b):
  r /= 255
  g /= 255
  b /= 255
  maxc = max(r, g, b)
  minc = min(r, g, b)
  v = maxc
  if minc == maxc:
      return 0.0, 0.0, v
  s = (maxc-minc) / maxc
  rc = (maxc-r) / (maxc-minc)
  gc = (maxc-g) / (maxc-minc)
  bc = (maxc-b) / (maxc-minc)
  if r == maxc:
      h = 0.0+bc-gc
  elif g == maxc:
      h = 2.0+rc-bc
  else:
      h = 4.0+gc-rc
  h = (h/6.0) % 1.0
  return h * 360, s * 100, v * 100

def findKMeans(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  pixel_values = image.reshape((-1,3))

  pixel_values = np.float32(pixel_values)

  # define stopping criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

  # number of clusters (K)
  k = 5
  _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  # convert back to 8 bit values
  centers = np.uint8(centers)

  # flatten the labels array
  labels = labels.flatten()

  # convert all pixels to the color of the centroids
  segmented_image = centers[labels.flatten()]

  # reshape back to the original image dimension
  segmented_image = segmented_image.reshape(image.shape)

  masked_image = np.copy(image)

  # convert to the shape of a vector of pixel values
  masked_image = masked_image.reshape((-1, 3))

#   k1 = masked_image[labels == 0]
#   k2 = masked_image[labels == 1]
#   k3 = masked_image[labels == 2]
#   k4 = masked_image[labels == 3]
#   k5 = masked_image[labels == 4]

  kall =[]
  
  for i in np.unique(labels):
     kall.append(masked_image[labels == i])

#245optimal
  count_list = []
  for k in kall:
    count = 0
    min_limit = min(200, int(len(k)*0.8))
    for i in range(0, min_limit):
      h,s,v = rgb_to_hsv(*k[i])
      if h>=125 and h<=245:
        count+=1
    count_list.append(count/min_limit)


  # for i in range(0,k):
  #   max = 0
  #   label = -1
  #   if np.count_nonzero(labels==i)>max:
  #     max = np.count_nonzero(labels==i)
  #     label = i

  # color (i.e cluster) to disable

  cluster = np.argmax(count_list)

  masked_image[labels != cluster] = [0, 0, 0]
  masked_image[labels == cluster] = [255, 255, 255]
  # convert back to original shape
  masked_image = masked_image.reshape(image.shape)
  # show the image
  return masked_image