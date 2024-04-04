import cv2
from segmentation.kmeans import findKMeans

for i in range(1, 26):
  image = cv2.imread(f"/")
  masked_image = findKMeans(image)

print("All images processed and saved successfully.")