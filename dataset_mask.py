import cv2
from segmentation.kmeans import findKMeans

for i in range(1, 83):
  image = cv2.imread(f"./images/image_{i}.png")
  masked_image = findKMeans(image)
  cv2.imwrite(f"./masks/image_{i}.png", masked_image)

print("All images processed and saved successfully.")