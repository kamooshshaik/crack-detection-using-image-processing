import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import skeletonize, disk, opening, remove_small_objects
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# Utility function for displaying images using matplotlib
def show_image(img, title='', cmap_type='gray'):
    plt.figure(figsize=(6, 6))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load Image
image_path = 'crack.jpg'
I = cv2.imread(image_path)
if I is None:
    raise FileNotFoundError("Image file not found! Check filename and path.")

# Display original image
show_image(I, 'Original Image', cmap_type=None)

# Convert to grayscale
grayImg = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
show_image(grayImg, 'Grayscale Image')

# Enhance contrast
enhancedImg = cv2.normalize(grayImg, None, 0, 255, cv2.NORM_MINMAX)
show_image(enhancedImg, 'Contrast Enhanced Image')

# Apply Gaussian filter
filteredImg = cv2.GaussianBlur(enhancedImg, (5, 5), 1.5)
show_image(filteredImg, 'Gaussian Filtered Image')

# Adaptive thresholding
binaryImg = cv2.adaptiveThreshold(filteredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
show_image(binaryImg, 'Binary Image')

# Morphological cleaning
se = disk(2)
cleanImg = opening(binaryImg, se)
cleanImg = remove_small_objects(cleanImg.astype(bool), 150)
cleanImg = cleanImg.astype(np.uint8) * 255
show_image(cleanImg, 'Cleaned Crack Image')

# Highlight cracks in dark yellow on original image
highlightedImg = I.copy()
Y, X = np.where(cleanImg == 255)
for k in range(len(Y)):
    highlightedImg[Y[k], X[k]] = [0, 180, 255]  # BGR for dark yellow
show_image(highlightedImg, 'Cracks Highlighted in Dark Yellow', cmap_type=None)

# Crack length measurement using skeleton
skeleton = skeletonize(cleanImg > 0)
length_mm = np.sum(skeleton) * 0.1  # Assuming 0.1 mm per pixel

# Compute crack width
distTrans = distance_transform_edt(cleanImg)
width_values = 2 * distTrans[skeleton]
avg_width_mm = np.mean(width_values[width_values > 0]) * 0.1  # in mm

# Highlight skeleton (width measure points) in red
widthImg = highlightedImg.copy()
width_y, width_x = np.where(skeleton)
for i in range(len(width_y)):
    cv2.circle(widthImg, (width_x[i], width_y[i]), 1, (0, 0, 255), -1)  # red dots
show_image(widthImg, 'Crack Width Points Highlighted (Red)', cmap_type=None)

# Final result with length and average width overlaid
resultImg = widthImg.copy()
text = f"Crack Length: {length_mm:.2f} mm | Avg Width: {avg_width_mm:.2f} mm"
cv2.putText(resultImg, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
show_image(resultImg, 'Final Result', cmap_type=None)

# Print results to console
print("🧾 Crack Detection Results:")
print(text)
