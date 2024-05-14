import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('Patat/Foto.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to Grayscale (OpenCV loads images in BGR format)

# Flatten the image into a 1D array of pixels
pixels = image_gray.reshape((-1, 1))
pixels = np.float32(pixels)

# Perform K-means clustering
num_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# Get the cluster centers and labels
cluster_centers = np.uint8(kmeans.cluster_centers_)
cluster_labels = kmeans.labels_

# Reshape the labels to the original image shape
reshaped_image = cluster_centers[cluster_labels].reshape(image_gray.shape)

# Calculate the size of each cluster
unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

# Filter out clusters smaller than 5 pixels
min_cluster_size = 5
valid_labels = unique_labels[label_counts >= min_cluster_size]

# Create a mask to keep only valid clusters
valid_mask = np.isin(cluster_labels, valid_labels)

# Reshape the valid mask to match the shape of the segmented image
valid_mask_reshaped = valid_mask.reshape(image_gray.shape)

# Apply the mask to the segmented image
segmented_image = reshaped_image.copy()
segmented_image[~valid_mask_reshaped] = 0  # Set non-valid clusters to zero (background)

# Calculate mean intensity values for each cluster
mean_intensities = []
for i in range(num_clusters):
    # Create a mask for the current cluster
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    mask[cluster_labels.reshape(image_gray.shape) == i] = 255

    # Calculate mean intensity within the cluster mask
    mean_intensity = cv2.mean(segmented_image, mask=mask)[0]
    mean_intensities.append(mean_intensity)

# Sort clusters based on mean intensity values (from black to white)
sorted_clusters = sorted(range(num_clusters), key=lambda k: mean_intensities[k])

# Reorder the labels to range from black to white
reordered_labels = np.zeros_like(cluster_labels)
for i, label in enumerate(sorted_clusters):
    reordered_labels[cluster_labels == label] = i

# Reshape the reordered labels to the original image shape
sorted_segmented_image = cluster_centers[reordered_labels].reshape(image_gray.shape)
cv2.imwrite('img/seg_image.png', segmented_image)

print('Image shape:', image_gray.shape)
print('Labels shape:', cluster_labels.shape)
print('Segmented image shape:', sorted_segmented_image.shape)

# Create separate images for each color cluster
cluster_images = []
dot_white_images = []

for i in range(num_clusters):
    # Create a mask for the current cluster
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    mask[reordered_labels.reshape(image_gray.shape) == i] = 255

    print('Mask shape:', mask.shape)

    # Apply the mask to the original image
    clustered_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert the clustered image to grayscale
    clustered_gray = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert to black and white
    _, clustered_bw = cv2.threshold(clustered_gray, 10, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to tidy up the image
    kernel = np.ones((2, 2), np.uint8)
    clustered_bw1 = cv2.erode(clustered_bw, kernel, iterations=1)
    clustered_bw2 = cv2.dilate(clustered_bw1, kernel)

    # Label connected components
    output = cv2.connectedComponentsWithStats(clustered_bw2, connectivity=4)
    (total_labels, label_ids, values, centroids) = output
    # Iterate through each component

    dot_white_image = np.zeros(image_gray.shape, dtype="uint8")

    for ids in range(0, total_labels):  # Skip background (label 0)
        # Get the area of the component
        area = values[ids, cv2.CC_STAT_AREA]

        # If the area is small (up to 5 pixels), change its color
        if area <= 500:
            # Mask the component
            dot_mask = (label_ids == ids).astype("uint8") * 255

            # Change color (e.g., to red) in the color change image
            dot_white_image = cv2.bitwise_or(dot_white_image, dot_mask)

    # clustered_bw[np.where(dot_mask)] = 255
    # clustered_bw = cv2.bitwise_or(clustered_bw, component_mask)

    # Add the images to the list
    cluster_images.append(clustered_bw)
    dot_white_images.append(dot_white_image)

# Display or save the cluster images
for i, clustered_bw2 in enumerate(cluster_images):
    cv2.imwrite(f'img/cluster_{i}.png', clustered_bw2)

# Display or save the cluster images
for i, dot_white_image in enumerate(dot_white_images):
    cv2.imwrite(f'img/color_change_image_{i}.png', dot_white_image)

# Apply edge detection
for i, cluster_image in enumerate(cluster_images):
    # Apply Gaussian smoothing
    # not useful #blurred_image = cv2.GaussianBlur(cluster_image, (5, 5), 1.5, cv2.BORDER_DEFAULT)
    # cv2.imwrite(f'blurred_image_{i}.png', blurred_image)
    edges = cv2.Canny(cluster_image, 10, 255)
    cv2.imwrite(f'img/edges_{i}.png', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
