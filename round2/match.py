import cv2
import os

# Path to the image
image_path = os.path.expanduser('hello/map.pgm')

# Load the image
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw keypoints on the image
output_image = cv2.drawKeypoints(gray_image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# Save and display the result
output_path = os.path.expanduser('./map65_orb_features.pgm')
cv2.imwrite(output_path, output_image)
print(f"ORB features marked and saved to {output_path}")

# Optionally display the result
cv2.imshow("ORB Features", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()