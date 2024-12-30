import os
import time
import subprocess
import cv2
import numpy as np
from threading import Thread
import pygame

# Initialize Pygame
pygame.init()

# Set the window size to 3 times the original dimensions
original_width, original_height = 400, 300
screen = pygame.display.set_mode((original_width * 3, original_height * 3))
pygame.display.set_caption("Map Matching Visualization (Expanded)")

def save_map(interval=25, save_path="hello/map"):
    """
    Continuously saves the map at specified intervals.

    :param interval: Time interval in seconds.
    :param save_path: Path to save the map file.
    """
    while True:
        try:
            # Delete the previous map file if it exists
            map_file = f"{save_path}.pgm"
            command = ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", save_path]
            subprocess.run(command, check=True)
            print(f"Map saved at {save_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to save the map: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        time.sleep(interval)

def match_and_mark(saved_map_path="hello/map.pgm", reference_map_path="hello/map56.pgm"):
    """
    Matches the saved map with a reference map using feature matching and marks the centroid.

    :param saved_map_path: Path to the recently saved map.
    :param reference_map_path: Path to the reference map.
    :return: Matched image in Pygame format.
    """
    # Load the maps
    saved_map = cv2.imread(saved_map_path, cv2.IMREAD_GRAYSCALE)
    reference_map = cv2.imread(reference_map_path, cv2.IMREAD_GRAYSCALE)

    if saved_map is None or reference_map is None:
        print("Error: Unable to load one or both maps.")
        return None

    # Resize reference_map to 10x the size of saved_map
    height, width = saved_map.shape
    resized_reference_map = cv2.resize(reference_map, (width * 10, height * 10), interpolation=cv2.INTER_LINEAR)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect and compute features
    keypoints1, descriptors1 = orb.detectAndCompute(saved_map, None)
    keypoints2, descriptors2 = orb.detectAndCompute(resized_reference_map, None)

    # Use the BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the coordinates of the matched keypoints
    points = [keypoints2[match.trainIdx].pt for match in matches[:20]]
    if points:
        points = np.array(points, dtype=np.float32)
        centroid_x = int(np.mean(points[:, 0]))
        centroid_y = int(np.mean(points[:, 1]))

        # Draw a green circle at the centroid
        cv2.rectangle(resized_reference_map, (centroid_x - 25, centroid_y - 25), (centroid_x + 25, centroid_y + 25), (0, 255, 0), thickness=-1)

    # Draw the matches
    matched_image = cv2.drawMatches(
        saved_map,
        keypoints1,
        resized_reference_map,
        keypoints2,
        matches[:20],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Convert the OpenCV image to a format compatible with Pygame
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    matched_image_surface = pygame.image.frombuffer(matched_image_rgb.tobytes(), matched_image_rgb.shape[1::-1], "RGB")
    return matched_image_surface

if __name__ == "__main__":
    # Start the map saving thread
    map_saving_thread = Thread(target=save_map, args=(5, "./map"), daemon=True)
    map_saving_thread.start()

    # Allow some time for maps to be saved before matching
    time.sleep(10)

    # Continuously perform map matching and display the results
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Match and display the features
        matched_image_surface = match_and_mark()
        if matched_image_surface:
            # Scale the matched image surface to fit the expanded window
            scaled_surface = pygame.transform.scale(matched_image_surface, (original_width * 3, original_height * 3))
            screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()

        time.sleep(5)

    pygame.quit()
