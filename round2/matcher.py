import rclpy
from rclpy.node import Node
import numpy as np
from scipy.signal import correlate2d
from PIL import Image
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data

class MapLocalizer(Node):
    def __init__(self, global_map_path, local_map_path):
        super().__init__('map_localizer')

        self.global_map = self.load_map(global_map_path)
        self.local_map = self.load_map(local_map_path)

        # Publisher for localized pose
        self.pose_publisher = self.create_publisher(PoseStamped, 'localized_pose', 10)

        # Perform localization
        self.localize_robot()

    def load_map(self, map_path):
        """Load a .pgm map as a numpy array."""
        image = Image.open(map_path).convert('L')  # Convert to grayscale
        return np.array(image)

    def localize_robot(self):
        """Localize the robot by matching the local map to the global map."""
        self.get_logger().info("Performing localization...")

        # Use cross-correlation to find the best match
        correlation = correlate2d(self.global_map, self.local_map, mode='valid')
        max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)

        # Extract the position from the correlation result
        localized_x = max_idx[1]
        localized_y = max_idx[0]

        self.get_logger().info(f"Robot localized at: x={localized_x}, y={localized_y}")

        # Publish the pose
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = localized_x
        pose.pose.position.y = localized_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0  # Default orientation (no rotation)

        self.pose_publisher.publish(pose)


def main(args=None):
    rclpy.init(args=args)

    # Replace these with the paths to your maps
    global_map_path = '/path/to/global_map.pgm'
    local_map_path = '/path/to/local_map.pgm'

    localizer = MapLocalizer(global_map_path, local_map_path)

    rclpy.spin(localizer)

    localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
