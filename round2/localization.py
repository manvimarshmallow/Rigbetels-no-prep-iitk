import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from scipy.spatial.distance import euclidean
import random
from rclpy.qos import qos_profile_sensor_data

class MPRTLocalizationBot(Node):
    def __init__(self):
        super().__init__('mprt_localization_bot')

        # Publisher for robot velocity
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for the robot's pose
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/amcl_pose', 10)

        # Subscriber for LiDAR scan data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Subscriber for map data
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Transform broadcaster for localization
        self.tf_broadcaster = TransformBroadcaster(self)

        # Transform listener to get robot's position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Default movement command
        self.velocity_command = Twist()

        # Timer to publish velocity at a fixed rate
        self.create_timer(0.1, self.publish_velocity)

        # Map data
        self.current_map = None
        self.map_resolution = None
        self.map_origin = None

        # MPRT parameters
        self.particles = []
        self.num_particles = 100
        self.robot_position = (0, 0, 0)  # x, y, theta

        self.state = "exploring"  # States: exploring, avoiding_obstacle
        self.get_logger().info("MPRT Localization Bot initialized.")

    def map_callback(self, msg):
        """Callback to process map data."""
        self.current_map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        if not self.particles:
            self.initialize_particles()

    def initialize_particles(self):
        """Initialize particles randomly within the map bounds."""
        height, width = self.current_map.shape
        for _ in range(self.num_particles):
            x = random.uniform(0, width * self.map_resolution) + self.map_origin[0]
            y = random.uniform(0, height * self.map_resolution) + self.map_origin[1]
            theta = random.uniform(-np.pi, np.pi)
            self.particles.append((x, y, theta))

    def scan_callback(self, msg):
        """Callback to process LiDAR data."""
        if self.current_map is not None:
            self.update_particles(msg)
            self.publish_pose()

    def update_particles(self, scan_msg):
        """Update particles based on LiDAR scan data."""
        weights = []
        for particle in self.particles:
            x, y, theta = particle
            weight = self.compute_particle_weight(x, y, theta, scan_msg)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(weights) for _ in weights]

        # Resample particles based on weights
        self.particles = random.choices(self.particles, weights, k=self.num_particles)

        # Update robot position to the mean of particles
        self.robot_position = self.compute_mean_particle_position()

    def compute_particle_weight(self, x, y, theta, scan_msg):
        """Compute weight of a particle based on map and LiDAR data."""
        if not self.is_particle_valid(x, y):
            return 0

        # Simulate scan matching (placeholder for actual computation)
        simulated_scan = self.simulate_scan(x, y, theta)
        return 1.0 / (1.0 + euclidean(scan_msg.ranges, simulated_scan))

    def simulate_scan(self, x, y, theta):
        """Simulate LiDAR scan for a given particle (placeholder)."""
        return [1.0] * 360  # Placeholder: Replace with actual simulation logic

    def is_particle_valid(self, x, y):
        """Check if a particle is within valid map boundaries."""
        map_x = int((x - self.map_origin[0]) / self.map_resolution)
        map_y = int((y - self.map_origin[1]) / self.map_resolution)

        if 0 <= map_x < self.current_map.shape[1] and 0 <= map_y < self.current_map.shape[0]:
            return self.current_map[map_y, map_x] == 0  # Free space
        return False

    def compute_mean_particle_position(self):
        """Compute the mean position and orientation of all particles."""
        x = np.mean([p[0] for p in self.particles])
        y = np.mean([p[1] for p in self.particles])
        theta = np.mean([p[2] for p in self.particles])
        return x, y, theta

    def publish_pose(self):
        """Publish the robot's pose based on particle filter."""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()

        x, y, theta = self.robot_position
        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        pose_msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        self.pose_publisher.publish(pose_msg)

    def publish_velocity(self):
        """Publish the current velocity command."""
        self.cmd_vel_publisher.publish(self.velocity_command)


def main(args=None):
    rclpy.init(args=args)
    node = MPRTLocalizationBot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
