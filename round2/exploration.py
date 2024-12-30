import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
from rclpy.qos import qos_profile_sensor_data
class FrontierExploration(Node):
    def __init__(self):
        super().__init__('frontier_exploration')
        self.map_data = None
        self.scan_data = None
        self.current_goal = None
        # Publishers and Subscribers
        # qos_profile= QoSProfile(depth=10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.map_subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        

        # Internal Variables
        self.rate = self.create_timer(0.3, self.explore_callback)  # 10 Hz loop
        
    def map_callback(self, msg):
        self.map_data = msg
        print("map_call")

    def scan_callback(self, msg):
        self.scan_data = msg
        print("scan_call")

    def find_frontiers(self):
        """Identify frontiers in the occupancy grid."""
        if self.map_data is None:
            print("nomapdata")
            return []

        # Convert OccupancyGrid to a 2D numpy array
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        grid = np.array(self.map_data.data).reshape((height, width))

        # Identify free (0) and unknown (-1) regions
        free_space = (grid == 0)
        unknown_space = (grid == -1)

        # Find the boundary between free and unknown regions
        frontiers = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if free_space[y, x]:
                    # Check if any neighboring cell is unknown
                    if unknown_space[y-1:y+2, x-1:x+2].any():
                        frontier_x = x * resolution + origin_x
                        frontier_y = y * resolution + origin_y
                        frontiers.append((frontier_x, frontier_y))

        return frontiers

    def explore_callback(self):
        """Main loop for exploration."""
        if self.map_data is None or self.scan_data is None:
            self.get_logger().info("Waiting for map and scan data...")
            return

        # Find frontiers
        frontiers = self.find_frontiers()
        if not frontiers:
            self.get_logger().info("No frontiers found. Exploration complete!")
            self.stop_robot()
            return

        # Choose the closest frontier
        if self.current_goal is None:
            robot_position = (self.map_data.info.origin.position.x, self.map_data.info.origin.position.y)
            self.current_goal = min(frontiers, key=lambda f: np.hypot(f[0] - robot_position[0], f[1] - robot_position[1]))
            self.get_logger().info(f"New goal set: {self.current_goal}")

        # Command robot toward the goal
        self.navigate_to_goal()

    def navigate_to_goal(self):
        """Navigate the robot toward the current goal."""
        if self.current_goal is None:
            return

        goal_x, goal_y = self.current_goal
        twist = Twist()

        # Simple proportional controller for navigation
        robot_x = 0  # Assume the robot starts at the map origin
        robot_y = 0
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = np.hypot(dx, dy)

        if distance < 0.5:  # Goal reached
            self.get_logger().info("Goal reached. Finding new frontier...")
            self.current_goal = None
            return

        twist.linear.x = min(0.2, distance)  # Forward speed
        twist.angular.z = np.arctan2(dy, dx)  # Turn toward the goal

        self.cmd_vel_publisher.publish(twist)

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExploration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
