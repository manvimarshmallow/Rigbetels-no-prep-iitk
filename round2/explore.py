import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np
class ObstacleAvoidanceBot(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_bot')

        # Publisher for robot velocity
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for LiDAR scan data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Default movement command
        self.velocity_command = Twist()

        # Timer to publish velocity at a fixed rate
        self.create_timer(0.1, self.publish_velocity)
    
        self.state = "exploring"  # States: exploring, avoiding_obstacle

        self.get_logger().info("Obstacle Avoidance Bot initialized.")

    def scan_callback(self, msg):
        """Callback to process LiDAR data and avoid obstacles."""
        ranges = msg.ranges
        min_distance = 0.5  # Minimum safe distance in meters
        # for i in range(ranges):
        #     if ranges[i] < 0.1 :
        #         ranges[i] = 10000
        ranges= np.array(ranges)
        ranges[ranges==0.0]= 100000
        # Divide LiDAR data into regions: left, front, right, and back
        segment_size = len(ranges) // 4
        left = min(ranges[:segment_size])
        front = min(ranges[segment_size:2*segment_size])
        right = min(ranges[2*segment_size:3*segment_size])
        back = min(ranges[3*segment_size:])

        # Obstacle avoidance logic
        if front < min_distance:
            self.get_logger().info("Obstacle detected ahead! Turning...")
            self.velocity_command.linear.x = 0.0
            self.velocity_command.angular.z = 0.5 
            self.state = "avoiding_obstacle"
        elif left < min_distance:
            self.get_logger().info("Obstacle detected on the left! Moving away.")
            self.velocity_command.linear.x = 0.2
            self.velocity_command.angular.z = -0.5
            self.state = "avoiding_obstacle"
        elif right < min_distance:
            self.get_logger().info("Obstacle detected on the right! Moving away.")
            self.velocity_command.linear.x = 0.2
            self.velocity_command.angular.z = 0.5
            self.state = "avoiding_obstacle"
        elif back < min_distance:
            self.get_logger().info("Obstacle detected at the back! Moving forward.")
            self.velocity_command.linear.x = 0.3
            self.velocity_command.angular.z = 0.0
            self.state = "avoiding_obstacle"
        else:
            if self.state == "avoiding_obstacle":
                self.get_logger().info("Path cleared. Resuming exploration.")
            # No obstacles, move forward
            self.velocity_command.linear.x = 0.3
            self.velocity_command.angular.z = 0.0
            self.state = "exploring"

    def publish_velocity(self):
        """Publish the current velocity command."""
        self.cmd_vel_publisher.publish(self.velocity_command)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceBot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

