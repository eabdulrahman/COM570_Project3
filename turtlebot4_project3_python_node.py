#!/usr/bin/env python3
"""
    BSU
    COMP 570 - ROBOTICS
    PROJECT 3
    GROUP 1, Team Members: 
    - Arshjot Saini
    - Eissa Abdulrahman
    - Rauf Bairamov
    - Revanth Reddy Kanubaddi
"""

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import face_recognition
import math
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from time import time  # Time-based rotation was instead of odometry

class MainNode(Node):
    def __init__(self):
        super().__init__('face_following_node')
        self.publisher = self.create_publisher(Twist, '/compy86/cmd_vel', 10)   #twist message publisher
        self.lidar_sub = self.create_subscription(LaserScan, '/compy86/scan', self.lidar_callback, 10)  #Lidar subscriper
        self.image_sub = self.create_subscription(Image, '/compy86/oakd/rgb/preview/image_raw', self.image_callback, 10)    #Camera subscriper
        self.bridge = CvBridge()
        self.cliff_detected = False
        self.is_turning = False
        self.create_subscription(Bool, "/compy86/hazard_detection", self.cliff_callback, 10)  # Clip sensor subscriper

        # Time variables for rotation, the appoximate time that takes turtlebot4 turns 90 degrees
        self.rotation_start_time = None
        self.rotation_duration = 3.0  # Rotation duration in seconds (for 90-degree turn)
        
        """
        The code to load known face from an image file was copied from the following:
        https://github.com/jsantore/FirstTurtleBotVision/blob/master/turtlebot_camera_reader/turtlebot4_camera_consumer_node.py
        https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py#L3
        """
        self.target_face = face_recognition.load_image_file("/home/ubuntu/Project3/src/turtlebot4_python_tutorials/image.jpeg")
        self.target_encoding = face_recognition.face_encodings(self.target_face)[0]
        self.known_face_codes = [self.target_encoding]
        self.known_names = ["Eissa"]

        self.safe_distance = 0.25       #safe distance of the robot is set to 0.25 meters and above
        self.closest_front_distance = 10.0
        self.face_detected = False
        self.face_offset = 0.0
        self.found_target_face = False
        self.searching = True  # Start in search mode

    def lidar_callback(self, msg: LaserScan):
        if self.is_turning:
            return  # Skip obstacle avoidance during cliff turn

        # Existing logic...
        center_indices = self.get_indices_in_angle_range(msg, -20, 20)
        front_distances = [msg.ranges[i] for i in center_indices if not math.isinf(msg.ranges[i])]
        self.closest_front_distance = min(front_distances, default=10.0)

        action = self.decide_obstacle_avoidance_action(msg.ranges, msg.angle_min, msg.angle_increment)
        self.move_robot(action)

    def get_indices_in_angle_range(self, msg: LaserScan, angle_min_deg, angle_max_deg):
        angle_min = math.radians(angle_min_deg)
        angle_max = math.radians(angle_max_deg)
        return [i for i in range(len(msg.ranges))
                if angle_min <= msg.angle_min + i * msg.angle_increment <= angle_max]

    #This function is control the movment of the robot
    def move_robot(self, action):
        twist = Twist()
        if action == 'forward':
            twist.linear.x = 0.2
        elif action == 'left':
            twist.angular.z = 0.5
        elif action == 'right':
            twist.angular.z = -0.5
        elif action == 'stop':
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.publisher.publish(twist)

    #This function is called when the cliff sensor is actived
    def cliff_callback(self, msg: Bool):
        if msg.data and not self.is_turning:
            self.is_turning = True
            self.perform_left_turn()

    #This function makes the robot turns left, rotates -90 degrees
    def perform_left_turn(self):
        # Start rotation using time-based logic
        if self.rotation_start_time is None:
            self.rotation_start_time = time()  # Record the start time of the turn

        twist = Twist()
        twist.angular.z = 0.5  # Set angular velocity for rotation

        # Keep rotating until the time exceeds the desired rotation duration
        current_time = time() - self.rotation_start_time
        if current_time < self.rotation_duration:
            self.publisher.publish(twist)  # Keep publishing the turn command
        else:
            # Stop rotation after the specified duration
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            self.is_turning = False
            self.rotation_start_time = None  # Reset the rotation time for future turns


    """
    merging camera subscriber and face recognition example
    https://github.com/jsantore/FirstTurtleBotVision/blob/master/turtlebot_camera_reader/turtlebot4_camera_consumer_node.py
    https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py#L3
    """
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        twist = Twist()
        name = "Unknown Face"

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_center_x = (left + right) // 2
            image_center_x = frame.shape[1] // 2
            self.face_offset = face_center_x - image_center_x

            matches = face_recognition.compare_faces(self.known_face_codes, face_encodings[0])
            face_distances = face_recognition.face_distance(self.known_face_codes, face_encodings[0])
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                # Known target face detected
                name = self.known_names[best_match_index]
                self.found_target_face = True
                self.get_logger().info(f"✅ Found target face: {name}")
                os.system(f'espeak "Hi {name}!"')

                # Approach the face
                if self.closest_front_distance < self.safe_distance:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info("Too close. Stopping.")
                else:
                    twist.linear.x = 0.2
                    twist.angular.z = -float(self.face_offset) / 200.0
                    self.get_logger().info(f"Approaching face. Offset: {self.face_offset}")
            else:
                # When an unknown face detected, the robot will rotate left to search of new faces.
                if not self.found_target_face:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.3  # Rotate left only
                    self.get_logger().info("Unknown face detected. Rotating left to search for known face...")

            # Draw box and label around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            if not self.found_target_face:
                # If No face was detected continue rotating left and search for faces
                twist.linear.x = 0.0
                twist.angular.z = 0.3
                self.get_logger().info("No face detected. Rotating left to search...")

        # Only publish twist if target not found or approaching
        if not self.found_target_face or twist.linear.x > 0:
            self.publisher.publish(twist)

        # Show camera view
        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os._exit(0)


        """
        This function decides robot movement direction based on obstacle positions using LiDAR data.
        Returns:
            'left' if obstacle on front and right,
            'right' if obstacle on front and left,
            'forward' if path is clear,
            'stop' if surrounded closely.
        This code generated using ChatGPT
        """
    def decide_obstacle_avoidance_action(self, lidar_ranges, angle_min, angle_increment):
        def get_index(deg):
            rad = math.radians(deg)
            return int((rad - angle_min) / angle_increment)

        front_indices = range(get_index(-20), get_index(20))
        left_indices = range(get_index(30), get_index(90))
        right_indices = range(get_index(-90), get_index(-30))

        front_distances = [lidar_ranges[i] for i in front_indices if not math.isinf(lidar_ranges[i])]
        left_distances = [lidar_ranges[i] for i in left_indices if not math.isinf(lidar_ranges[i])]
        right_distances = [lidar_ranges[i] for i in right_indices if not math.isinf(lidar_ranges[i])]

        front_min = min(front_distances, default=10.0)
        left_min = min(left_distances, default=10.0)
        right_min = min(right_distances, default=10.0)

        danger_threshold = 0.5

        if front_min < danger_threshold:
            if right_min < danger_threshold and left_min >= danger_threshold:
                return 'right'
            elif left_min < danger_threshold and right_min >= danger_threshold:
                return 'left'
            elif left_min < danger_threshold and right_min < danger_threshold:
                return 'stop'
            else:
                return 'left'  # Default strategy
        return 'forward'

#The main function
def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
