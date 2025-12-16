#!/usr/bin/env python3
"""
Isaac ROS VSLAM Tutorial
This script demonstrates Visual Simultaneous Localization and Mapping concepts
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import numpy as np


class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vslam_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscribers for camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_rect_color',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for pose estimates
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)

        # VSLAM state variables
        self.prev_frame = None
        self.prev_kp = None
        self.curr_pose = np.eye(4)  # 4x4 identity matrix
        self.kp_detector = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.get_logger().info('Isaac VSLAM Node initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints
        kp = self.kp_detector.detectAndCompute(gray, None)

        if self.prev_frame is not None and self.prev_kp is not None:
            # Match keypoints between frames
            matches = self.bf_matcher.knnMatch(self.prev_kp[1], kp[1], k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 10:  # Need minimum matches for pose estimation
                # Extract corresponding points
                src_pts = np.float32([self.prev_kp[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[0][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Estimate essential matrix and pose
                E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.camera_matrix,
                                              method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None:
                    # Decompose essential matrix to get rotation and translation
                    _, R, t, _ = cv2.recoverPose(E, dst_pts, src_pts, self.camera_matrix)

                    # Update current pose
                    transformation = np.eye(4)
                    transformation[:3, :3] = R
                    transformation[:3, 3] = t.flatten()
                    self.curr_pose = self.curr_pose @ transformation

                    # Publish pose estimate
                    self.publish_pose_estimate()

        # Update previous frame data
        self.prev_frame = gray
        self.prev_kp = kp

    def publish_pose_estimate(self):
        """Publish the estimated pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Extract position and orientation from pose matrix
        pose_msg.pose.position.x = self.curr_pose[0, 3]
        pose_msg.pose.position.y = self.curr_pose[1, 3]
        pose_msg.pose.position.z = self.curr_pose[2, 3]

        # Convert rotation matrix to quaternion (simplified)
        # In a real implementation, proper conversion would be used
        pose_msg.pose.orientation.w = 1.0  # Simplified

        self.pose_pub.publish(pose_msg)

        # Also publish odometry
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.pose.pose = pose_msg.pose
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)

    vslam_node = IsaacVSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()