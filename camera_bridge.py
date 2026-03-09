#!/usr/bin/env python3
"""
SDV Camera Bridge Node
Publishes all cameras to ROS2 topics:
  /sdv/camera/front    → CSI front camera (640x480)
  /sdv/camera/rear     → CSI rear camera  (640x480)
  /sdv/camera/left     → CSI left camera  (640x480)
  /sdv/camera/right    → CSI right camera (640x480)
  /sdv/camera/rgb      → RealSense color  (1280x720)
  /sdv/camera/depth    → RealSense depth  (forwarded)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import numpy as np
import sys
import threading
import time

# Add pal to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.utilities.vision import Camera2D


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('sdv_camera_bridge')
        self.get_logger().info('🎥 SDV Camera Bridge Starting...')

        # QoS for image publishing
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Publishers ──
        self.pub_front = self.create_publisher(Image, '/sdv/camera/front', qos)
        self.pub_rear  = self.create_publisher(Image, '/sdv/camera/rear',  qos)
        self.pub_left  = self.create_publisher(Image, '/sdv/camera/left',  qos)
        self.pub_right = self.create_publisher(Image, '/sdv/camera/right', qos)

        # ── CSI Cameras ──
        # Camera IDs: 0=right, 1=rear, 2=left, 3=front
        self.cameras = {}
        cam_config = [
            ('front', '3'), ('rear',  '1'),
            ('left',  '2'), ('right', '0'),
        ]

        for name, cam_id in cam_config:
            try:
                self.cameras[name] = Camera2D(
                    cameraId=cam_id,
                    frameWidth=640,
                    frameHeight=480,
                    frameRate=30
                )
                self.get_logger().info(f'  ✅ CSI {name} camera (id={cam_id}) opened')
            except Exception as e:
                self.get_logger().warn(f'  ⚠️  CSI {name} camera failed: {e}')
                self.cameras[name] = None

        # ── Timers ──
        # CSI cameras at 30fps
        self.create_timer(1.0/30.0, self.publish_csi)

        self.get_logger().info('✅ Camera Bridge READY!')
        self.get_logger().info('   Topics:')
        self.get_logger().info('   /sdv/camera/front  /sdv/camera/rear')
        self.get_logger().info('   /sdv/camera/left   /sdv/camera/right')

    def frame_to_msg(self, frame, frame_id):
        """Convert numpy BGR frame to ROS2 Image message."""
        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height          = frame.shape[0]
        msg.width           = frame.shape[1]
        msg.encoding        = 'bgr8'
        msg.step            = frame.shape[1] * 3
        msg.data            = frame.tobytes()
        return msg

    def publish_csi(self):
        """Read and publish all CSI cameras."""
        pub_map = {
            'front': self.pub_front,
            'rear':  self.pub_rear,
            'left':  self.pub_left,
            'right': self.pub_right,
        }

        for name, pub in pub_map.items():
            cam = self.cameras.get(name)
            if cam is None:
                continue
            try:
                cam.read()
                if cam.imageData is not None:
                    msg = self.frame_to_msg(cam.imageData, f'csi_{name}')
                    pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'CSI {name} read error: {e}', throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Camera Bridge shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
