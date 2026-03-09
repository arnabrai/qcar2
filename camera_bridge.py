#!/usr/bin/env python3
"""
SDV Camera Bridge Node - Threaded Version
Each camera runs in its own thread for true 30fps
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import sys
import threading
import time

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.utilities.vision import Camera2D


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('sdv_camera_bridge')
        self.get_logger().info('🎥 SDV Camera Bridge Starting...')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.publishers = {
            'front': self.create_publisher(Image, '/sdv/camera/front', qos),
            'rear':  self.create_publisher(Image, '/sdv/camera/rear',  qos),
            'left':  self.create_publisher(Image, '/sdv/camera/left',  qos),
            'right': self.create_publisher(Image, '/sdv/camera/right', qos),
        }

        # Camera ID mapping: 0=right, 1=rear, 2=left, 3=front
        cam_ids = {'front': '3', 'rear': '1', 'left': '2', 'right': '0'}

        # Start one thread per camera
        self.running = True
        for name, cam_id in cam_ids.items():
            t = threading.Thread(
                target=self.camera_thread,
                args=(name, cam_id),
                daemon=True
            )
            t.start()
            self.get_logger().info(f'  ✅ Thread started for CSI {name} (id={cam_id})')

        self.get_logger().info('✅ Camera Bridge READY — 30fps per camera!')

    def camera_thread(self, name, cam_id):
        """Each camera runs independently at 30fps."""
        try:
            cam = Camera2D(cameraId=cam_id, frameWidth=640, frameHeight=480, frameRate=30)
        except Exception as e:
            self.get_logger().error(f'Failed to open {name} camera: {e}')
            return

        pub = self.publishers[name]
        rate = 1.0 / 30.0

        while self.running and rclpy.ok():
            t0 = time.time()
            try:
                cam.read()
                if cam.imageData is not None:
                    msg = Image()
                    msg.header.stamp    = self.get_clock().now().to_msg()
                    msg.header.frame_id = f'csi_{name}'
                    msg.height          = cam.imageData.shape[0]
                    msg.width           = cam.imageData.shape[1]
                    msg.encoding        = 'bgr8'
                    msg.step            = cam.imageData.shape[1] * 3
                    msg.data            = cam.imageData.tobytes()
                    pub.publish(msg)
            except Exception:
                pass

            elapsed = time.time() - t0
            sleep_t = rate - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        del cam

    def destroy_node(self):
        self.running = False
        time.sleep(0.2)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
