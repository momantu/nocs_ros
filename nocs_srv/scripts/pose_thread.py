#!/usr/bin/env python  
import roslib
import rospy
from geometry_msgs.msg import Pose
import tf
import tf.transformations as tft
import threading
import time


def generate_pose(T, camera_optical_frame):
    print('publish TF for ', T)
    rate = rospy.Rate(3)  # Hz

    p = Pose()
    p.position.x = T[0, 3]
    p.position.y = T[1, 3]
    p.position.z = T[2, 3]
    R = T[:3, :3]
    roll, pitch, yaw = tft.euler_from_matrix(T)
    q = tft.quaternion_from_euler(roll, pitch, yaw)
    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]

    tf_brocast(p, camera_optical_frame)


def tf_brocast(p, frame_id):
    br = tf.TransformBroadcaster()
    br.sendTransform((p.position.x, p.position.y, p.position.z), \
                     (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w), \
                     rospy.Time.now(), "object_predicted", frame_id)


class Concur(threading.Thread):
    def __init__(self):
        super(Concur, self).__init__()
        self.iterations = 0
        self.daemon = True  # Allow main to exit even if still running.
        self.paused = True  # Start out paused.
        self.state = threading.Condition()

    def set_T(self, T, camera_optical_frame):
        self.T = T
        self.camera_optical_frame = camera_optical_frame

    def run(self):
        self.resume()
        while True:
            with self.state:
                if self.paused:
                    self.state.wait()  # Block execution until notified.
            generate_pose(self.T, self.camera_optical_frame)
            time.sleep(2.5)

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()  # Unblock self if waiting.

    def pause(self):
        with self.state:
            self.paused = True  # Block self.
            print("\n Pause sending Transform.")
