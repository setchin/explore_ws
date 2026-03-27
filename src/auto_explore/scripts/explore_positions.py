#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import PositionTarget, State as MavrosState
from mavros_msgs.srv import SetMode, CommandBool
from quadrotor_msgs.msg import PositionCommand, GoalSet


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ExplorePositionsNode(object):
    def __init__(self):
        rospy.init_node('explore_positions', anonymous=False)

        self.config = load_config(self._resolve_config_path())
        self.takeoff_height = self.config.get('takeoff_height', 1.5)
        self.takeoff_hover_time = self.config.get('takeoff_hover_time', 2.0)
        self.positions_to_explore = self.config.get('position_to_explore', [{"x": 0.0, "y": 0.0, "z": 1.0, "yaw_deg": 0.0}])
        self.use_velocity_cmd = self.config.get('use_velocity_cmd', False)
        self.yaw_tolerance_deg = self.config.get('yaw_tolerance_deg', 5.0)
        self.goal_hover_time = self.config.get('goal_hover_time', 1.0)
        self.goal_reach_tol = self.config.get('goal_reach_tolerance_m', 0.1)

        self.local_pose = PoseStamped()
        self.local_velocity = TwistStamped()
        self.current_state = MavrosState()
        self.latest_position_cmd = None

        self.state = 'TAKEOFF_HOVER'
        self.takeoff_reached_time = None
        self.current_goal_idx = 0
        self.goal_published = False
        self.hover_start_time = None
        self.track_ego_planner = True

        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_with_id', GoalSet, queue_size=10)

        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.local_pose_cb)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.local_velocity_cb)
        rospy.Subscriber('/mavros/state', MavrosState, self.state_cb)
        rospy.Subscriber('/position_cmd', PositionCommand, self.position_cmd_cb)

        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        rospy.loginfo('[AUTO_EXPLORE] Loaded %d exploration points.', len(self.positions_to_explore))

    def _resolve_config_path(self):
        config_path = rospy.get_param('~config_path', '')
        if config_path and os.path.isfile(config_path):
            return config_path

        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.normpath(os.path.join(script_dir, '..', 'cfg', 'landing_param_astro_with_yaw.yaml'))
        if os.path.isfile(default_path):
            rospy.loginfo('[AUTO_EXPLORE] Using default config path: %s', default_path)
            return default_path

        raise RuntimeError('Config file not found. Set ~config_path to a valid yaml file.')

    def local_pose_cb(self, msg):
        self.local_pose = msg

    def local_velocity_cb(self, msg):
        self.local_velocity = msg

    def state_cb(self, msg):
        self.current_state = msg

    def position_cmd_cb(self, msg):
        if (rospy.Time.now() - msg.header.stamp).to_sec() < 0.5:
            self.latest_position_cmd = msg
        else:
            self.latest_position_cmd = None

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def get_current_yaw(self):
        q = self.local_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def handle_flight_mode(self):
        if self.current_state.mode != 'OFFBOARD':
            try:
                self.set_mode_client(custom_mode='OFFBOARD')
            except rospy.ServiceException as e:
                rospy.logwarn('[AUTO_EXPLORE] SetMode failed: %s', e)

        if not self.current_state.armed:
            try:
                self.arming_client(True)
            except rospy.ServiceException as e:
                rospy.logwarn('[AUTO_EXPLORE] Arming failed: %s', e)

    def publish_goal(self, xyz):
        goal_msg = GoalSet()
        goal_msg.goal[0] = xyz[0]
        goal_msg.goal[1] = xyz[1]
        goal_msg.goal[2] = xyz[2]
        while self.goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)
        self.goal_pub.publish(goal_msg)

    def publish_hold_position(self, x, y, z, yaw=None):
        sp = PositionTarget()
        sp.header.stamp = rospy.Time.now()
        sp.header.frame_id = 'map'
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        if yaw is None:
            sp.type_mask = (
                PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE
            )
        else:
            sp.type_mask = (
                PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE
            )
            sp.yaw = yaw

        sp.position.x = x
        sp.position.y = y
        sp.position.z = z
        self.setpoint_pub.publish(sp)

    def publish_setpoint_from_position_cmd(self, fallback_x, fallback_y, fallback_z, fallback_yaw):
        msg = self.latest_position_cmd
        sp = PositionTarget()
        sp.header.stamp = rospy.Time.now()
        sp.header.frame_id = 'map'
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        if msg is not None:
            if self.use_velocity_cmd:
                sp.type_mask = (
                    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                    PositionTarget.IGNORE_YAW_RATE
                )
                sp.position.x = msg.position.x
                sp.position.y = msg.position.y
                sp.position.z = msg.position.z
                sp.velocity.x = msg.velocity.x
                sp.velocity.y = msg.velocity.y
                sp.velocity.z = msg.velocity.z
                sp.yaw = msg.yaw
            else:
                sp.type_mask = (
                    PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                    PositionTarget.IGNORE_YAW_RATE
                )
                sp.position.x = msg.position.x
                sp.position.y = msg.position.y
                sp.position.z = msg.position.z
                sp.yaw = msg.yaw
        else:
            sp.type_mask = (
                PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE | PositionTarget.IGNORE_YAW
            )
            sp.position.x = fallback_x
            sp.position.y = fallback_y
            sp.position.z = fallback_z
            sp.yaw = fallback_yaw
            rospy.logwarn_throttle(2.0, '[AUTO_EXPLORE] position_cmd missing, using fallback.')

        self.setpoint_pub.publish(sp)

    def tick_takeoff_hover(self):
        self.handle_flight_mode()

        cur = self.local_pose.pose.position
        vel = self.local_velocity.twist.linear

        self.publish_hold_position(0.0, 0.0, self.takeoff_height)

        pos_tol = 0.1
        vel_tol = 0.1
        dist = np.linalg.norm([cur.x, cur.y, cur.z - self.takeoff_height])
        vel_norm = np.linalg.norm([vel.x, vel.y, vel.z])
        now = rospy.Time.now().to_sec()

        if self.takeoff_reached_time is None and dist < pos_tol and vel_norm < vel_tol:
            self.takeoff_reached_time = now
            rospy.loginfo('[AUTO_EXPLORE] Reached takeoff point, start hover timing.')

        if self.takeoff_reached_time is not None and now - self.takeoff_reached_time >= self.takeoff_hover_time:
            rospy.loginfo('[AUTO_EXPLORE] Takeoff hover finished, entering EXPLORE state.')
            self.state = 'EXPLORE'

    def tick_explore(self):
        self.handle_flight_mode()

        if len(self.positions_to_explore) == 0:
            self.publish_hold_position(
                self.local_pose.pose.position.x,
                self.local_pose.pose.position.y,
                self.local_pose.pose.position.z
            )
            rospy.logwarn_throttle(2.0, '[AUTO_EXPLORE] No goals in position_to_explore, holding.')
            return

        if self.current_goal_idx >= len(self.positions_to_explore):
            last_goal = self.positions_to_explore[-1]
            last_yaw = np.deg2rad(last_goal.get('yaw_deg', last_goal.get('yaw', 0.0)))
            self.publish_hold_position(last_goal.get('x', 0.0), last_goal.get('y', 0.0), last_goal.get('z', 1.0), last_yaw)
            rospy.loginfo_throttle(2.0, '[AUTO_EXPLORE] Mission complete, hovering at final point.')
            return

        goal = self.positions_to_explore[self.current_goal_idx]
        target_xyz = [goal.get('x', 0.0), goal.get('y', 0.0), goal.get('z', 1.0)]
        target_yaw = np.deg2rad(goal.get('yaw_deg', goal.get('yaw', 0.0)))

        if not self.goal_published:
            self.publish_goal(target_xyz)
            self.goal_published = True
            self.hover_start_time = None
            rospy.loginfo('[AUTO_EXPLORE] Goal %d/%d: [%.2f, %.2f, %.2f], yaw %.1f deg',
                          self.current_goal_idx + 1,
                          len(self.positions_to_explore),
                          target_xyz[0], target_xyz[1], target_xyz[2],
                          np.rad2deg(target_yaw))

        cur_pos = self.local_pose.pose.position
        dist = np.linalg.norm([
            cur_pos.x - target_xyz[0],
            cur_pos.y - target_xyz[1],
            cur_pos.z - target_xyz[2],
        ])

        cur_vel = self.local_velocity.twist.linear
        vel_norm = np.linalg.norm([cur_vel.x, cur_vel.y, cur_vel.z])
        if dist <= self.goal_reach_tol and vel_norm < 0.1:
            self.track_ego_planner = False
            cur_yaw = self.get_current_yaw()
            yaw_err_deg = np.rad2deg(abs(self.wrap_to_pi(target_yaw - cur_yaw)))

            if yaw_err_deg <= self.yaw_tolerance_deg:
                rospy.loginfo("Fine tuning pose and yaw using Position Command, dist to goal: %.2f m", dist)
                self.publish_hold_position(cur_pos.x, cur_pos.y, cur_pos.z, target_yaw) 
                if self.hover_start_time is None:
                    self.hover_start_time = rospy.Time.now().to_sec()
                    rospy.loginfo('[AUTO_EXPLORE] Goal %d heading aligned, hover timer started.', self.current_goal_idx + 1)

                if rospy.Time.now().to_sec() - self.hover_start_time >= self.goal_hover_time:
                    rospy.loginfo('[AUTO_EXPLORE] Goal %d completed.', self.current_goal_idx + 1)
                    self.current_goal_idx += 1
                    self.goal_published = False
                    self.hover_start_time = None
                    self.track_ego_planner = True
            else:
                self.hover_start_time = None
                rospy.loginfo_throttle(1.0, '[AUTO_EXPLORE] Aligning yaw at goal %d: err %.2f deg.',
                                       self.current_goal_idx + 1, yaw_err_deg)
                rospy.loginfo("Aligning yaw using Position Command, dist to goal: %.2f m", dist)
                self.publish_hold_position(cur_pos.x, cur_pos.y, cur_pos.z, target_yaw) 
        elif self.track_ego_planner:
            rospy.loginfo("Tracking ego planner, dist to goal: %.2f m", dist)
            self.publish_setpoint_from_position_cmd(
            self.local_pose.pose.position.x,
            self.local_pose.pose.position.y,
            self.local_pose.pose.position.z,
            target_yaw,
            )
        else:
            rospy.loginfo("Moving back to goal using Position Command, dist to goal: %.2f m", dist)
            self.publish_hold_position(target_xyz[0], target_xyz[1], target_xyz[2])

    def spin(self):
        rate = rospy.Rate(rospy.get_param('~rate', 100))
        while not rospy.is_shutdown():
            if self.state == 'TAKEOFF_HOVER':
                self.tick_takeoff_hover()
            else:
                self.tick_explore()
            rate.sleep()


def main():
    node = ExplorePositionsNode()
    node.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
