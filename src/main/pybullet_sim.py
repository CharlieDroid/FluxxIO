import pybullet as p
import pybullet_data
import math
import time
import numpy as np


def simp_angle(a):
    _2pi = 2 * np.pi
    return (a + np.pi) % _2pi - np.pi


def compute_distance_reward(d):
    a = 0.1  # 0.1m
    if d < a:
        m1 = -1/a
        reward = m1*d + 1
    else:
        m2 = -1/(1 - a)
        b = -m2*a
        reward = m2*d + b
    return np.clip(reward, -1, 1)


def cart2sphere(p_pos, q_pos):
    g = np.array(q_pos) - np.array(p_pos)
    rho = np.linalg.norm(g)
    theta = np.arctan2(g[1], g[0])  # arctan2 already adds -pi
    phi = np.arccos(g[2]/rho)
    return (rho, theta, phi)


class ObservationSpace:
    def __init__(self, sample_state, min_max_joint_pos_list, min_max_joint_velo_list):
        self.shape = (len(sample_state),)
        lower_bound = [float("-inf") for _ in range(len(sample_state))]
        upper_bound = [float("inf") for _ in range(len(sample_state))]
        max_angular_velo = math.pi*160
        lower_bound[:3] = [-max_angular_velo for _ in range(3)]
        upper_bound[:3] = [max_angular_velo for _ in range(3)]

        lower_bound[3:6] = [-math.pi for _ in range(3)]
        upper_bound[3:6] = [math.pi for _ in range(3)]

        lower_bound[6] = -math.pi
        upper_bound[6] = math.pi

        lower_bound[7] = 0
        upper_bound[7] = 1

        lower_bound[8] = -1
        upper_bound[8] = 1

        lower_bound[9] = 0
        upper_bound[9] = 1

        min_joint_pos_list, max_joint_pos_list = min_max_joint_pos_list
        min_joint_velo_list, max_joint_velo_list = min_max_joint_velo_list
        for i in range(10, 18):
            if i % 2 == 0:  # pos
                lower_bound[i] = min_joint_pos_list[(i // 2) - 5]
                upper_bound[i] = max_joint_pos_list[(i // 2) - 5]
            else:
                lower_bound[i] = min_joint_velo_list[(i // 2) - 5]
                upper_bound[i] = max_joint_velo_list[(i // 2) - 5]

        for i in range(18, 22):
            lower_bound[i] = -1
            upper_bound[i] = 1

        self.bounds = [lower_bound, upper_bound]


class ActionSpace:
    def __init__(self, n_actions):
        self.shape = (n_actions,)
        self.bounds = (-1, 1)


class Environment:
    def __init__(self, robot_path, record=False, render="default", gravity=-9.80665, debug=False, print_reward=False):
        # TODO: combine observation and reward because functions are called twice
        # TODO: check timestep difference
        # should consider removing this
        self.gear_reduction = 12
        self.gear_efficiency = 0.8
        self.gear_reduction_wheel = 1
        self.max_motor_torque = 0.24
        self.max_motor_speed = math.pi*160  # in rad/s 4800 RPM
        self.spring_force = 0.8
        max_accel = 0.8  # m/s from online
        self.max_linear_velo = max_accel * 20  # seconds

        self.robot_start_pos = (0, 0, 0.16)
        self.robot_start_orn = p.getQuaternionFromEuler((0, 0, 0))
        self.debug = debug
        self.print_reward = print_reward
        self.render = render
        if self.debug or (render == "human"):
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        p.setGravity(0, 0, gravity)
        self.plane_id = p.loadURDF("plane.urdf", useMaximalCoordinates=1)
        self.robot_id = p.loadURDF(robot_path, self.robot_start_pos, self.robot_start_orn,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
        self._set_camera()
        # check function for more explanation
        self.add_constraints(4, 1, 5, 2)
        self.add_constraints(10, 7, 5, 8)
        self.prepare_joints()
        self.controls = {}
        if self.debug:
            self.add_debug_params()
        else:
            self.init_control_joints()
        if record:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                f"wbr_vid.mp4")

        # environment vars
        self.name = "FluxxIO"
        self.reward_range = (-np.inf, np.inf)
        self.curriculum_level = 0
        self.curriculum_counter = 0
        self.dt = 1/240  # 240 Hz sampling rate of simulation
        self.goal = [0, -1, 0]  # 1 meter in front of robot
        self.last_actions = [0, 0, 0, 0]  # Velo Controls
        self.time_step = 0
        self.time_step_max = 4800
        self.action_space = ActionSpace(len(self._get_controls_ids()))
        self.observation_space = self.init_observation_space()
        p.addUserDebugPoints([self.goal], [[1, 0, 1]], 10)

    def run(self):
        # self.reset()
        while True:
            for joint_name in list(self.controls.keys()):
                val = p.readUserDebugParameter(self.controls[joint_name])
                joint_id = self.get_joint_id(joint_name)
                force = self.max_motor_torque*self._get_multiplier(joint_name)
                if ("wheel" in joint_name) or ("motor" in joint_name):
                    p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL,
                                            targetVelocity=val, force=force)

            ### DEBUGGING CODE HERE ###
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            orn = p.getEulerFromQuaternion(orn)
            distance, theta, phi = cart2sphere(pos, self.goal)
            theta = simp_angle(theta + np.pi/2)
            error = abs(theta - orn[2]) / np.pi
            z_axis = orn[2]
            print(f"{error=}, {theta=}, {z_axis=}")
            ############################
            time.sleep(0.01)
            p.stepSimulation()  # default is 240 Hz, documentation suggests not to change it

    def curriculum(self, orientation, direction, distance, action, joint_position, touch):
        orn_weight = 3
        dir_weight = 1
        distance_weight = 4
        joint_action_weight = 0.2
        joint_pos_weight = 3

        if joint_position > 0.1 and touch > -1 and orientation > 0.1:
            self.curriculum_counter += 1
        else:
            self.curriculum_counter = 0

        if self.curriculum_level == 0 and self.curriculum_counter > self.time_step_max // 3:
            self.curriculum_level += 1
            self.curriculum_counter = 0
            print(f"...curriculum level up to {self.curriculum_level}...")

        if self.curriculum_level == 0:
            return orn_weight, 0, 0, 0, joint_pos_weight
        elif self.curriculum_level == 1:
            orn_weight = 1
            return orn_weight, dir_weight, distance_weight, joint_action_weight, joint_pos_weight

    def _get_reward(self, obs):
        # scale everything between -1 and 1
        # at the end increase it to desired values

        # orientation reward min = -1 max = 1
        orn_reward = 0.5 * (math.cos(3*obs[3]) + math.cos(3*obs[4]))

        # desired direction min = 0 max = 1
        dir_reward = 1 - obs[8]

        # distance reward if distance = [3, 0.1, 0] then reward = [-1, 0, 1]
        pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        distance, _, _ = cart2sphere(pos, self.goal)
        distance_reward = compute_distance_reward(distance)

        # joint actions min = 0 max = 1
        total_actions = sum(abs(a) for a in self.last_actions)
        joint_action_penalty = -total_actions / len(self.last_actions)

        # desired joint position (should be changed) min = -1 max = 1
        total_joint_error = 0
        for joint_name in list(self.controls.keys()):
            if "motor" in joint_name:
                upper_limit = p.getJointInfo(self.robot_id, self.controls[joint_name])[9]
                state = p.getJointState(self.robot_id, self.controls[joint_name])
                # give positive reward if near joint position
                # the farther away it is, it will become negative
                total_joint_error += ((upper_limit/2) - abs(state[0]))/(upper_limit/2)
        joint_pos_penalty = total_joint_error / len(self.controls)

        # body touch
        touch_penalty = 0
        for joint_id in range(p.getNumJoints(self.robot_id)):
            infos = p.getJointInfo(self.robot_id, joint_id)
            link_name = infos[12].decode('UTF-8')
            if ("wheel" in link_name) or ("closing" in link_name):
                pass
            else:
                if p.getContactPoints(self.robot_id, -1, joint_id, -1):
                    touch_penalty += -5
        if p.getContactPoints(self.robot_id, -1, -1, -1):
            touch_penalty += -7

        orn_weight, dir_weight, distance_weight, joint_action_weight, joint_pos_weight = \
            self.curriculum(orn_reward, dir_reward, distance_reward,
                            joint_action_penalty, joint_pos_penalty, touch_penalty)
        orn_reward *= orn_weight
        dir_reward *= dir_weight
        distance_reward *= distance_weight
        joint_action_penalty *= joint_action_weight
        joint_pos_penalty *= joint_pos_weight

        reward = (orn_reward +
                  dir_reward +
                  distance_reward +
                  joint_action_penalty +
                  joint_pos_penalty +
                  touch_penalty)
        reward = np.clip(reward, -10, 10)

        if self.print_reward:
            curriculum_counter = self.curriculum_counter
            print(
                f"{reward=}, {orn_reward=}, {dir_reward=}, {distance_reward=}, {joint_action_penalty=}, {joint_pos_penalty=}, {touch_penalty=}, {curriculum_counter=}")

        return reward

    def step(self, action):
        if self.render == "human":
            self._set_camera()

        # receives list
        self._do_action(action)
        p.stepSimulation()

        ob = self._get_obs()
        reward = self._get_reward(ob)

        truncated = self.time_step >= self.time_step_max
        terminated = (abs(ob[3]) >= np.pi / 2.25) or (abs(ob[4]) >= np.pi / 2.25)  # if it falls down then terminate
        if terminated:
            reward = -100
        self.time_step += 1
        return ob, reward, terminated, truncated, {}

    def reset(self):
        # orn = p.getQuaternionFromEuler(np.random.uniform(-np.pi / 2, np.pi / 2, 3))
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        dist, _, _ = cart2sphere(pos, self.goal)
        if dist < 0.1:
            print("...goal reached...")
            x_y = np.random.uniform(-1, 1, 2)
            self.goal = [x_y[0], x_y[1], 0]
        p.resetBasePositionAndOrientation(self.robot_id, self.robot_start_pos, self.robot_start_orn)
        self.prepare_joints()
        self.time_step = 0
        for _ in range(self.time_step_max // 10):
            p.stepSimulation()
        return self._get_obs(), {}

    def _get_obs(self):
        """
        angular velo of body (rad/s),
        orientation (rad),
        direction (psi z-axis and phi x-axis) (rad),
        heading err (optional) (0 to 1),
        stair boolean (0),
        joint position (rad),
        joint velo (rad/s),
        last joint action (rad/s)
        """
        obs = []
        # angular velo (rad/s xyz 0 1 2) 3
        angular_velo = p.getBaseVelocity(self.robot_id)[1]
        obs.extend(angular_velo)

        # orientation (rad xyz 3 4 5) 3
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        orn = p.getEulerFromQuaternion(orn)
        obs.extend(orn)

        # direction (psi z-axis 6) 1
        _, psi, phi = cart2sphere(pos, self.goal)
        psi = simp_angle(psi + np.pi / 2)
        obs.append(psi)

        # direction (phi x-axis 7) 1
        obs.append(phi)

        # heading error (8) 1 [ONLY ACCOUNTING FOR PSI (NO PHI)]
        heading_err = abs(psi - orn[2]) / np.pi
        obs.append(heading_err)

        # stairs (9) 1
        stair = 0
        obs.append(stair)

        # joint pos and velo (4 joints * 2 = 8 [10, 11, 12, 13, 14, 15, 16, 17])
        joint_states = p.getJointStates(self.robot_id, self._get_controls_ids())
        for joint_state in joint_states:
            obs.append(simp_angle(joint_state[0]))
            obs.append(joint_state[1])

        # last actions (18, 19, 20, 21) 4
        obs.extend(self.last_actions)
        return obs

    def _do_action(self, actions):
        for idx in range(len(actions)):
            joint_name = list(self.controls.keys())[idx]
            joint_id = self.controls[joint_name]
            force = self.max_motor_torque*self._get_multiplier(joint_name)
            speed = actions[idx]*(self.max_motor_speed*self.gear_efficiency/self._get_multiplier(joint_name))
            p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL,
                                    targetVelocity=speed,
                                    force=force)
        self.last_actions = actions

    def init_control_joints(self):
        for joint_id in range(p.getNumJoints(self.robot_id)):
            joint_name = self.get_joint_name(joint_id)
            if ("wheel" in joint_name) or ("motor" in joint_name):
                self.controls[joint_name] = joint_id

    def init_observation_space(self):
        min_joint_pos_list = []
        max_joint_pos_list = []
        min_joint_velo_list = []
        max_joint_velo_list = []
        for joint_name in list(self.controls.keys()):
            max_speed = self.max_motor_speed * self.gear_efficiency / self._get_multiplier(joint_name)
            if "motor" in joint_name:
                joint_id = self.get_joint_id(joint_name)
                lower_limit_pos, upper_limit_pos = p.getJointInfo(self.robot_id, joint_id)[8:10]
                # with some leeway
                min_joint_pos_list.append(lower_limit_pos*1.1)
                max_joint_pos_list.append(upper_limit_pos*1.1)
            elif "wheel" in joint_name:
                min_joint_pos_list.append(-math.pi)
                max_joint_pos_list.append(math.pi)

            min_joint_velo_list.append(-max_speed)
            max_joint_velo_list.append(max_speed)

        min_max_pos = [min_joint_pos_list, max_joint_pos_list]
        min_max_velo = [min_joint_velo_list, max_joint_velo_list]
        return ObservationSpace(self._get_obs(), min_max_pos, min_max_velo)

    def add_debug_params(self):
        print("### ROBOT ID ###")
        max_joint_name_length = 0
        for joint_id in range(p.getNumJoints(self.robot_id)):
            infos = p.getJointInfo(self.robot_id, joint_id)
            joint_name = infos[1].decode('UTF-8')
            max_joint_name_length = max(max_joint_name_length, len(joint_name))

        for joint_id in range(p.getNumJoints(self.robot_id)):
            infos = p.getJointInfo(self.robot_id, joint_id)
            joint_name = infos[1].decode('UTF-8')
            link_name = infos[12].decode('UTF-8')
            print(f"id={infos[0]:<2}  joint_name={joint_name:<{max_joint_name_length}}  link_name={link_name}")
            if ("wheel" in joint_name) or ("motor" in joint_name):
                max_speed = self.max_motor_speed*self.gear_efficiency/self._get_multiplier(joint_name)
                self.controls[joint_name] = p.addUserDebugParameter(joint_name, -max_speed, max_speed, 0)

        print("### PLANE ID ###")
        max_joint_name_length = 0
        for joint_id in range(p.getNumJoints(self.plane_id)):
            infos = p.getJointInfo(self.plane_id, joint_id)
            joint_name = infos[1].decode('UTF-8')
            max_joint_name_length = max(max_joint_name_length, len(joint_name))

        for joint_id in range(p.getNumJoints(self.plane_id)):
            infos = p.getJointInfo(self.plane_id, joint_id)
            joint_name = infos[1].decode('UTF-8')
            link_name = infos[12].decode('UTF-8')
            print(f"id={infos[0]:<2}  joint_name={joint_name:<{max_joint_name_length}}  link_name={link_name}")
            if ("wheel" in joint_name) or ("motor" in joint_name):
                max_speed = self.max_motor_speed * self.gear_efficiency / self._get_multiplier(joint_name)
                self.controls[joint_name] = p.addUserDebugParameter(joint_name, -max_speed, max_speed, 0)

    def add_constraints(self, link_parent_idx, link_child_idx, joint_parent_idx, joint_child_idx):
        """
        Adds point2point constraint to simulate closing loop.
        Closing frame is the frame produced by onshape-to-robot.
        :param link_parent_idx: parent link, look for joint with link parent name
        :param link_child_idx: child link, look for joint with link child name
        :param joint_parent_idx: joint, should be closing_link_name_2
        :param joint_child_idx: joint, should be closing_link_name_1
        :return:
        """
        joint_parent_infos = p.getJointInfo(self.robot_id, joint_parent_idx)
        joint_child_infos = p.getJointInfo(self.robot_id, joint_child_idx)
        axis_idx = 13
        # constraint ID could be saved
        p.createConstraint(self.robot_id,
                           link_parent_idx,
                           self.robot_id,
                           link_child_idx,
                           p.JOINT_POINT2POINT,
                           joint_parent_infos[axis_idx],
                           joint_parent_infos[axis_idx + 1],
                           joint_child_infos[axis_idx + 1])

    def prepare_joints(self):
        # turn off motors for bearings and simulate spring for foot joint
        for joint_id in range(p.getNumJoints(self.robot_id)):
            name = self.get_joint_name(joint_id)
            if (name == "foot_right") or (name == "foot_left"):
                p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, targetPosition=0,
                                        force=self.spring_force)
            else:
                p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    def get_joint_name(self, joint_id):
        return p.getJointInfo(self.robot_id, joint_id)[1].decode('UTF-8')

    def get_joint_id(self, joint_name):
        for joint_id in range(p.getNumJoints(self.robot_id)):
            if self.get_joint_name(joint_id) == joint_name:
                return joint_id

    def _get_controls_ids(self):
        return list(self.controls.values())

    def _get_multiplier(self, name):
        if "wheel" in name:
            return self.gear_reduction_wheel
        return self.gear_reduction

    def _set_camera(self):
        camera_distance = 1.0
        camera_pitch = -45.0
        camera_yaw = -45.0
        pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        camera_target_position = (pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.1)

        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)


if __name__ == "__main__":
    env = Environment("simulation_robot/robot.urdf", render="human", gravity=-9.80665, debug=True)
    env.run()
