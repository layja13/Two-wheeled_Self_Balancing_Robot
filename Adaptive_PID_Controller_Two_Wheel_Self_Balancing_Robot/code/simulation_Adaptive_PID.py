import pybullet as p
import pybullet_data
import math
import random
import RL
from collections import deque
import numpy as np
import time

# Obtaining position, linear velocity and angular velocity
def get_robot_state(robot_id, current_pitch):
    velocity, angular_velocity = p.getBaseVelocity(robot_id)

    target_position = 0
    position_error = current_pitch - target_position

    return position_error, velocity[0], angular_velocity[1]

def simulation_Adaptive_PID_Balancing(Adaptive_mechanism):
    # Pybullet configurations and environment
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1./500.)
    p.setGravity(0, 0, -9.81)    
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,-0.15]
    startOrientation = p.getQuaternionFromEuler([0,0,-0.15])
    robot_id = p.loadURDF("/self_balancing_robot.urdf", startPos, startOrientation) # Upload robot

    
    # PID parameters
    PID =  [10, 2, 10]
    derivative_threshold, integral_threshold, average_error_threshold, std_dev_threshold, std_dev_av_der_error_threshold, kp_rate_change_range, kd_rate_change_range, ki_rate_change_range, kp_no, kd_no, ki_no = Adaptive_mechanism 
    control_output_max = 21.68 # Max value of PID (equal to max speed of motors rad/s)
    control_output_min = -21.68  # Min value of PID
    target_pitch = 0 # Setpoint of PID
    kp, ki, kd = PID
    integral_max = 100  # Against Wind Up
    integral_min = -100  # Against Wind Up
    integral_error = 0
    previous_error = 0
    angles = deque(maxlen=100)
    std_dev_der_error = deque(maxlen=100)

    # Fitness/reward function parameters
    time_sum = 0
    threshold_min = 20
    seconds = 15
    max_time = 500*seconds # Steps to finish episode if robot keeps inside the umbral range angle | 500 = 1 second
    total_reward = 0

    # Motor limitations
    max_torque = 3
    max_velocity =  21.68  # rad/s

    # Noise of sensor and delay of controller (steps)
    sensor_noise = 1.5
    delay_steps = 5
    control_output_buffer = deque(maxlen=delay_steps)

    start_time = time.time()  # Inicio del tiempo para controlar cuándo aplicar la fuerza
    perturbation_applied = False 

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Perturbation
        if elapsed_time > 5 and not perturbation_applied:
            force_direction = [10, 0, 0]  
            p.applyExternalForce(objectUniqueId=robot_id, linkIndex=-1, forceObj=force_direction, posObj=[0,0,0], flags=p.WORLD_FRAME)
            perturbation_applied = True

        # Angle of interest
        _, robot_orientation = p.getBasePositionAndOrientation(robot_id)
        robot_euler_angles = p.getEulerFromQuaternion(robot_orientation)
        current_pitch_radians = robot_euler_angles[0]
        current_pitch = math.degrees(current_pitch_radians) + random.uniform(-sensor_noise, sensor_noise)

        # PID computation
        control_output, previous_error_output, integral_error_output, derivative_error  = RL.PID_output(kp, ki, kd, target_pitch, current_pitch, previous_error, integral_error, integral_max,
                                                                            integral_min, control_output_max, control_output_min)

        previous_error = previous_error_output
        integral_error = integral_error_output

        # Delay in controller
        # Using the output of controller of 2 steps ago, simulating delay
        control_output_buffer.append(control_output)

        if len(control_output_buffer) == delay_steps:
            delayed_control_output = control_output_buffer[0]
        else:
            delayed_control_output = 0


        # Limitation velocity in motors
        limited_velocity = max(min(delayed_control_output, max_velocity), -max_velocity)
        position, linear_velocity, angular_velocity = get_robot_state(robot_id, current_pitch)

        angles.append(current_pitch)
        av_error = np.mean(angles)
        dev_av_error = np.std(angles)

        std_dev_der_error.append(derivative_error)
        std_dev_av_der_error = np.std(std_dev_der_error)

       
        # Fitness Function
        if abs(current_pitch) < threshold_min:
            time_sum += 1
            reward = RL.reward_function(position, linear_velocity, angular_velocity)
            total_reward += reward
            if time_sum >= max_time:
                p.disconnect()
                return total_reward
        else:
            # Robot falls
            p.disconnect()
            return total_reward
        
      

        kp_n, kd_n, ki_n = RL.adjust_pid_gains(derivative_error, integral_error, av_error, dev_av_error, std_dev_av_der_error,
                    derivative_threshold, integral_threshold, average_error_threshold, std_dev_threshold, std_dev_av_der_error_threshold,
                    kp_rate_change_range, kd_rate_change_range, ki_rate_change_range,
                    kp_no, kd_no, ki_no,
                    kp, kd, ki)

        kp, kd, ki = kp_n, kd_n, ki_n

   


        # Control of robot's motors
        p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=limited_velocity*(-1), force=max_torque)
        p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=limited_velocity*(-1), force=max_torque)


        p.stepSimulation()


