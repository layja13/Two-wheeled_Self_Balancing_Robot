import pybullet as p
import time
import pybullet_data
import math
import random
import RL
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

#Obtaining position, linear velocity and angular velocity
def get_robot_state(robot_id, current_pitch):   
    velocity, angular_velocity = p.getBaseVelocity(robot_id)

    target_position = 0
    position_error = current_pitch - target_position  

    return position_error, velocity[0], angular_velocity[1]  


# Pybullet configurations and environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(1./500.)
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,-0.15]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robot_id = robotId = p.loadURDF("/self_balancing_robot.urdf", startPos, startOrientation) # Upload robot

# PID parameters
PID =  [1.7801379, 0.02908636, 1.20059332]
control_output_max = 21.68  # Max value of PID (equal to max speed of motors rad/s)
control_output_min = -21.68  # Min value of PID 
target_pitch = 0 # Setpoint of PID
kp, ki, kd = PID
integral_max = 100  # Against Wind Up
integral_min = -100  # Against Wind Up
integral_error = 0
previous_error = 0


# RL parameters
time_sum = 0
angles_dev = []
threshold_min = 20
seconds = 15
max_time = 500*seconds # Steps to finish episode if robot keeps inside the umbral range angle | 500 = 1 second

# Data vectors
angle_positions = []
translation_positions = []
linear_velocities = []
angle_velocities = []

# Motor limitations
max_torque = 3  
max_velocity =  21.68  # rad/s


# Noise of sensor and delay of controller (steps)
sensor_noise = 2
delay_steps = 5
control_output_buffer = deque(maxlen=delay_steps)

time_check = 0

for i in range(seconds*500):
    # Angle of interest
    _, robot_orientation = p.getBasePositionAndOrientation(robotId)
    robot_euler_angles = p.getEulerFromQuaternion(robot_orientation)
    current_pitch_radians = robot_euler_angles[0]
    current_pitch = math.degrees(current_pitch_radians) + random.uniform(-sensor_noise, sensor_noise)

    # PID computation
    control_output, previous_error_output, integral_error_output  = RL.PID_output(kp, ki, kd, target_pitch, current_pitch, previous_error, integral_error, integral_max,
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
    position_error, linear_velocity, angular_velocity = get_robot_state(robot_id,current_pitch)

    
    angle_positions.append(current_pitch)
    translation_positions.append(position_error)
    linear_velocities.append(linear_velocity)
    angle_velocities.append(angular_velocity)

    # Reinforment Learning 
    if abs(current_pitch) < threshold_min:
        time_sum += 1
        #print("\ntime_sum: ", time_sum)
        reward = RL.reward_function(current_pitch, position_error, linear_velocity, angular_velocity)
        angles_dev.append(current_pitch)

        if time_sum >= max_time:
            reward = RL.reward_function(current_pitch, position_error, linear_velocity, angular_velocity)
    else:
        # Robot falls
        reward = RL.reward_function(current_pitch, position_error, linear_velocity, angular_velocity)


    # Control of robot's motors
    p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=limited_velocity*(-1), force=max_torque)
    p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=limited_velocity*(-1), force=max_torque)


    p.stepSimulation()
    time.sleep(1/500)

# Graphs perfomance
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(angle_positions)
plt.title("Angle Position")

plt.subplot(2, 2, 2)
plt.plot(translation_positions)
plt.title("Translation Position") 

plt.subplot(2, 2, 3)
plt.plot(linear_velocities)
plt.title("Linear Velocity")

plt.subplot(2, 2, 4)
plt.plot(angle_velocities)
plt.title("Angle Position")

plt.tight_layout()
plt.show()

angle_positions = np.array(angle_positions)
error_av = np.mean(angle_positions)
std_deviation = np.std(angle_positions)
print("\nStandard deviation:", std_deviation)
print("\nAverage error:", error_av)


  