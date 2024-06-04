import pybullet as p
import time
import pybullet_data
import math
import random
from collections import deque
import RL
import matplotlib.pyplot as plt
import numpy as np

#Obtaining position, linear velocity and angular velocity
def get_robot_state(robot_id):   
    position, _ = p.getBasePositionAndOrientation(robot_id)
    velocity, angular_velocity = p.getBaseVelocity(robot_id)

    return position[0], velocity[0], angular_velocity[1]

# Pybullet configurations and environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(1./1000.)
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,-0.15]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robot_id = robotId = p.loadURDF("/self_balancing_robot.urdf", startPos, startOrientation) # Upload robot


# CTRNN parameters
# weights, biases and time constant start randomly if None, dt is 0.1 if None by default
weights = np.array([[  -3.51017443,  2.94394757 ],
       [  0.60747822,  2.97098874]])
biases = np.array([[-0.48398066,  0.06634242]])
time_constants = np.array([[1.69312033, 1.01910074]])
dt = 0.01
num_nodes = 2
seconds = 15
max_time = 500*seconds # Steps to finish episode if robot keeps inside the umbral range angle | 500 = 1 second
ctrnn = RL.CTRNN(num_nodes, max_time, weights, biases, time_constants, dt) # CTRNN network

# RL parameters
time_sum = 0
threshold_min = 30
total_reward = 0

# Data vectors
angle_positions = []
translation_positions = []
linear_velocities = []
angle_velocities = []
ctrnn_states_over_time = []
ctrnn_outputs_over_time = []

# Motor limitations
max_torque = 3  
max_velocity =  21.68  # rad/s 

# Noise of sensor and delay of controller (steps)
sensor_noise = 1.5
delay_steps = 5
control_output_buffer = deque(maxlen=delay_steps)

for i in range(seconds*500):
    # Angle of interest
    _, robot_orientation = p.getBasePositionAndOrientation(robotId)
    robot_euler_angles = p.getEulerFromQuaternion(robot_orientation)
    current_pitch_radians = robot_euler_angles[0]
    current_pitch = math.degrees(current_pitch_radians) + random.uniform(-sensor_noise, sensor_noise)

    # CTRNN computation
    CTRNN_states = ctrnn.activate(current_pitch)
    ctrnn_states_over_time.append(ctrnn.get_states().copy())
    CTRNN_output = ctrnn.output_CTRNN()

    CTRNN_output =  CTRNN_output*max_velocity  #Scaling with motor max velocities.

    # Delay in controller
    # Using the output of controller of 2 steps ago, simulating delay
    control_output_buffer.append(CTRNN_output)

    if len(control_output_buffer) == delay_steps:
        delayed_control_output = control_output_buffer[0]
    else:
        delayed_control_output = 0

    position, linear_velocity, angular_velocity = get_robot_state(robot_id)


    angle_positions.append(current_pitch)
    translation_positions.append(position)
    linear_velocities.append(linear_velocity)
    angle_velocities.append(angular_velocity)

    ctrnn_outputs_over_time.append(delayed_control_output)

    # Reinforment Learning 
    if abs(current_pitch) < threshold_min:
        time_sum += 1
        #print("\ntime_sum: ", time_sum)
        reward = RL.reward_function(current_pitch, position, linear_velocity, angular_velocity)        

        if time_sum >= max_time:
            reward = RL.reward_function(current_pitch, position, linear_velocity, angular_velocity)
    else:
        # Robot falls
        reward = RL.reward_function(current_pitch, position, linear_velocity, angular_velocity)

    


    # Control of robot's motors
    p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=delayed_control_output, force=max_torque)
    p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=delayed_control_output, force=max_torque)


    p.stepSimulation()
    time.sleep(1/1000)


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
plt.title("Angle Velocity")
plt.tight_layout()
plt.show()

angle_positions = np.array(angle_positions)
error_av = np.mean(angle_positions)
std_deviation = np.std(angle_positions)
print("\nStandard deviation:", std_deviation)
print("\nAverage error:", error_av)



