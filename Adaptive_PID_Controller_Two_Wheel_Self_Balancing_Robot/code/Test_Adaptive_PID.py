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

scenario_1 = True
scenario_2 = False
scenario_3 = False
scenario_4 = False
if scenario_1:
    startPos = [0,0,-0.15]
    # Scenario 1 (standard) 
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    robot_id = p.loadURDF("self_balancing_robot.urdf", startPos, startOrientation) # Upload robot


if scenario_2:
# Scenario 2 (Inclination) 
    startPos = [0,0,0.1]
    startOrientation = p.getQuaternionFromEuler([0,0,80]) # Position of the robot
    robot_id = p.loadURDF("self_balancing_robot.urdf", startPos, startOrientation) # Upload robot

    # Inclination Ramp
    rampOrientation = p.getQuaternionFromEuler([0, 0.3, 0])  # Ajusta el segundo valor para cambiar la inclinación
    rampCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[1.5, 0.5, 0.05])
    rampVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[1.5, 0.5, 0.05], rgbaColor=[1, 0, 0, 1])
    rampId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rampCollisionShapeId, baseVisualShapeIndex=rampVisualShapeId, basePosition=[0.5, 0, 0.025], baseOrientation=rampOrientation)
    frict = 3
    p.changeDynamics(rampId, -1, lateralFriction=frict)

if scenario_3:
    # Scenario 3
    startPos = [0,0,-0.15]
    startOrientation = p.getQuaternionFromEuler([0,0,0]) # Position of the robot
    robot_id = p.loadURDF("self_balancing_robot.urdf", startPos, startOrientation) # Upload robot
    start_time = time.time()  
    object_added_1 = False

if scenario_4:
    # Scenario 4
    startPos = [0,0,-0.15]
    startOrientation = p.getQuaternionFromEuler([0,0,0]) # Position of the robot
    robot_id = p.loadURDF("self_balancing_robot.urdf", startPos, startOrientation) # Upload robot
    start_time = time.time()  
    object_added_2 = False

# PID parameters
PID =  [1, 1, 1]
control_output_max = 21.68  # Max value of PID (equal to max speed of motors rad/s)
control_output_min = -21.68  # Min value of PID 
target_pitch = 0 # Setpoint of PID
kp, kd, ki = PID
integral_max = 100  # Against Wind Up
integral_min = -100  # Against Wind Up
integral_error = 0
previous_error = 0
kp_values = []
ki_values = []
kd_values = []


# PID gains adjustment mechanism parameters     
Adaptive_mechanism =  [21, 20, 18, 24, 10, 1.2, 1.2, 1.1, 5, 0, 4]

#[21.  20.  18.  24.  10.   1.2  1.2  1.1  0.   0.   4. ] genetic algorithm best_report

# [4,        25,       4,       16,        1.05625869, 1.1276059, 1.2, 15,   50,    0 ] manual fine tuning
#[4,        25,       4,       16,        1.05625869, 1.1276059, 1.2, 22,   50,    0 ] genetic algorithm


derivative_threshold, integral_threshold, average_error_threshold, std_dev_threshold, std_dev_av_der_error_threshold,  kp_change_rate, kd_change_rate, ki_change_rate, kp_no, kd_no, ki_no = Adaptive_mechanism
angles = deque(maxlen=75)
std_dev_der_error = deque(maxlen=100)


# Fitness/reward function parameters
time_sum = 0
threshold_min = 20
seconds = 7
max_time = 500*seconds # Steps to finish episode if robot keeps inside the umbral range angle | 500 = 1 second

# Data vectors
angle_positions = []
linear_velocities = []
angle_velocities = []

# Motor limitations
max_torque = 3  
max_velocity =  21.68  # rad/s

# Noise of sensor and delay of controller (steps)
sensor_noise = 1.5
delay_steps = 5
control_output_buffer = deque(maxlen=delay_steps)
step= 0


start_time = time.time()  
object_added_1 = False 
object_added_2 = False 
flag= False 


for i in range(seconds*500):

    # Angle of interest
    robot_pos, robot_orientation = p.getBasePositionAndOrientation(robot_id)
    robot_euler_angles = p.getEulerFromQuaternion(robot_orientation)
    current_pitch_radians = robot_euler_angles[0]
    current_pitch = math.degrees(current_pitch_radians) + random.uniform(-sensor_noise, sensor_noise)

    # Object falls and hits the robot after 5 secs -----------------------------------------------------
    current_time = time.time()
    elapsed_time = current_time - start_time

    if scenario_3 and elapsed_time > 5 and not object_added_1:
    # Object falls towards robot -----------------------------------------------------
        # Posición del robot en este momento
        robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
        # Position of the object is above of the robot
        obj_start_pos = [robot_pos[0], robot_pos[1]- 0.1, robot_pos[2] + 2]
        # Creation of the object
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        multi_body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_id, baseVisualShapeIndex=visual_shape_id, basePosition=obj_start_pos)
        object_added_1 = True  #
    

    if scenario_4 and elapsed_time > 2 and not object_added_2 :
        # Object horizontally towards to the robot
        obj_start_pos = [robot_pos[0], robot_pos[1] - 1, robot_pos[2]]
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        multi_body_id = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=box_id, baseVisualShapeIndex=visual_shape_id, basePosition=obj_start_pos)
        object_added_2 = True
        flag = True
    if flag:
        force_vector = [0, 0.4, 9.81*0.5]  # Ajusta este vector para cambiar la dirección y magnitud de la fuerza
        p.applyExternalForce(objectUniqueId=multi_body_id, linkIndex=-1, forceObj=force_vector, posObj=[0,0,0], flags=p.LINK_FRAME)
    # --------------------------------------------------------------------------------------------------

    # PID computation
    control_output, previous_error_output, integral_error_output, derivative_error  = RL.PID_output(kp, kd, ki, target_pitch, current_pitch, previous_error, integral_error, integral_max,
                                                                        integral_min, control_output_max, control_output_min)

    previous_error = previous_error_output
    integral_error = integral_error_output

    kp_values.append(kp)
    ki_values.append(ki)
    kd_values.append(kd)
    
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
    linear_velocities.append(linear_velocity)
    angle_velocities.append(angular_velocity)

    angles.append(current_pitch)
    av_error = np.mean(angles)
    dev_av_error = np.std(angles)
    std_dev_av_der_error = np.std(std_dev_der_error)


    # Scenario Detection for Gain Scheduling
    scenario = RL.scenario_detection(robot_id)

    # If not scenario detected, then Adaptive Gain process
    if not scenario:
        kp_n, kd_n, ki_n = RL.adjust_pid_gains(derivative_error, integral_error, av_error, dev_av_error, std_dev_av_der_error,
                    derivative_threshold, integral_threshold, average_error_threshold, std_dev_threshold, std_dev_av_der_error_threshold,
                    kp_change_rate, kd_change_rate, ki_change_rate,
                    kp_no, kd_no, ki_no,
                    kp, kd, ki)

        kp, kd, ki = kp_n, kd_n, ki_n

    # If scenario detected, switch PID controller gains
    else:
        pid = RL.gain_scheduling(scenario)
        kp, ki, kd = pid["kp"], pid["ki"], pid["kd"]

    print("kp: ", kp, "| kd:", kd, "| ki: ", ki)


    # Control of robot's motors
    p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=limited_velocity*(-1), force=max_torque)
    p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=limited_velocity*(-1), force=max_torque)


    p.stepSimulation()
    time.sleep(1/500)

# Graphs perfomance
plt.figure(figsize=(10, 8))
plt.subplot(3, 3, 1)
plt.plot(angle_positions)
plt.title("Angle Position")

plt.subplot(3, 3, 2)
plt.plot(linear_velocities, )
plt.title("Linear Velocity")

plt.subplot(3, 3, 3)
plt.plot(angle_velocities)
plt.title("Angular Velocity")

plt.subplot(3, 3, 4)
plt.plot(kp_values, label='Kp')
plt.title("Proportional Gain (Kp)")

plt.subplot(3, 3, 5)
plt.plot(kd_values, label='Kd')
plt.title("Derivative Gain (Kd)")

plt.subplot(3, 3, 6)
plt.plot(ki_values, label='Ki')
plt.title("Integral Gain (Ki)")


plt.tight_layout()
plt.show()

angle_positions = np.array(angle_positions)
error_av = np.mean(angle_positions)
std_deviation = np.std(angle_positions)
print("\nStandard deviation:", std_deviation)
print("\nAverage error:", error_av)


  