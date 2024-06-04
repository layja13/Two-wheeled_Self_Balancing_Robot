import pybullet as p
import pybullet_data
import math
import random
import RL
from collections import deque

# Obtaining position, linear velocity and angular velocity
def get_robot_state(robot_id):   
    position, _ = p.getBasePositionAndOrientation(robot_id)
    velocity, angular_velocity = p.getBaseVelocity(robot_id)

    return position[0], velocity[0], angular_velocity[1]


def simulation_Adaptive_CTRNN_Balancing(individual):
    # Pybullet configurations and environment
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1./500.)
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,-0.15]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    robot_id = robotId = p.loadURDF("/self_balancing_robot.urdf", startPos, startOrientation) # Upload robot

    # CTRNN parameters
    # weights, biases and time constant start randomly if None, dt is 0.1 if None by default
    weights = individual["weights"]
    biases = individual["biases"]
    time_constants = individual["time_constants"]
    dt = 0.1
    num_nodes = 2
    seconds = 15
    iterations = 500*seconds # Steps to finish episode if robot keeps inside the umbral range angle | 500 = 1 second
    ctrnn = RL.CTRNN(num_nodes, iterations, weights, biases, time_constants, dt) # CTRNN network

    # RL parameters
    time_sum = 0
    threshold_min = 30
    seconds = 25
    iterations = 500*seconds # Maximum number of iterations to finish episode if robot keeps inside the umbral 
    total_reward = 0

    # Motor limitations
    max_torque = 3  
    max_velocity =  21.68  # rad/s 


    # Noise of sensor and delay of controller (steps)
    sensor_noise = 1.5
    delay_steps = 5
    control_output_buffer = deque(maxlen=delay_steps)

    while True:
        # Angle of interest
        _, robot_orientation = p.getBasePositionAndOrientation(robotId)
        robot_euler_angles = p.getEulerFromQuaternion(robot_orientation)
        current_pitch_radians = robot_euler_angles[0]
        current_pitch = math.degrees(current_pitch_radians) + random.uniform(-sensor_noise, sensor_noise)

        # CTRNN computation
        CTRNN_states = ctrnn.activate(current_pitch)
        CTRNN_output = ctrnn.output_CTRNN()

        CTRNN_output =  CTRNN_output*max_velocity  #Scaling with motor max velocities.

        # Delay in controller
        # Using the output of controller of 2 steps ago, simulating delay
        control_output_buffer.append(CTRNN_output)

        if len(control_output_buffer) == delay_steps:
            delayed_control_output = control_output_buffer[0]
        else:
            delayed_control_output = 0

        position, linear_velocity, angular_velocity = get_robot_state(robotId)
        
        #print("\nOutput: ", CTRNN_output, "    ||   Current Pitch:", current_pitch)


        # Reinforment Learning 
        if abs(current_pitch) < threshold_min:
            time_sum += 1
            reward = RL.reward_function(current_pitch, position, linear_velocity, angular_velocity)
            total_reward += reward
            #print("\ntime_sum: ", time_sum)
            if time_sum >= iterations:
                p.disconnect()
                return total_reward
        else:
            # Robot falls
            #print("Current Pitch:", current_pitch)
            p.disconnect()
            return total_reward
    

        # Control of robot's motors
        p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=delayed_control_output, force=max_torque)
        p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=delayed_control_output, force=max_torque)


        p.stepSimulation()

        