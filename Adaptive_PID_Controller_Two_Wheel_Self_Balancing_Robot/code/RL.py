import math
import random
import numpy as np
import simulation_Adaptive_PID
import pybullet as p


def reward_function(angle, velocity, angular_velocity):
    angular_velocity_degrees = math.degrees(angular_velocity)

    angle_weight = 2.5
    velocity_weight = 1.5
    angular_velocity_weight = 1

    target_all = 0

    angle_error = target_all - angle
    velocity_error = target_all - velocity
    angular_vel_error = target_all - angular_velocity_degrees
    
    factor_angle = 30
    factor_velocity = 2
    factor_angular_velocity = 115

    angle_reward = angle_weight * (1 - math.sqrt(abs(angle_error)) / math.sqrt(factor_angle))
    velocity_reward = velocity_weight * (1 - math.sqrt(abs(velocity_error)) / math.sqrt(factor_velocity))
    angular_velocity_reward = angular_velocity_weight * (1 - math.sqrt(abs( angular_vel_error)) / math.sqrt(factor_angular_velocity))

    angle_reward = max(angle_reward, 0)
    velocity_reward = max(velocity_reward, 0)
    angular_velocity_reward = max(angular_velocity_reward, 0)

    reward = angle_reward + velocity_reward + angular_velocity_reward
   
    return reward



# PID controller
def PID_output(kp, kd, ki, target_pitch, current_pitch, previous_error, integral_error, integral_max, integral_min, control_output_max, control_output_min):

    error = target_pitch - current_pitch
    derivative_error = error - previous_error
    integral_error += error

    if integral_error > integral_max:
        integral_error = integral_max
    elif integral_error < integral_min:
        integral_error = integral_min

    control_output = kp * error + ki * integral_error + kd * derivative_error
    control_output = max(min(control_output, control_output_max), control_output_min)
    previous_error = error

    return control_output, previous_error, integral_error, derivative_error


def genetic_algorithm(args_list):
    population, pop_size, fitnesses_vector, search_space, mutation_rate = args_list

    index_parents = parents_selection(pop_size, fitnesses_vector)
    parent_1 = population[index_parents[0]]
    parent_2 = population[index_parents[1]]
    daughter = crossover(parent_1, parent_2)
    mutated_daughter = mutation(daughter, search_space, mutation_rate)
    return mutated_daughter


def parents_selection(pop_size, fitnesses):
    ranked_fitness = np.argsort(fitnesses)
    wheel = np.cumsum(range(pop_size))
    max_wheel = np.sum(range(pop_size))

    pick_1 = np.random.rand() * max_wheel
    ind_1 = 0
    while pick_1 > wheel[ind_1]:
        ind_1 += 1

    pick_2 = np.random.rand() * max_wheel
    ind_2 = 0
    while pick_2 > wheel[ind_2]:
        ind_2 += 1

    index_1 = int(ranked_fitness[ind_1])
    index_2 = int(ranked_fitness[ind_2])

    return index_1, index_2


def crossover(ind1, ind2):
    crossover_point = random.randint(1, len(ind1)-1)
    new_ind = np.concatenate([ind1[:crossover_point], ind2[crossover_point:]])
    return new_ind


def mutation(ind, search_space, mutation_rate):
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            if i < 5:
                mutation_step = random.randint(-3, 3)
                ind[i] += mutation_step
            elif i >= 5 and i < 8:
                mutation_step = random.randint(-1, 1)
                ind[i] *=  ind[i] + mutation_step*0.1*ind[i]   # Ajustamos el gen por un porcentaje de su valor actual.
            else:
                mutation_step = random.randint(-3, 3)
                ind[i] += mutation_step

            ind[i] = max(search_space[i][0], min(ind[i], search_space[i][1]))

    return ind


def fitness(ind):
    cost =  simulation_Adaptive_PID.simulation_Adaptive_PID_Balancing(ind)
    return cost



def init_population(pop_size, search_space):
    return [init_individuals(search_space[0], search_space[1], search_space[2], search_space[3],
                             search_space[4], search_space[5], search_space[6],
                             search_space[7], search_space[8], search_space[9], search_space[9]) for _ in range(pop_size)]


def init_individuals(det_range, iet_range, avet_range, devstd_range, devstd_der_range, kp_rate_range ,kd_rate_range ,ki_rate_range, kp_no_range, kd_no_range, ki_no_range): #kp_rate ,kd_rate ,ki_rate kp_range_space, kd_range_space, ki_range_space
    derivative_error_t = random.randint(det_range[0], det_range[1])
    integral_error_t = random.randint(iet_range[0], iet_range[1])
    average_error_t = random.randint(avet_range[0], avet_range[1])
    dev_std_t = random.randint(devstd_range[0], devstd_range[1])
    dev_der_err_std_t = random.randint(devstd_der_range[0], devstd_der_range[1])
    kp_rate = random.uniform(kp_rate_range[0], kp_rate_range[1])
    kd_rate = random.uniform(kd_rate_range[0], kd_rate_range[1])
    ki_rate = random.uniform(ki_rate_range[0], ki_rate_range[1])
    kp_no = random.randint(kp_no_range[0], kp_no_range[1])
    kd_no = random.randint(kd_no_range[0], kd_no_range[1])
    ki_no = random.randint(ki_no_range[0], ki_no_range[1])

    return [derivative_error_t, integral_error_t, average_error_t, dev_std_t,dev_der_err_std_t,
            kp_rate, kd_rate, ki_rate, kp_no, kd_no, ki_no] # Notice that this returns a vector


def adjust_pid_gains(derivative_error, integral_error, average_error, std_dev_error, std_dev_der_error,
                      derivative_threshold, integral_threshold, average_error_threshold, std_dev_threshold, std_dev_der_error_threshold,
                      kp_change_rate, kd_change_rate, ki_change_rate,
                      kp_no, kd_no, ki_no,
                      kp, kd, ki):
        
    if average_error > kp_no and abs(derivative_error) > kd_no and integral_error > ki_no:
        pass
    else:
        if average_error > kp_no:
            if average_error > average_error_threshold or std_dev_error > std_dev_threshold: 
                kp *= kp_change_rate  
            else:
                kp /= kp_change_rate
        else:
            pass

        if abs(derivative_error) > kd_no:
            if  abs(derivative_error) > derivative_threshold or std_dev_der_error > std_dev_der_error_threshold:
                kd *= kd_change_rate
            else:
                kd /= kd_change_rate
        else: 
            pass
            
        if integral_error > ki_no:
            if integral_error > integral_threshold:
                    ki *= ki_change_rate
            else:
                ki /= ki_change_rate
        else:
            pass

    kp = max(1, min(kp, 100))
    ki = max(0.1, min(ki, 150))
    kd = max(0.05, min(kd, 10))

    return kp, kd, ki


def scenario_detection(robot_id):
    _, orientacion = p.getBasePositionAndOrientation(robot_id)
    orientacion_euler = p.getEulerFromQuaternion(orientacion)
    thresh = 0.3 
    if abs(orientacion_euler[1]) > thresh:  
        return "inclination"
    
    _, vel_angular = p.getBaseVelocity(robot_id)
    thresh_impact = 5  
    if np.linalg.norm(vel_angular) > thresh_impact:
        return "impact"

    return False


def gain_scheduling(scenario):
    PID_gains = {
    "impact": {"kp": 100.424, "ki": 0.180, "kd": 135.80782592},
    "inclination": {"kp": 68.56406275 , "ki":1.30893118  , "kd":125.72503309}
    }

    return PID_gains[scenario]

    



        
