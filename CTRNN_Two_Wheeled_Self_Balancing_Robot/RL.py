import random
import numpy as np
import math 
import simulation_PID
import simulation_CTRNN


def fitnesses_CTRNN(individual):
        cost = simulation_CTRNN.simulation_Adaptive_CTRNN_Balancing(individual)  
        return cost


class CTRNN:
    def __init__(self, size, iterations, weights=None, biases=None, time_constants=None, dt=None):
        self.size = size
        self.weights = weights if weights is not None else np.random.normal(size=(size, size)) * 0.2
        self.biases = biases if biases is not None else np.zeros((1, size))
        self.time_constants = time_constants if time_constants is not None else 1 / (0.1 + (10 * np.random.random((1, size))))
        self.dt = dt if dt is not None else 0.1
        self.states = np.zeros((size, iterations))
 
        
    def activate(self, external_input):
        delta = -self.states[:, 0] + np.tanh(np.dot(self.weights, self.states[:, 0]) + self.biases + external_input)
        self.states[:,1] = self.states[:, 0] + self.dt * np.multiply(delta, self.time_constants)
        self.states[:,0] = self.states[:,1]
        return self.states[:, 1] + self.dt * np.multiply(delta, self.time_constants)
    

    def output_CTRNN(self):
        average_state = np.mean(self.states[:, 1])
        return average_state
    
    def update_weights(self, weights):
        self.weights = weights
    
    def get_states(self):
        return self.states


    def mutate_CTRNN(self, individual, mutation_rate, mutation_weight, search_space):
        if np.random.rand() < mutation_rate:
            individual['weights'] += np.random.normal(0, mutation_weight, size=individual['weights'].shape)
            np.clip(individual['weights'], search_space["weights"][0], search_space["weights"][1], out=individual['weights'])

        if np.random.rand() < mutation_rate:
            individual['biases'] += np.random.normal(0, mutation_weight, size=individual['biases'].shape)
            np.clip(individual['biases'], search_space["biases"][0], search_space["biases"][1], out=individual['biases'])
            
        if np.random.rand() < mutation_rate:
            individual['time_constants'] += np.random.normal(0, mutation_weight, size=individual['time_constants'].shape)
            np.clip(individual['time_constants'], search_space["time_constants"][0], search_space["time_constants"][1], out=individual['time_constants'])
        
        return individual
    
    
    def crossover_CTRNN(self, ind1, ind2):
        crossover_point_weights = np.random.randint(1, ind1['weights'].shape[0])
        crossover_point_biases = np.random.randint(1, ind1['biases'].shape[1])
        crossover_point_time_constant = np.random.randint(1, ind1['time_constants'].shape[1])
        
        new_weights = np.concatenate((ind1['weights'][:crossover_point_weights, :],
                                    ind2['weights'][crossover_point_weights:, :]))
        
        new_biases = np.concatenate((ind1['biases'][:, :crossover_point_biases],
                                    ind2['biases'][:, crossover_point_biases:]), axis=1)
        
        new_inv_tau = np.concatenate((ind1['time_constants'][:, :crossover_point_time_constant],
                                    ind2['time_constants'][:, crossover_point_time_constant:]), axis=1)
        
        return {'weights': new_weights, 'biases': new_biases, 'time_constants': new_inv_tau}



    def genetic_algorithm_CTRNN(self, args_list):
        population, pop_size, fitnesses_vector, search_space, mutation_rate, mutation_weight = args_list 

        index_parents = parents_selection(pop_size, fitnesses_vector)
        parent_1 = population[index_parents[0]]
        parent_2 = population[index_parents[1]]
        daughter = self.crossover_CTRNN(parent_1, parent_2)
        mutated_daughter = self.mutate_CTRNN(daughter, mutation_rate, mutation_weight, search_space)
        return mutated_daughter



def init_individual_CTRNN(size, search_space):
    weights_range = search_space["weights"]
    biases_range = search_space["biases"]
    time_constants_range = search_space["time_constants"]

    weights = np.random.uniform(weights_range[0], weights_range[1], size=(size, size))
    biases = np.random.uniform(biases_range[0], biases_range[1], (1, size))
    time_constants = 1 / np.random.uniform(time_constants_range[0], time_constants_range[1], (1, size))

    return {"weights":weights, "biases": biases, "time_constants":time_constants }


def init_population_CTRNN(pop_size, size, search_space):
    population = [init_individual_CTRNN(size, search_space) for _ in range(pop_size)]
    return population



def reward_function(angle, position, velocity, angular_velocity):
    angular_velocity_degrees = math.degrees(angular_velocity)

    angle_weight = 2.5
    position_weight = 2
    velocity_weight = 0.7
    angular_velocity_weight = 0.7

    target_all = 0

    angle_error = target_all - angle
    position_error = target_all - position
    velocity_error = target_all - velocity
    angular_vel_error = target_all - angular_velocity_degrees
    
    factor_angle = 30
    factor_position =2
    factor_velocity = 2
    factor_angular_velocity = 115

    angle_reward = angle_weight * (1 - math.sqrt(abs(angle_error)) / math.sqrt(factor_angle))
    position_reward = position_weight * (1 - math.sqrt(abs(position_error)) / math.sqrt(factor_position))
    velocity_reward = velocity_weight * (1 - math.sqrt(abs(velocity_error)) / math.sqrt(factor_velocity))
    angular_velocity_reward = angular_velocity_weight * (1 - math.sqrt(abs( angular_vel_error)) / math.sqrt(factor_angular_velocity))

    angle_reward = max(angle_reward, 0)
    position_reward = max(position_reward, 0)
    velocity_reward = max(velocity_reward, 0)
    angular_velocity_reward = max(angular_velocity_reward, 0)

    reward = angle_reward + position_reward + velocity_reward + angular_velocity_reward
   
    return reward


# Genetic Algorithm
# It also has Microbial GA which I couldn't use

def genetic_algorithm_PID(args_list):
    population, pop_size, fitnesses_vector, search_space, mutation_rate = args_list 

    index_parents = parents_selection(pop_size,  fitnesses_vector)
    parent_1 = population[index_parents[0]]
    parent_2 = population[index_parents[1]]
    daughter = crossover_PID(parent_1, parent_2)
    mutated_daughter = mutation_PID(daughter, search_space, mutation_rate)
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


def crossover_PID(ind1, ind2):
    crossover_point = random.randint(1, len(ind1)-1)
    new_ind = np.concatenate([ind1[:crossover_point], ind2[crossover_point:]])
    return new_ind


def mutation_PID(ind, search_space, mutation_rate):
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            mutation_amount = 0.1 * ind[i]
            ind[i] += mutation_amount if random.random() < 0.5 else -mutation_amount
    return ind


def fitnesses_PID(ind):
    cost = simulation_PID.simulation_Adaptive_PID_Balancing(ind)  
    return cost


# PID controller
def PID_output(kp, ki, kd, target_pitch, current_pitch, previous_error, integral_error, integral_max, integral_min, control_output_max, control_output_min):
    
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

    return control_output, previous_error, integral_error


def init_population_PID(pop_size, search_space):
    kp_range, kd_range, ki_range = search_space
    return [init_individuals_PID(kp_range, kd_range, ki_range) for _ in range(pop_size)]


def init_individuals_PID(kp_range, kd_range, ki_range):
    kp = random.randint(kp_range[0], kp_range[1])  # Range for kp
    ki = random.uniform(ki_range[0], ki_range[1])  # Range for ki
    kd = random.uniform(kd_range[0], kd_range[1])   # Range for kd
    return [kp, kd, ki] # Notice that this returns a vector


def tournament_selection(population, fitnesses):
    P = len(population)
    index1 = random.randint(0, P-1)
    index2 = random.randint(0, P-1)

    P1 = population[index1]
    P2 = population[index2]

    if fitnesses[index1] > fitnesses[index2]:
        Winner = P1
        Loser = P2
    else:
        Winner = P2
        Loser = P1

    return Winner, Loser, index1, index2