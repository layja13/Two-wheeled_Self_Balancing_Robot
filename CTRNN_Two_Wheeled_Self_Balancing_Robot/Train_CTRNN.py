import simulation_CTRNN
import RL
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import numpy as np


# Genetic Algorithm parameters
pop_size = 400
num_generations = 200
mutation_rate = 0.1
mutation_weight = 0.1


# Search space
weight_range = [-4, 4]  
bias_range = [-0.9, 0.9]  
tau_range = [0, 10]  
search_space = {
    'weights': weight_range,
    'biases': bias_range,
    'time_constants': tau_range
}

# CTRNN parameters
num_nodes = 2
weights = None
biases = None
time_constants = None
dt = 0.01
seconds = 15
iterations = 500*seconds
ctrnn = RL.CTRNN(num_nodes, iterations, weights, biases, time_constants, dt) # CTRNN network
t = 0

# Parameters 
num_cores = 10
fitness_over_time = []


# Genetic Algorithm
def main():
    global_best_fitness = float(-np.inf)
    best_individual = None
    
    population = RL.init_population_CTRNN(pop_size, num_nodes, search_space)

    for generation in range(num_generations):
        with Pool() as pool:
            fitnesses = pool.map(RL.fitnesses_CTRNN, population)
        

        best = max(fitnesses)
        if best > global_best_fitness:
            global_best_fitness = best
            best_individual = population[fitnesses.index(best)]


        #print("\n fitnesses:", fitnesses)
        print("\n suma fitnesses:", sum(fitnesses))
        print(f"\n Highest fitness generation:" , best)
        print("\n global_best_fitness and individual: ", global_best_fitness,"\n",best_individual)
        print("\n\nGeneration:", generation)
        #print("\n Population:", population)

        fitness_over_time.append(best) 

        args_list = (population, pop_size, fitnesses, search_space, mutation_rate, mutation_weight)
        args_list = [args_list] * pop_size 

        #Selection, crossover, and mutation in parallel
        with Pool() as pool:
            new_population = pool.map(ctrnn.genetic_algorithm_CTRNN, args_list)
        
        population = new_population

    # Choosing best individual in the end
    print("Best solution:", best_individual, "Fitness:",  global_best_fitness)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_finish =  time.time() - time_start 
    print("Time consumed:", time_finish)
    print("Average time consumed by generation:", time_finish/num_generations)
    
    plt.plot(fitness_over_time)
    plt.savefig('plot_PID_evolution.png')
    plt.plot()




   