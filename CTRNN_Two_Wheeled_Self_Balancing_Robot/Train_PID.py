import simulation_PID 
import RL
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import numpy as np


# Genetic Algorithm parameters
pop_size = 400
num_generations = 400
num_genes = 3  # kp, ki, kd
mutation_rate = 0.1


# Search space
kp_space_search = [10, 40]
kd_space_search = [0, 2] 
ki_space_search = [15,80]
search_space = [kp_space_search, kd_space_search, ki_space_search]
num_cores = 10

fitness_over_time = []


# Genetic Algorithm
def main():
    global_best_fitness = float(-np.inf)
    best_individual = None
    population = RL.init_population_PID(pop_size, search_space)  # Search space range for each component of PID kp,kd,ki

    for generation in range(num_generations):
        with Pool(num_cores) as pool:
            fitnesses = pool.map(RL.fitnesses_PID, population)
        

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

        new_population = np.zeros((pop_size, num_genes), dtype=float)
        args_list = (population, pop_size, fitnesses, search_space, mutation_rate)
        args_list = [args_list] * pop_size 


        #Selection, crossover, and mutation in parallel
        with Pool(num_cores) as pool:
            new_population = pool.map(RL.genetic_algorithm_PID, args_list)
        
        population = np.array(new_population)

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




   