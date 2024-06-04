import RL
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import numpy as np

# Genetic Algorithm parameters
pop_size = 250
num_generations = 300
num_genes = 10
mutation_rate = 0.12

# Search space
derivative_error_thresh_range = (0, 25)
integral_error_thresh_range = (0, 25)
average_error_range = (0, 25)
dev_std_thresh_range = (0, 25)
std_dev_av_der_error_thresh_range = (0, 25)
kp_rate_change_range = (1, 1.20)
kd_rate_change_range = (1, 1.20)
ki_rate_change_range = (1, 1.20)
kp_no_range = (0, 20)
kd_no_range = (0, 50)
ki_no_range = (0, 60)
search_space = [derivative_error_thresh_range, integral_error_thresh_range, average_error_range, dev_std_thresh_range, std_dev_av_der_error_thresh_range, kp_rate_change_range, kd_rate_change_range, ki_rate_change_range, kp_no_range, kd_no_range, ki_no_range] 

# Data
num_cores = 10

# Genetic Algorithm
def main():
    global_best_fitness = float(-np.inf)
    best_individual = None
    population = RL.init_population(pop_size, search_space)  # Search space range for each component of PID kp,kd,ki

    for generation in range(num_generations):
        with Pool(num_cores) as pool:
            fitnesses = pool.map(RL.fitness, population)
        

        best = max(fitnesses)
        if best > global_best_fitness:
            global_best_fitness = best
            best_individual = population[fitnesses.index(best)]


        print(f"\n Highest fitness generation:" , best)
        print("\n global_best_fitness and individual: ", global_best_fitness,"\n",best_individual)
        print("\n\nGeneration:", generation)

        #fitness_over_time.append(best) 
        with open("fitness_data.txt", "a") as f:
                f.write(f"{generation},{best}\n")

        args_list = (population, pop_size, fitnesses, search_space, mutation_rate)
        args_list = [args_list] * pop_size 


        #Selection, crossover, and mutation in parallel
        with Pool(num_cores ) as pool:
            new_population = pool.map(RL.genetic_algorithm, args_list)
        
        population = np.array(new_population)


    # Choosing best individual in the end
    print("Best solution:", best_individual, "Fitness:",  global_best_fitness)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_finish =  time.time() - time_start 
    print("Time consumed:", time_finish)
    print("Average time consumed by generation:", time_finish/num_generations)
    data = np.loadtxt("fitness_data.txt", delimiter=",")
    plt.plot(data[:,0], data[:,1])
    plt.title('Fitness evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig('fitness_evolution.png')
    plt.show()
