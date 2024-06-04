import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("fitness_data.txt", delimiter=",")
plt.plot(data[:,0], data[:,1])
plt.title('Fitness evolution')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig('fitness_evolution.png')
plt.show()