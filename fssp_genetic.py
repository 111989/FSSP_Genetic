""" Genetic Algorithm implementation for 
    solving the Flow Shop Scheduling problem """

import argparse
import numpy as np
import pandas as pd
import copy
import time
from statistics import median

class Population:
    def __init__(self, population_size: int, n_jobs: int) -> None:

        self.population_size = population_size
        self.n_jobs = n_jobs
        self.population = []
        self.generate_initial_population()
        self.parents = []
        self.successors = []

    # Population Initialization: Random Initialization
    # Generates 'population_size' number of random per
    # -mutations of jobs to create an initial population. 
    # These random permutations are called 'chromosomes,' 
    # each of which is of length 'n_jobs.'
    def generate_initial_population(self) -> None:

        count = 0
        while count < self.population_size:
            chromosome = random_permutation(self.n_jobs)
            if chromosome not in self.population:
                self.population.append(chromosome)
                count += 1

    # genetic operator
    # Crossover Operator: Uniform Crossover
    # Here, we essentially flip an unbiased 
    # coin to decide whether or not genetic
    # material from parents must be included
    # in the successor.
    def crossover(self, crossover_rate: float) -> None:

        self.successors = copy.deepcopy(self.parents)
        shuffled_chromosomes = random_permutation(population_size)
        bias = self.n_jobs//2
        for i in range(population_size//2):
            crossover_probability = np.random.rand()
            if crossover_probability <= crossover_rate:
                random_parent_1 = self.parents[shuffled_chromosomes[2*i]][:]
                random_parent_2 = self.parents[shuffled_chromosomes[2*i+1]][:]
                successor_1 = [None for _ in range(self.n_jobs)]
                successor_2 = [None for _ in range(self.n_jobs)]
                shuffled_genes = random_permutation(self.n_jobs)
                for j in range(bias):
                    successor_1[shuffled_genes[j]] = random_parent_1[shuffled_genes[j]]
                    successor_2[shuffled_genes[j]] = random_parent_2[shuffled_genes[j]]
                
                s1 = [random_parent_2[j] for j in range(self.n_jobs) if random_parent_2[j] not in successor_1]
                s2 = [random_parent_1[j] for j in range(self.n_jobs) if random_parent_1[j] not in successor_2]
                for j in range(self.n_jobs - bias):
                    successor_1[successor_1.index(None)] = s1[j]
                    successor_2[successor_2.index(None)] = s2[j]
                
                self.successors[shuffled_chromosomes[2*i]] = successor_1[:]
                self.successors[shuffled_chromosomes[2*i+1]] = successor_2[:]

    # genetic operator
    # Applies random changes to parents to form children; 
    # randomly shuffles genes. 'mutation_selection_rate' 
    # controls the standard deviation of the mutation at 
    # the first generation; 'n_jobs' is the range of the 
    # initial population.
    def mutation(self, mutation_rate: float, mutation_selection_rate: float) -> None:

        n_mutation_jobs = round(self.n_jobs * mutation_selection_rate)
        for i in range(len(self.successors)):
            mutation_probability = np.random.rand()
            if mutation_probability <= mutation_rate:
                random_choice_ = random_choice(self.n_jobs, n_mutation_jobs)
                temp_allele = self.successors[i][random_choice_[0]]
                for j in range(n_mutation_jobs-1):
                    self.successors[i][random_choice_[j]] = self.successors[i][random_choice_[j+1]] 
                self.successors[i][random_choice_[n_mutation_jobs-1]] = temp_allele

    # genetic operator
    # Fitness based selection; selects parents that 
    # contribute to the population at the next generation.
    def selection(self, total_chromosomes: list, chromosome_fitness: list) -> None:   
        median_fitness = median(chromosome_fitness)
        self.population = []
        for index in range(2*self.population_size):
            if chromosome_fitness[index] >= median_fitness:
                self.population.append(total_chromosomes[index])
        
        

def random_permutation(x: int) -> list:
    return list(np.random.permutation(x))

def random_choice(a: int, size: int) -> list:
    return list(np.random.choice(a = a, size = size, replace = False))



# Calculates machining idle times for 'n_jobs' 
# number of jobs and 'n_machines' number of machines. 
# Time complexity: O(nm), n = n_jobs, m = n_machines
# 'd' stores the sum of the idle times preceded by 
# the corresponding job, 'v' indicates the sum of 
# idle times.
def algorithm(matrix: list, total_chromosomes: list, population_size: int, n_jobs: int, n_machines: int):

    # memory initializations
    chromosome_fitness, chromosome_fit = [], []
    s, d = [[0]*n_machines]*(2*population_size), [[0]*n_machines]*(2*population_size)
    D = [[[0]*n_jobs]*n_machines]*(2*population_size)

    # begin
    v = [0]*(2*population_size)
    for k in range(2*population_size):
        for i in range(n_machines):
            s[k][i] = matrix[total_chromosomes[k][0]][i]
            d[k][i] = v[k]
            D[k][i][total_chromosomes[k][0]] = v[k]
            v[k] += matrix[total_chromosomes[k][0]][i]

        for j in range(n_jobs): D[k][0][j] = 0
    
        for j in range(1, n_jobs):
            for i in range(n_machines-1):
                s[k][i] += matrix[total_chromosomes[k][j]][i]
                D[k][i+1][j] = max(0, s[k][i] + d[k][i] - s[k][i+1] - d[k][i+1])
                d[k][i+1] += D[k][i+1][j]
            s[k][n_machines-1] += matrix[total_chromosomes[k][j]][i+1]
        
        v[k] = 0
        for i in range(n_machines): v[k] += d[k][i]
        chromosome_fitness.append(1/v[k])
        chromosome_fit.append(v[k])

    return chromosome_fitness, chromosome_fit
        


def comparison(total_chromosomes: list, chromosome_fit: list, population_size: int, temp_optimal_value: float):
    optimal_value = float('inf')
    for i in range(2*population_size):
        if chromosome_fit[i] < optimal_value:
            optimal_sequence = copy.deepcopy(total_chromosomes[i])
            optimal_value = chromosome_fit[i]
    if optimal_value <= temp_optimal_value:
        temp_optimal_value = optimal_value
    return optimal_sequence, optimal_value



def main():
    
    # generate initial population
    pop = Population(population_size, n_jobs)
    temp_optimal_value = float('inf')
    start_time = time.time()

    # parent selection
    # crossover with probability 'crossover_rate'
    # mutation with probability 'mutation_rate'
    # Combine parent and successor chromosomes to get 
    # 'total_chromosomes,' which comprises of job and
    # /or machining sequence permutation vectors.
    # fitness calculation
    # survivor selection
    # comparison
    for _ in range(n_iterations):     

        pop.parents = copy.deepcopy(pop.population)
        pop.crossover(crossover_rate)
        pop.mutation(mutation_rate, mutation_selection_rate)
        total_chromosomes = copy.deepcopy(pop.parents) + copy.deepcopy(pop.successors) 
        chromosome_fitness, chromosome_fit = algorithm(matrix, total_chromosomes, population_size, n_jobs, n_machines)
        pop.selection(total_chromosomes, chromosome_fitness)
        optimal_sequence, optimal_value = comparison(total_chromosomes, chromosome_fit, population_size, temp_optimal_value)
        
    print('Optimal sequence: ', optimal_sequence)
    print('Optimal value: {}'.format(optimal_value))
    print('Time elapsed: {}'.format(time.time() - start_time))



if __name__ == '__main__':

    # fetch dataset    
    dataframe = pd.read_excel(io = '20x5_flow_shop_dataset.xlsx', sheet_name = 'S1', index_col = [0])
    matrix = dataframe.values.tolist()
    n_jobs, n_machines = len(matrix), len(matrix[0])

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', type = int, default = 20, help = 'Population size')
    parser.add_argument('--crossover', type = float, default = 0.8, help = 'Crossover rate')
    parser.add_argument('--mutation', type = float, default = 0.2, help = 'Mutation rate')
    parser.add_argument('--selection', type = float, default = 0.2, help = 'Mutation selection rate')
    parser.add_argument('--iterations', type = int, default = 1000, help = 'Number of iterations')
    
    args = vars(parser.parse_args())
    population_size = int(args['population'])
    crossover_rate = float(args['crossover'])
    mutation_rate = float(args['mutation'])
    mutation_selection_rate = float(args['selection'])
    n_iterations = int(args['iterations'])

    main()
