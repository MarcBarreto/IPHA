import numpy as np
import matplotlib.pyplot as plt
from Resnet import infer
from Utils import get_noise
from abc import ABC, abstractmethod

class IPHA(ABC):
    def __init__(self, constant, num_iterations, num_individuals, name = 'IPHA'):
        self.c = get_noise(constant)
        self.num_iterations = num_iterations
        self.num_individuals = num_individuals
        self.name = name

    def __call__(self, x, label):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim != 3 or (x.shape[0] != 3 and x.shape[-1] != 3):
            raise ValueError(f'Error: The image shape should be: [3, <height>, <width>] or [<height>, <width>, <channel>]')

        if x.shape[0] != 3:
            x = x.transpose(2, 0, 1)

        important_features, non_important_features, worst_fitness = self.optimizer(x, label)

        if non_important_features is None:
            non_important_features = 1 - important_features
        else:
            test_non_imp_features = 1 - important_features
            test_eval = self.fitness(x, label, np.array([test_non_imp_features]))
            if test_eval < worst_fitness:
                non_important_features = test_non_imp_features

        return important_features, non_important_features, self.c

    @abstractmethod
    def optimizer(self, x, label):
        pass

    def generate_population(self, k):
        population = []
        
        for _ in range(k):
            mask = np.random.randint(0, 2, size=(1, 32, 32))
            population.append(mask)
            
        return population.copy()

    @abstractmethod
    def fitness(self, x, label, mask):
        pass
    
    def eval(self, img, label, mask):
        return infer(mask * img + (1 - mask) * self.c, label)

    def feature_impact_index(self, x, label, x_selected):
        x_score = infer(x, label)

        x_selected_score = infer(x_selected, label)
        
        return abs(x_score - x_selected_score), x_score, x_selected_score

    def calculate_pr_score(self, original_image, compare_image):
        preserved_pixels_original = np.sum(np.all(original_image != self.c, axis=-1))
        
        total_original_pixels = original_image.shape[0] * original_image.shape[1]
        
        preserved_pixels_compare = np.sum(np.all(compare_image != self.c, axis=-1))
        
        if total_original_pixels == 0:
            return 0
        
        preservation_ratio = preserved_pixels_compare / preserved_pixels_original
        
        return preservation_ratio
    
    def compare_images(self, image, label, x_important : np.ndarray, x_non_important : np.ndarray):       
        fi_important, img_score, important_score = self.feature_impact_index(image, label, x_important)
        
        fi_non_important, _, non_important_score = self.feature_impact_index(image, label, x_non_important)

        return img_score, important_score, non_important_score, fi_important, fi_non_important
    
class IPHA_GA(IPHA):
    def __init__(self, constant, num_iterations, num_individuals, select = 2, pc = 0.7, pm = None, cf = 0.0001):
        super().__init__(constant, num_iterations, num_individuals, 'genetic algorithm')

        self.select = select
        self.pc = pc
        self.pm = pm
        self.cf = cf
        
        self.history = None
        
    def optimizer(self, x, label):
        i = 0
        terminationCounter = 0
        
        count_best_fitness = 0
        mean_fitness = 0.0
        best_ftns_each_iteration = []
        mean_ftns_each_iteration = []
        mean = .0

        best_fitness = float('-inf')
        best_mask = None

        worst_fitness = float('inf')
        worst_mask = None
        
        population = self.generate_population(self.num_individuals)
        
        population_fitness = self.fitness(x, label, np.repeat(population, 3, axis = 1))
        
        while i < self.num_iterations and best_fitness != 1.0:
            parents_idx = self.get_parents_idx(population_fitness, self.select)

            population, population_fitness, offsprings_idx, worst_mask, worst_fitness = self.crossover(population, x, parents_idx, self.pc, population_fitness, label, worst_mask, worst_fitness)
            
            population, population_fitness, _, worst_mask, worst_fitness = self.mutate(population, x, offsprings_idx, population_fitness, label, worst_mask, worst_fitness, self.pm)
            
            new_best_fitness, count_best_fitness, idx = self.get_max(population_fitness)

            best_mask = population[idx]
            
            mean = sum(population_fitness) / len(population)
    
            best_ftns_each_iteration.append(best_fitness)
            mean_ftns_each_iteration.append(mean)
            mean_fitness += mean
            i+= 1
            
            fitness_change = new_best_fitness - best_fitness
                
            if best_fitness >= 0.99:
                if fitness_change >= self.cf:
                    terminationCounter = 0
                
                else:
                    terminationCounter += 1

            best_fitness = new_best_fitness
                
            if terminationCounter >= 10:
                break
                
        mean_fitness = mean_fitness / i

        self.history = {}
        self.history['best_fitness'] = best_fitness
        self.history['last_iteration'] = i
        self.history['count_best_fitness'] = count_best_fitness
        self.history['mean_fitness'] = mean_fitness
        self.history['best_ftns_each_iteration'] = best_ftns_each_iteration
        self.history['mean_ftns_each_iteration'] = mean_ftns_each_iteration

        return np.repeat(best_mask, 3, axis = 0), np.repeat(worst_mask, 3, axis = 0), worst_fitness

    def fitness(self, image, label, mask):
        return self.eval(image, label, mask)
    
    def crossover(self, population, x, parents_idx, pc, population_fitness, label, worst_mask, worst_fitness):   
        parents = [population[idx] for idx in parents_idx]
        
        num_individuals = len(parents)
        
        for i in range(0, num_individuals, 2):
            if np.random.rand() <= pc and i + 1 < num_individuals:
                parent1 = parents[i].copy().flatten()
                parent2 = parents[i+1].copy().flatten()

                crossover_point = np.random.randint(0, len(parent1))

                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]], axis=0).reshape(1, parents[i].shape[1], parents[i].shape[2])
                
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]], axis = 0).reshape(1, parents[i+1].shape[1], parents[i+1].shape[2])
                
                population, population_fitness, parents_idx, worst_mask, worst_fitness = self.check_eval_offspring([child1, child2], [i, i + 1], population, x, label, population_fitness, worst_mask, worst_fitness)         
                    
        return population, population_fitness, parents_idx, worst_mask, worst_fitness

    def mutate(self, population, x, parents_idx, population_fitness, label, worst_mask, worst_fitness, pm = None):
        offsprings = [population[idx] for idx in parents_idx]
        
        _, h, w = x.shape
        
        if pm is None:
            pm = 1 / (h * w)
        
        for idx in range(len(offsprings)): 
            for i in range(h):
                for j in range(w):
                    p = np.random.rand()
                    if p <= pm:
                        if offsprings[idx][0, i, j] == 1:
                            offsprings[idx][0, i, j] = 0
                        else:
                            offsprings[idx][0, i, j] = 1
        
        return self.check_eval_offspring(offsprings, parents_idx, population, x, label, population_fitness, worst_mask, worst_fitness)
    
    def check_eval_offspring(self, offsprings, parents_idx, population, x, label, population_fitness, worst_mask, worst_fitness):
        idx_sorted = np.argsort(population_fitness)[::-1]
        eval_idx = len(population_fitness) - 1
        offsprings_idx = []

        if worst_fitness > population_fitness[idx_sorted[-1]]:
            worst_fitness = population_fitness[idx_sorted[-1]]
            worst_mask = population[idx_sorted[-1]]

        offsprings_score = self.fitness(x, label, np.repeat(offsprings, 3, axis = 1))
        
        for idx, offspring in enumerate(offsprings):
            individual_idx = idx_sorted[eval_idx]
            if offsprings_score[idx] > population_fitness[individual_idx]:
                population[individual_idx] = offspring
                population_fitness[individual_idx] = offsprings_score[idx]
                offsprings_idx.append(individual_idx)
                eval_idx -= 1

            if offsprings_score[idx] < worst_fitness:
                worst_fitness = offsprings_score[idx]
                worst_mask = offspring
        
        missing_offsprings = len(offsprings) - len(offsprings_idx)
        if missing_offsprings > 0:
            sorted_parents = sorted(parents_idx, key=lambda i: population_fitness[i], reverse = True)
            offsprings_idx.extend(sorted_parents[:missing_offsprings])
            
        return population, population_fitness, offsprings_idx, worst_mask, worst_fitness
    
    def get_parents_idx(self, scores, num_parents):
        sorted_indices = np.argsort(scores)[::-1]
        
        return sorted_indices[:num_parents]

    def get_2min(self, population):
        min1 = 2000
        min2 = 2100
        min1_index, min2_index = -1, -1
        for i in range(len(population)):
            if population[i] < min1:
                min2 = min1
                min2_index = min1_index
                
                min1 = population[i]
                min1_index = i
            elif population[i] < min2:
                min2 = population[i]
                min2_index = i
    
        return min1_index, min2_index

    def get_max(self, population):
        max = float('-inf')
        best_individual_idx = 0
        count_max_individual = 0
        
        for idx, individual in enumerate(population):
            if individual > max:
                max = individual
                best_individual_idx = idx
                
            if individual == 1.0:
                count_max_individual += 1
                
        return max, count_max_individual, best_individual_idx