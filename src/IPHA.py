import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from abc import ABC, abstractmethod

class IPHA(ABC):
    def __init__(self, model, constant, num_iterations, num_individuals, name = 'IPHA'):
        self.f = model
        self.c = constant
        self.num_iterations = num_iterations
        self.num_individuals = num_individuals
        self.name = name

    def __call__(self, x, label):
        """
        Processes an input image to generate important and non-important features based on the provided label.

        This method takes an image `x` and a corresponding `label` to calculate two feature sets: `important_features` and 
        `non_important_features`. The function ensures the input image is in the correct format and applies optimization and 
        fitness evaluation to determine the best representation of these features. It returns two versions of the image: one 
        emphasizing the important features and the other emphasizing the non-important features.

        Parameters:
        - x: The input image, either as a NumPy array of shape `[3, height, width]` (for RGB channels first) or 
            `[height, width, 3]` (for RGB channels last).
        - label: The label associated with the image, used for feature optimization.

        Returns:
        - x_important: A version of the input image emphasizing the important features.
        - x_non_important: A version of the input image emphasizing the non-important features.

        Raises:
        - ValueError: If the input image does not conform to the expected shape `[3, <height>, <width>]` or 
        `[<height>, <width>, 3]`.

        Notes:
        - The input image is checked and converted to a NumPy array if necessary. If the image does not have 3 channels, it 
        is transposed to match the `[3, height, width]` format.
        - The optimizer is applied to generate the important and non-important features, with the worst fitness value used 
        to adjust the features if necessary.
        - The final images, `x_important` and `x_non_important`, are generated by combining the original image and the 
        corresponding features, and they are then transposed back to the format `[height, width, 3]` before being returned.
        """
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
            test_eval = self.fitness(x, label, test_non_imp_features)
            if test_eval < worst_fitness:
                non_important_features = test_non_imp_features
            
        x_important = important_features * x + non_important_features * self.c
        x_non_important = non_important_features * x + important_features * self.c

        x_important = x_important.transpose(1, 2, 0)
        x_non_important = x_non_important.transpose(1, 2, 0)
        
        return x_important, x_non_important

    @abstractmethod
    def optimizer(self, x, label):
        pass

    def generate_population(self, image, k):
        """
        Generates a population of random binary masks to be applied to the input image.

        This method creates a population of binary masks, each of which is a 2D array with the same height and width as the 
        input image. The number of masks generated is specified by `k`. Each mask is randomly initialized with values 0 or 1.

        Parameters:
        - image: The input image used as a reference for the shape of the masks. The shape of the image must be in the format 
        `(channels, height, width)`.
        - k: The number of random masks to generate for the population.

        Returns:
        - population: A list of `k` binary masks, each of shape `(1, height, width)` where each element is either 0 or 1.
        """
        population = []
        
        for _ in range(k):
            mask = np.random.randint(0, 2, size=(1, image.shape[1], image.shape[2]))
            population.append(mask)
            
        return population

    @abstractmethod
    def fitness(self, x, label, mask):
        pass
    
    def eval(self, x, label, mask):
        """
        Evaluates the fitness of an image based on a given mask, modifying the image and applying a fitness function.

        This method takes an image `x` and a binary `mask`, where the mask determines which parts of the image are important 
        (for the evaluation process). The parts of the image corresponding to `mask == 1` are kept, while the remaining parts 
        are replaced by a constant value `self.c`. The modified image is then passed through a fitness function `self.f` to 
        evaluate its performance based on the provided `label`.

        Parameters:
        - x: The input image in the format `[channels, height, width]`.
        - label: The label corresponding to the input image, used for evaluation by the fitness function.
        - mask: A binary mask of the same shape as `x`, indicating which parts of the image are considered important.

        Returns:
        - The result of applying the fitness function `self.f` to the modified image `img` and the provided `label`.
        """
        img = mask * x + (1 - mask) * self.c
        img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)

        return self.f(img, label)

    def feature_impact_index(self, x, label, x_selected):
        """
        Calculates the feature impact index between the original image and a modified version based on selected features.

        This method evaluates the difference in the fitness scores of the original image `x` and a modified version `x_selected` 
        that has been altered based on feature selection. The impact of the feature selection is quantified by the absolute 
        difference in their fitness scores. Additionally, the scores of both the original and modified images are returned.

        Parameters:
        - x: The original image in the format `[channels, height, width]`.
        - label: The label corresponding to the input image, used for evaluation by the fitness function.
        - x_selected: The modified image with selected features, which will be compared to the original image `x`.

        Returns:
        - A tuple containing:
        1. The absolute difference in the fitness scores between the original image `x` and the selected features image `x_selected`.
        2. The fitness score of the original image `x`.
        3. The fitness score of the modified image `x_selected`.
        """
        x_score = self.f(x, label)
        x_selected_score = self.f(x_selected, label)
        return abs(x_score - x_selected_score), x_score, x_selected_score

    def calculate_pr_score(self, original_image, compare_image):
        """
        Calculates the Preservation Ratio (PR) score between the original image and a comparison image.

        This method compares the number of pixels that remain unchanged (preserved) in both the original image and the 
        comparison image. The preservation ratio is defined as the ratio of preserved pixels in the comparison image 
        to the preserved pixels in the original image. This score provides an indication of how much of the original 
        image structure has been maintained after the modification.

        Parameters:
        - original_image: The original image in the format `[height, width, channels]`.
        - compare_image: The modified image that is compared to the original image, also in the format `[height, width, channels]`.

        Returns:
        - The preservation ratio, which is a float value representing the proportion of preserved pixels in the comparison image 
        relative to the original image. If there are no preserved pixels in the original image, the score will be 0.
        """
        preserved_pixels_original = np.sum(np.all(original_image != self.c, axis=-1))
        
        total_original_pixels = original_image.shape[0] * original_image.shape[1]
        
        preserved_pixels_compare = np.sum(np.all(compare_image != self.c, axis=-1))
        
        if total_original_pixels == 0:
            return 0
        
        preservation_ratio = preserved_pixels_compare / preserved_pixels_original
        
        return preservation_ratio
    
    def compare_images(self, image, label, important_mask : np.ndarray, non_important_mask : np.ndarray, show = True):
        """
        Compares the original image with the important and non-important features by visualizing them and calculating 
        various metrics such as feature impact index (FI) and preservation ratio (PR).

        This method takes an original image and two masks (for important and non-important features), and compares them 
        by visualizing the original image, the important features, and the non-important features. It also calculates 
        two key evaluation metrics:
        1. **Feature Impact Index (FI)** - A measure of how much the important or non-important features impact the 
        overall evaluation score.
        2. **Preservation Ratio (PR)** - A measure of how much the important or non-important features preserve the 
        structure of the original image.

        Parameters:
        - image: The original image to compare with the masks, in the format `[height, width, channels]`.
        - label: The true label for the image, used for scoring.
        - important_mask: A binary mask that marks the important features of the image.
        - non_important_mask: A binary mask that marks the non-important features of the image.
        - show: A boolean flag indicating whether to display the images and scores using `matplotlib`. Default is `True`.

        Returns:
        - img_score: The score of the original image.
        - important_score: The score of the image with important features.
        - non_important_score: The score of the image with non-important features.
        - fi_important: The feature impact index of the important features.
        - fi_non_important: The feature impact index of the non-important features.
        - important_pr_score: The preservation ratio of the important features.
        - non_important_pr_score: The preservation ratio of the non-important features.
        """
        img = np.array(image)
        
        x_important = np.clip(important_mask, 0, 255).astype(np.uint8)

        important_pr_score = self.calculate_pr_score(img, x_important)
        
        x_important = Image.fromarray(x_important)

        x_non_important = np.clip(non_important_mask, 0, 255).astype(np.uint8)

        non_important_pr_score = self.calculate_pr_score(img, x_non_important)
        
        x_non_important = Image.fromarray(x_non_important)
        
        fi_important, img_score, important_score = self.feature_impact_index(image, label, x_important)
        
        fi_non_important, _, non_important_score = self.feature_impact_index(image, label, x_non_important)

        if show:
            fig, ax = plt.subplots(1, 3, figsize = (12, 6))
    
            fig.suptitle('Original Image Vs Important Features Vs Non Important Features', fontsize = 20)
            ax[0].imshow(image)
            ax[0].set_title(f'Score: {img_score:.2f}', fontsize = 12)
            
            ax[1].set_title(f'Score: {important_score:.2f}. FI: {fi_important:.2f}. PR: {important_pr_score:.2f}', fontsize = 12)
            ax[1].imshow(x_important)
    
            ax[2].set_title(f'Score: {non_important_score:.2f}. FI: {fi_non_important:.2f}. PR: {non_important_pr_score:.2f}', fontsize = 12)
            ax[2].imshow(x_non_important)
            
            plt.tight_layout()
            plt.show()

        return img_score, important_score, non_important_score, fi_important, fi_non_important, important_pr_score, non_important_pr_score

class IPHA_HC(IPHA):
    def __init__(self, model, constant, num_iterations, num_neighbors):
        super().__init__(model, constant, num_iterations, num_neighbors, 'hill_climbing')
    
    def optimizer(self, x, label):
        """
        Optimizes the selection of features in an image by iteratively generating and evaluating masks.

        This method uses a population-based search approach to find the best mask for the input image by evaluating 
        masks based on their fitness scores. It starts with a randomly generated population, iteratively refines it by 
        exploring the neighboring masks, and selects the mask that achieves the highest fitness score.

        Parameters:
        - x: The input image in the format `[channels, height, width]`.
        - label: The true label associated with the image, used to evaluate the fitness of each mask.

        Returns:
        - best_mask: The optimized mask that results in the highest fitness score.
        - None, None: These return values are placeholders, as the method does not use them for the current logic 
        (they are included for consistency with other functions that might return additional values).
        """
        best_mask = self.generate_population(x, 1)[0]
        
        best_mask = np.repeat(best_mask, x.shape[0], axis=0)

        best_eval = float('-inf')
        
        for i in range(self.num_iterations):
            neighbors = self.get_neighbors(best_mask, self.num_individuals)
            
            next_eval = float('-inf')
            next_node = None

            for mask in neighbors:
                mask_eval = self.fitness(x, label, mask)
                if next_eval < mask_eval:
                    next_eval = mask_eval
                    next_node = mask

            if best_eval < next_eval:
                best_mask = next_node
                best_eval = next_eval
                
        return best_mask, None, None

    def fitness(self, image, label, mask):
        return self.eval(image, label, mask)
    
    def get_neighbors(self, best_mask, num_neighbors):
        """
        Generates a set of neighboring masks by modifying a given mask.

        This method creates a specified number of neighboring masks by randomly selecting positions in the input mask 
        and resetting a local region around the selected position. The local region is defined by a 3x3 window centered 
        on the selected position. The neighboring masks are used in the optimization process to explore different feature 
        combinations.

        Parameters:
        - best_mask: The current mask that is considered the best mask. This mask is used as the base to generate neighbors.
        - num_neighbors: The number of neighboring masks to generate.

        Returns:
        - neighbors: A list of neighboring masks, each of the same shape as `best_mask`.
        """
        neighbors = []
        _, h_mask, w_mask = best_mask.shape
        
        for _ in range(num_neighbors):
            while True:
                h = random.randint(0, h_mask - 1)
                w = random.randint(0, w_mask - 1)
                if best_mask[:, h, w].any() == 1:
                    break
    
            neighbor = best_mask.copy()
            
            for i in range(h - 1, h + 2):
                for j in range(w - 1, w + 2):
                    if 0 <= i < h_mask and 0 <= j < w_mask:
                        neighbor[:, i, j] = 0
            
            neighbors.append(neighbor)
    
        return neighbors

class IPHA_GA(IPHA):
    def __init__(self, model, constant, num_iterations, num_individuals, select = 2, pc = 0.7, pm = None, thresh = 0.99, cf = 0.001):
        super().__init__(model, constant, num_iterations, num_individuals, 'genetic algorithm')

        self.select = select
        self.pc = pc
        self.pm = pm
        self.thresh = thresh
        self.cf = cf
        
        self.history = None
        
    def optimizer(self, x, label):
        """
        Optimizes a given image using a genetic algorithm approach to find the best feature mask for classification.

        This method uses a population of masks and evolves them through selection, crossover, and mutation operations.
        The goal is to maximize a fitness function that evaluates the quality of the mask in relation to the given image and label.

        Parameters:
        - x: The image to be optimized.
        - label: The label corresponding to the image.

        Returns:
        - best_mask: The mask that yielded the best fitness score after the optimization process.
        - worst_mask: The mask that yielded the worst fitness score during the optimization process.
        - worst_fitness: The fitness score corresponding to the worst mask found during the optimization.

        Process:
        1. **Population Generation:** A population of possible masks is generated using the `generate_population` method.
        2. **Fitness Evaluation:** The fitness of each mask in the population is calculated using the `fitness` function.
        3. **Selection:** Parents for the next generation are selected based on their fitness scores using `get_parents_idx`.
        4. **Crossover:** Offspring masks are generated by combining the selected parents using `crossover`.
        5. **Mutation:** The offspring masks undergo mutation with a certain probability using the `mutate` method.
        6. **Fitness Comparison:** After each generation, the best mask is identified and its fitness score is recorded.
        7. **Termination Criteria:** The algorithm terminates if the fitness of the best mask reaches the threshold or if no significant improvement is seen for a set number of iterations (`terminationCounter`).

        The optimization terminates early if the change in fitness is below a certain threshold for a consecutive number of iterations, or if the maximum number of iterations is reached.

        History of the optimization is stored in `self.history`, which tracks the best fitness, the mean fitness, and the fitness values for each iteration.
        """

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
        
        population = self.generate_population(x, self.num_individuals)
        
        population_fitness = [self.fitness(x, label, np.repeat(population[i], x.shape[0], axis = 0)) for i in range(len(population))]
        
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
                
            if best_fitness >= self.thresh:
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
        
        return np.repeat(best_mask, x.shape[0], axis = 0), np.repeat(worst_mask, x.shape[0], axis = 0), worst_fitness

    def fitness(self, image, label, mask):
        return self.eval(image, label, mask)
    
    def crossover(self, population, x, parents_idx, pc, population_fitness, label, worst_mask, worst_fitness):   
        """
        Performs crossover between selected parents to create offspring.

        This method performs crossover on a population of masks, generating offspring by combining the genetic material (masks) of selected parents. The crossover occurs with a certain probability (`pc`). The offspring generated are evaluated, and if their fitness is better than the worst fitness seen so far, they are added to the population.

        Parameters:
        - population: List of the current population of masks.
        - x: The image being optimized.
        - parents_idx: Indices of the selected parents for crossover.
        - pc: Crossover probability (determines how likely crossover will happen between two parents).
        - population_fitness: List of fitness scores corresponding to the population.
        - label: The label corresponding to the image `x`.
        - worst_mask: The current worst mask in the population.
        - worst_fitness: The fitness score corresponding to the worst mask.

        Returns:
        - population: The updated population, including any new offspring created by crossover.
        - population_fitness: The fitness scores of the updated population.
        - parents_idx: The updated indices of parents selected for crossover (if needed for future generations).
        - worst_mask: The worst mask, if it was replaced by an offspring with worse fitness.
        - worst_fitness: The fitness score corresponding to the worst mask.

        Process:
        1. **Parent Selection:** Pairs of parents are selected from the population based on their indices (`parents_idx`).
        2. **Crossover Probability Check:** A random number is generated, and if it's less than or equal to `pc`, the parents undergo crossover.
        3. **Crossover Operation:** For each pair of parents, the genetic material is mixed at a random crossover point to create two new offspring.
        4. **Offspring Evaluation:** The new offspring are evaluated using the `check_eval_offspring` method, which checks if the offspring have better fitness than the current population.
        5. **Update Population:** If the offspring have better fitness than the worst mask in the population, they are added to the population, and the worst mask may be replaced.

        The method ensures that the population evolves by combining the best traits of selected parents, potentially leading to better solutions (masks) in subsequent generations.
        """

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
        """
        Applies mutation to selected individuals in the population.

        This method performs mutation on selected individuals (offspring) in the population by flipping bits in the binary mask (represented as a 2D array). The mutation is applied with a certain probability (`pm`), which is determined based on the size of the image unless specified. If no mutation probability is provided (`pm` is `None`), it is set to a default value based on the image size.

        Parameters:
        - population: List of the current population of masks.
        - x: The image being optimized.
        - parents_idx: Indices of the selected individuals to apply mutation to.
        - population_fitness: List of fitness scores corresponding to the population.
        - label: The label corresponding to the image `x`.
        - worst_mask: The worst mask in the current population.
        - worst_fitness: The fitness score corresponding to the worst mask.
        - pm: Mutation probability. The probability with which each bit in a mask will be flipped.

        Returns:
        - population: The updated population with mutated masks.
        - population_fitness: The updated fitness scores of the population after mutation.
        - worst_mask: The worst mask, which may be replaced if a mutated mask performs worse.
        - worst_fitness: The fitness score corresponding to the worst mask.

        Process:
        1. **Mutation Probability (`pm`) Calculation:** If `pm` is not provided, it is calculated as the inverse of the total number of pixels in the image.
        2. **Iterate Over Offsprings:** Each individual in the offspring is selected and mutated.
        3. **Bit Flip:** For each pixel in the mask, a random number is generated. If this random number is less than or equal to the mutation probability (`pm`), the pixel (bit) is flipped (from 0 to 1 or vice versa).
        4. **Offspring Evaluation:** The mutated offspring are evaluated using the `check_eval_offspring` method, which checks if their fitness is better than the worst mask.
        5. **Update Population:** If a mutated mask has better fitness than the worst mask, it replaces the worst mask in the population.

        The mutation process introduces diversity into the population, which can help the algorithm avoid local optima and explore new solutions.
        """
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
        """
        Evaluates and replaces individuals in the population based on offspring performance.

        This method compares the fitness of newly generated offspring with the existing population. If an offspring has better fitness than an individual in the population, it replaces that individual. It also updates the worst mask and fitness score, ensuring the population always contains the best individuals.

        Parameters:
        - offsprings: List of newly generated offspring (masks).
        - parents_idx: Indices of the selected parents from which offspring were generated.
        - population: The current population of individuals (masks).
        - x: The image being optimized.
        - label: The label corresponding to the image `x`.
        - population_fitness: List of fitness scores corresponding to the population.
        - worst_mask: The current worst mask in the population.
        - worst_fitness: The fitness score corresponding to the worst mask.

        Returns:
        - population: The updated population with potentially replaced individuals.
        - population_fitness: The updated fitness scores of the population after evaluating the offspring.
        - offsprings_idx: The indices of the replaced individuals in the population.
        - worst_mask: The worst mask, which may have been replaced by an offspring if it performs worse.
        - worst_fitness: The fitness score corresponding to the worst mask.

        Process:
        1. **Sorting Population by Fitness:** The population is sorted in descending order based on fitness scores. The last individual is considered the worst.
        2. **Offspring Evaluation:** Each offspring is evaluated by calculating its fitness score (`offspring_score`), and it is compared with the worst individual in the population.
        3. **Replacement:** If an offspring has better fitness than an individual in the population, it replaces that individual, and the population is updated.
        4. **Updating Worst Mask:** If an offspring has worse fitness than the current worst mask, the worst mask is updated to the offspring.
        5. **Handling Missing Offsprings:** If the number of offspring is less than the number of replacements (due to sorting or other factors), the method adds the best parents to fill the gap by copying their positions in the population.
        6. **Returning Updated Population:** The method returns the updated population, fitness scores, and the indices of replaced individuals.
        """

        idx_sorted = np.argsort(population_fitness)[::-1]
        eval_idx = len(population_fitness) - 1
        offsprings_idx = []

        if worst_fitness > population_fitness[idx_sorted[-1]]:
            worst_fitness = population_fitness[idx_sorted[-1]]
            worst_mask = population[idx_sorted[-1]]
        
        for offspring in offsprings:
            offspring_score = self.fitness(x, label, np.repeat(offspring, x.shape[0], axis = 0))
            individual_idx = idx_sorted[eval_idx]
            if offspring_score > population_fitness[individual_idx]:
                population[individual_idx] = offspring
                population_fitness[individual_idx] = offspring_score
                offsprings_idx.append(individual_idx)
                eval_idx -= 1

            if offspring_score < worst_fitness:
                worst_fitness = offspring_score
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