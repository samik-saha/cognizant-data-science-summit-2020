import random
import numpy as np
import unittest

N = 8
MUTATION_RATE = 0.01
POPULATION_SIZE = 50

class DNA:
    def __init__(self):
        self.chessboard=[0] * N
        for i in range(N):
            self.chessboard[i] = random.randint(0, N-1)

    def fitness(self):
        score = 0
        for i in range(N):
            for j in range(i+1, N):
                if self.chessboard[i] != self.chessboard[j] and \
                    abs(j - i) != abs(self.chessboard[j] - self.chessboard[i]):
                    score+=1
                    
        return score/28
    
    def crossover(self, partner, crossover_point=-1):
        child = DNA() # create a DNA object
        
        # if crossover_point is not specified or out of range
        # choose a random location
        if crossover_point == -1 or crossover_point not in range(0, N):
            crossover_point = random.randint(0, N-1) # randomly choose a crossover point

        # first part of DNA comes from this parent
        child.chessboard[0:crossover_point]=self.chessboard[0:crossover_point]
        # last part of DNA comes from partner
        child.chessboard[crossover_point:N]=partner.chessboard[crossover_point:N]
        return child
    
    def mutate(self):
        for i in range(N):
            if random.random() < MUTATION_RATE:
                self.chessboard[i] = random.randint(0, N-1)

                
class TestDNA(unittest.TestCase):
    def setUp(self):
        self.dna = DNA()
        
    def test_solution_fitness(self):
        self.dna.chessboard = [4,2,0,6,1,7,5,3]
        self.assertEqual(self.dna.fitness(), 1.0, "Solutions should have fitness 1.0")
        
    def test_fitness_score(self):
        self.dna.chessboard = [1,1,1,1,1,1,1,1]
        self.assertEqual(self.dna.fitness(), 0.0, "Expected fitness is 0")
        
    def test_mutate(self):
        global MUTATION_RATE
        MUTATION_RATE = 1
        old_chessboard = self.dna.chessboard.copy()
        self.dna.mutate()
        self.assertNotEqual(old_chessboard, self.dna.chessboard)
        
    def test_crossover(self):
        p = 4
        partner = DNA()
        child = self.dna.crossover(partner, crossover_point=p)
        self.assertEqual(child.chessboard[0:p], self.dna.chessboard[0:p])
        self.assertEqual(child.chessboard[p:N], partner.chessboard[p:N])

class Population:
    def __init__(self):
        self.population = []
        self.fitness_scores = []
        for i in range(POPULATION_SIZE):
            dna = DNA()
            self.population.append(dna)
        self._calculate_fitness_scores()
        
    def draw(self):
        new_population=[]
        for i in range(POPULATION_SIZE):
            partners = random.choices(self.population, weights=self.fitness_scores, k=2)
            child = partners[0].crossover(partners[1])
            child.mutate()
            new_population.append(child)
        self.population = new_population
        self._calculate_fitness_scores()
    
    def _calculate_fitness_scores(self):
        self.fitness_scores = [e.fitness() for e in self.population]
        
    def avg_fitness(self):
        return np.average(self.fitness_scores)
        
    def get_fittest(self):
        max_index = np.argmax(self.fitness_scores)
        return self.population[max_index], self.fitness_scores[max_index]

class TestPopulation(unittest.TestCase):
    def setUp(self):
        global POPULATION_SIZE
        POPULATION_SIZE = 10
        self.population = Population()
    
    def test_poplation_size(self):
        self.assertEqual(len(self.population.population), POPULATION_SIZE)
    
    def test_avg_fitness(self):
        af = self.population.avg_fitness()
        # fitness should be within 0 and 1
        self.assertTrue(af >= 0 and af <=1)
        
    def test_max_fitness(self):
        _, mf = self.population.get_fittest()
        # fitness should be within 0 and 1
        self.assertTrue(mf >= 0 and mf <=1)
        
    def test_draw(self):
        old_population = self.population.population.copy()
        self.population.draw()
        # size of the new population is still the same
        self.assertEqual(len(self.population.population), POPULATION_SIZE)
        
        # new population is different from old population
        self.assertNotEqual(old_population, self.population.population)
        
    
if __name__ == '__main__':
    population = Population()
    for i in range(20000):
        population.draw()
        dna, max_fitness = population.get_fittest()
        avg_fitness = population.avg_fitness()
        print(f'Generation %3d :%.5f, %.5f ,%s' % (i, avg_fitness, max_fitness, dna.chessboard))
        if max_fitness == 1.0:
            break