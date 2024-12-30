import numpy as np
import time

# Base class for all optimizers
class BaseOptimizer:
    def __init__(self, mean_returns, std_devs, correlation_matrix, risk_free_rate=0.02):
        self.mean_returns = mean_returns
        self.std_devs = std_devs
        self.correlation_matrix = correlation_matrix
        self.covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix
        self.n_assets = len(mean_returns)
        self.risk_free_rate = risk_free_rate

    def _create_solution(self):
        weights = np.random.random(self.n_assets)
        return weights / np.sum(weights)

    def _calculate_fitness(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights) * 12
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights))) * np.sqrt(12)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return sharpe_ratio

    def _enforce_constraints(self, weights):
        weights = np.maximum(weights, 0)
        return weights / np.sum(weights)

# Genetic Algorithm Implementation
class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, mean_returns, std_devs, correlation_matrix, population_size=50, generations=100):
        super().__init__(mean_returns, std_devs, correlation_matrix)
        self.population_size = population_size
        self.generations = generations

    def optimize(self):
        population = [self._create_solution() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = float('-inf')

        for _ in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._calculate_fitness(ind) for ind in population]
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_solution = population[max_idx].copy()
                best_fitness = fitness_scores[max_idx]

            # Tournament Selection
            new_population = []
            for _ in range(self.population_size):
                tournament = np.random.choice(self.population_size, 3)
                winner_idx = tournament[np.argmax([fitness_scores[i] for i in tournament])]
                new_population.append(population[winner_idx])

            # Crossover and Mutation
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = new_population[i], new_population[i+1]
                crossover_point = np.random.randint(1, self.n_assets)
                
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                
                # Mutation
                if np.random.random() < 0.1:
                    child1[np.random.randint(self.n_assets)] = np.random.random()
                if np.random.random() < 0.1:
                    child2[np.random.randint(self.n_assets)] = np.random.random()
                
                offspring.extend([self._enforce_constraints(child1), 
                                self._enforce_constraints(child2)])
            
            population = offspring[:self.population_size]

        # Calculate final portfolio metrics
        portfolio_return = np.sum(self.mean_returns * best_solution) * 12
        portfolio_volatility = np.sqrt(np.dot(best_solution.T, np.dot(self.covariance_matrix, best_solution))) * np.sqrt(12)

        return {
            'optimal_weights': best_solution,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': best_fitness
        }

# PSO Implementation
class PSOOptimizer(BaseOptimizer):
    def __init__(self, mean_returns, std_devs, correlation_matrix, n_particles=30, max_iter=100):
        super().__init__(mean_returns, std_devs, correlation_matrix)
        self.n_particles = n_particles
        self.max_iter = max_iter

    def optimize(self):
        # Initialize particles and velocities
        particles = [self._create_solution() for _ in range(self.n_particles)]
        velocities = [np.random.randn(self.n_assets) * 0.1 for _ in range(self.n_particles)]
        
        # Initialize personal and global best
        personal_best = particles.copy()
        personal_best_fitness = [self._calculate_fitness(p) for p in particles]
        
        global_best_idx = np.argmax(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]

        # PSO parameters
        w = 0.7  # inertia weight
        c1 = 1.5  # cognitive weight
        c2 = 1.5  # social weight

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                
                # Update velocity
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] = self._enforce_constraints(particles[i] + velocities[i])
                
                # Update personal and global best
                fitness = self._calculate_fitness(particles[i])
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    if fitness > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness

        # Calculate final portfolio metrics
        portfolio_return = np.sum(self.mean_returns * global_best) * 12
        portfolio_volatility = np.sqrt(np.dot(global_best.T, np.dot(self.covariance_matrix, global_best))) * np.sqrt(12)

        return {
            'optimal_weights': global_best,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': global_best_fitness
        }

# Compare all algorithms
def compare_algorithms(mean_returns, std_devs, correlation_matrix, n_runs=5):
    algorithms = {
        'GA': lambda: GeneticAlgorithm(mean_returns, std_devs, correlation_matrix),
        'PSO': lambda: PSOOptimizer(mean_returns, std_devs, correlation_matrix)
    }
    
    results = {name: {'sharpe': [], 'time': [], 'returns': [], 
                     'volatility': [], 'weights': []} for name in algorithms}
    
    for _ in range(n_runs):
        for name, algo_constructor in algorithms.items():
            start_time = time.time()
            optimizer = algo_constructor()
            result = optimizer.optimize()
            execution_time = time.time() - start_time
            
            results[name]['sharpe'].append(result['sharpe_ratio'])
            results[name]['time'].append(execution_time)
            results[name]['returns'].append(result['expected_return'])
            results[name]['volatility'].append(result['volatility'])
            results[name]['weights'].append(result['optimal_weights'])
    
    return results
import numpy as np
import time



# Artificial Bee Colony Implementation
class ABCOptimizer(BaseOptimizer):
    def __init__(self, mean_returns, std_devs, correlation_matrix, colony_size=50, max_cycles=100, limit=20):
        super().__init__(mean_returns, std_devs, correlation_matrix)
        self.colony_size = colony_size
        self.max_cycles = max_cycles
        self.limit = limit

    def optimize(self):
        # Initialize food sources
        food_sources = [self._create_solution() for _ in range(self.colony_size)]
        fitness = [self._calculate_fitness(fs) for fs in food_sources]
        trials = [0] * self.colony_size
        
        best_solution = food_sources[np.argmax(fitness)].copy()
        best_fitness = max(fitness)

        for _ in range(self.max_cycles):
            # Employed Bee Phase
            for i in range(self.colony_size):
                # Select random partner (excluding self)
                partner = np.random.choice([j for j in range(self.colony_size) if j != i])
                
                # Generate new solution
                phi = np.random.uniform(-1, 1, self.n_assets)
                new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[partner])
                new_solution = self._enforce_constraints(new_solution)
                
                # Evaluate new solution
                new_fitness = self._calculate_fitness(new_solution)
                
                # Update if better
                if new_fitness > fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker Bee Phase
            probabilities = np.array(fitness) / np.sum(fitness)
            for _ in range(self.colony_size):
                selected_source = np.random.choice(self.colony_size, p=probabilities)
                partner = np.random.choice([j for j in range(self.colony_size) if j != selected_source])
                
                phi = np.random.uniform(-1, 1, self.n_assets)
                new_solution = food_sources[selected_source] + phi * (food_sources[selected_source] - food_sources[partner])
                new_solution = self._enforce_constraints(new_solution)
                
                new_fitness = self._calculate_fitness(new_solution)
                
                if new_fitness > fitness[selected_source]:
                    food_sources[selected_source] = new_solution
                    fitness[selected_source] = new_fitness
                    trials[selected_source] = 0
                else:
                    trials[selected_source] += 1

            # Scout Bee Phase
            for i in range(self.colony_size):
                if trials[i] >= self.limit:
                    food_sources[i] = self._create_solution()
                    fitness[i] = self._calculate_fitness(food_sources[i])
                    trials[i] = 0

            # Update best solution
            current_best = max(fitness)
            if current_best > best_fitness:
                best_solution = food_sources[np.argmax(fitness)].copy()
                best_fitness = current_best

        # Calculate final metrics
        portfolio_return = np.sum(self.mean_returns * best_solution) * 12
        portfolio_volatility = np.sqrt(np.dot(best_solution.T, np.dot(self.covariance_matrix, best_solution))) * np.sqrt(12)

        return {
            'optimal_weights': best_solution,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': best_fitness
        }

# Cuckoo Search Implementation
class CuckooSearchOptimizer(BaseOptimizer):
    def __init__(self, mean_returns, std_devs, correlation_matrix, n_nests=25, max_gen=100, pa=0.25, alpha=0.01):
        super().__init__(mean_returns, std_devs, correlation_matrix)
        self.n_nests = n_nests
        self.max_gen = max_gen
        self.pa = pa  # Probability of abandoning nests
        self.alpha = alpha  # Step size

    def optimize(self):
        # Initialize nests
        nests = [self._create_solution() for _ in range(self.n_nests)]
        fitness = [self._calculate_fitness(n) for n in nests]
        
        best_solution = nests[np.argmax(fitness)].copy()
        best_fitness = max(fitness)

        for _ in range(self.max_gen):
            # Generate new cuckoo
            i = np.random.randint(self.n_nests)
            step = np.random.normal(0, 1, self.n_assets) * self.alpha
            new_nest = self._enforce_constraints(nests[i] + step)
            
            # Evaluate and replace if better
            new_fitness = self._calculate_fitness(new_nest)
            if new_fitness > fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
                
                if new_fitness > best_fitness:
                    best_solution = new_nest.copy()
                    best_fitness = new_fitness
            
            # Abandon worst nests
            sorted_idx = np.argsort(fitness)
            n_abandon = int(self.pa * self.n_nests)
            
            for idx in sorted_idx[:n_abandon]:
                if np.random.random() < self.pa:
                    nests[idx] = self._create_solution()
                    fitness[idx] = self._calculate_fitness(nests[idx])

        # Calculate final metrics
        portfolio_return = np.sum(self.mean_returns * best_solution) * 12
        portfolio_volatility = np.sqrt(np.dot(best_solution.T, np.dot(self.covariance_matrix, best_solution))) * np.sqrt(12)

        return {
            'optimal_weights': best_solution,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': best_fitness
        }

# Firefly Algorithm Implementation
class FireflyAlgorithm(BaseOptimizer):
    def __init__(self, mean_returns, std_devs, correlation_matrix, n_fireflies=30, max_gen=100, 
                 alpha=0.2, beta0=1.0, gamma=1.0):
        super().__init__(mean_returns, std_devs, correlation_matrix)
        self.n_fireflies = n_fireflies
        self.max_gen = max_gen
        self.alpha = alpha  # Randomization parameter
        self.beta0 = beta0  # Attractiveness at distance=0
        self.gamma = gamma  # Light absorption coefficient

    def optimize(self):
        # Initialize fireflies
        fireflies = [self._create_solution() for _ in range(self.n_fireflies)]
        brightness = [self._calculate_fitness(f) for f in fireflies]
        
        best_solution = fireflies[np.argmax(brightness)].copy()
        best_fitness = max(brightness)

        for gen in range(self.max_gen):
            # Reduce randomization parameter over time
            alpha_t = self.alpha * (0.97 ** gen)
            
            # Update each firefly
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if brightness[j] > brightness[i]:
                        # Calculate distance
                        r = np.sqrt(np.sum((fireflies[i] - fireflies[j]) ** 2))
                        
                        # Calculate attractiveness
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        
                        # Move firefly i towards j
                        fireflies[i] = self._enforce_constraints(
                            fireflies[i] + 
                            beta * (fireflies[j] - fireflies[i]) + 
                            alpha_t * np.random.normal(0, 1, self.n_assets)
                        )
                        
                        # Update brightness
                        brightness[i] = self._calculate_fitness(fireflies[i])
                        
                        # Update best solution
                        if brightness[i] > best_fitness:
                            best_solution = fireflies[i].copy()
                            best_fitness = brightness[i]

        # Calculate final metrics
        portfolio_return = np.sum(self.mean_returns * best_solution) * 12
        portfolio_volatility = np.sqrt(np.dot(best_solution.T, np.dot(self.covariance_matrix, best_solution))) * np.sqrt(12)

        return {
            'optimal_weights': best_solution,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': best_fitness
        }

# Update compare_algorithms function
def compare_algorithms(mean_returns, std_devs, correlation_matrix, n_runs=5):
    algorithms = {
        'GA': lambda: GeneticAlgorithm(mean_returns, std_devs, correlation_matrix),
        'PSO': lambda: PSOOptimizer(mean_returns, std_devs, correlation_matrix),
        'ABC': lambda: ABCOptimizer(mean_returns, std_devs, correlation_matrix),
        'CS': lambda: CuckooSearchOptimizer(mean_returns, std_devs, correlation_matrix),
        'FA': lambda: FireflyAlgorithm(mean_returns, std_devs, correlation_matrix)
    }
    
    results = {name: {'sharpe': [], 'time': [], 'returns': [], 
                     'volatility': [], 'weights': []} for name in algorithms}
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        for name, algo_constructor in algorithms.items():
            print(f"Running {name}...", end=' ')
            start_time = time.time()
            optimizer = algo_constructor()
            result = optimizer.optimize()
            execution_time = time.time() - start_time
            
            results[name]['sharpe'].append(result['sharpe_ratio'])
            results[name]['time'].append(execution_time)
            results[name]['returns'].append(result['expected_return'])
            results[name]['volatility'].append(result['volatility'])
            results[name]['weights'].append(result['optimal_weights'])
            print(f"Done! (Time: {execution_time:.2f}s)")
    
    return results


# Main execution
if __name__ == "__main__":
    # Market data
    mean_returns = np.array([-0.00163145, 0.00505178, 0.00419812, 0.00356098, 
                            0.00671718, 0.02562289, 0.00857421])
    
    std_devs = np.array([0.03470809, 0.01023848, 0.0029451, 0.01241735, 
                         0.02522352, 0.18962086, 0.00793627])
    
    correlation_matrix = np.array([
        [1., -0.04632627, -0.54592618, 0.03961123, 0.14952714, 0.09034429, -0.19058109],
        [-0.04632627, 1., 1., 0.50991417, 0.22963001, 0.12988318, 0.26853222],
        [-0.54592618, 1., 1., 0.34882358, -0.04208465, 0.42668526, 0.26853222],
        [0.03961123, 0.50991417, 0.34882358, 1., 0.47954908, 0.35215217, 0.97869841],
        [0.14952714, 0.22963001, -0.04208465, 0.47954908, 1., 0.54182599, 0.35083072],
        [0.09034429, 0.12988318, 0.42668526, 0.35215217, 0.54182599, 1., 0.92904025],
        [-0.19058109, 0.26853222, 0.26853222, 0.97869841, 0.35083072, 0.92904025, 1.]
    ])

    # Run comparison
    print("Running algorithm comparison...")
    results = compare_algorithms(mean_returns, std_devs, correlation_matrix, n_runs=5)

    # Print results
    print("\nAlgorithm Comparison Results (averaged over 5 runs):")
    print("=" * 60)
    
    for algo in results:
        print(f"\n{algo} Results:")
        print("-" * 30)
        print(f"Average Sharpe Ratio: {np.mean(results[algo]['sharpe']):.4f} ± {np.std(results[algo]['sharpe']):.4f}")
        print(f"Average Return: {np.mean(results[algo]['returns']):.2%} ± {np.std(results[algo]['returns']):.2%}")
        print(f"Average Volatility: {np.mean(results[algo]['volatility']):.2%} ± {np.std(results[algo]['volatility']):.2%}")
        print(f"Average Execution Time: {np.mean(results[algo]['time']):.3f}s ± {np.std(results[algo]['time']):.3f}s")
        
        print("\nAverage Optimal Weights:")
        avg_weights = np.mean(results[algo]['weights'], axis=0)
        for i, weight in enumerate(avg_weights, 1):
            print(f"Asset {i}: {weight:.2%}")

    # Print stability comparison
    print("\nSolution Stability (Standard Deviation of Sharpe Ratios):")
    print("-" * 60)
    for algo in results:
        print(f"{algo}: {np.std(results[algo]['sharpe']):.6f}")

    # Print algorithm ranking
    algo_performance = [(algo, np.mean(results[algo]['sharpe'])) for algo in results]
    algo_performance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nAlgorithm Ranking by Sharpe Ratio:")
    print("-" * 60)
    for i, (algo, sharpe) in enumerate(algo_performance, 1):
        print(f"{i}. {algo}: {sharpe:.4f}")