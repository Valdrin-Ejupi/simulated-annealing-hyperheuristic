from models import Parser, Solver
import random
import os
import time
import multiprocessing
import numpy as np

class AdaptiveSelector:
    def __init__(self, methods):
        self.methods = methods
        self.names = [m.__name__ for m in methods]
        
        # Initial probabilities: SA=40%, ILS=35%, GA=25%
        self.weights = {"simulated_annealing_hybrid_parallel": 0.4, 
                       "iterated_local_search": 0.35,
                       "hybrid_parallel_evolutionary_search": 0.25}
        
        self.successes = {name: 0 for name in self.names}
        self.calls = {name: 0 for name in self.names}
    
    def select(self):
        """Select method based on current probabilities"""
        total = sum(self.weights.values())
        probs = [self.weights[name]/total for name in self.names]
        idx = np.random.choice(len(self.methods), p=probs)
        return self.methods[idx], self.names[idx]
    
    def update(self, method_name, improved, improvement=0):
        """Update method performance and adjust probabilities"""
        self.calls[method_name] += 1
        if improved:
            self.successes[method_name] += 1
            # Boost probability for successful methods
            self.weights[method_name] *= 1.1
        else:
            # Reduce probability for failed methods
            self.weights[method_name] *= 0.95
        
        # Keep minimum 5% probability
        self.weights[method_name] = max(self.weights[method_name], 0.05)
    
    def get_stats(self):
        """Get current probabilities and success rates"""
        total = sum(self.weights.values())
        return {name: {
            'prob': self.weights[name]/total*100,
            'success_rate': self.successes[name]/self.calls[name]*100 if self.calls[name] > 0 else 0,
            'calls': self.calls[name]
        } for name in self.names}

def run_adaptive_hypercycle(parser, solver, method_pool, rounds=10, time_limit=600):
    print(f"Starting adaptive hyperheuristic: {rounds} rounds, {time_limit/60:.1f}min limit")
    
    # Initialize
    selector = AdaptiveSelector(method_pool)
    data = parser.parse()
    start_time = time.time()
    
    # Initial solution
    solution = solver.generate_initial_solution_grasp(data, p=0.05, max_time=5)
    print(f"Initial score: {solution.fitness_score:,}")
    best_solution = solution
    improvements = 0
    
    for round_num in range(rounds):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            print(f" Time limit reached ({elapsed:.1f}s)")
            break
        
        remaining_time = time_limit - elapsed
        method, method_name = selector.select()
        
        print(f"\nRound {round_num+1}: {method_name.split('_')[0].upper()} "
              f"(Time left: {remaining_time:.1f}s)")
        
        previous_score = best_solution.fitness_score
        
        try:
            # Run selected method
            if "annealing" in method_name:
                max_iter = min(500, int(remaining_time * 10))
                _, candidate = method(data, max_iterations=max_iter, initial_solution=best_solution)
            elif "local_search" in method_name:
                time_limit_method = min(60, remaining_time - 5)
                candidate = method(data, initial_solution=best_solution, 
                                 time_limit=time_limit_method, max_iterations=500)
            elif "evolutionary" in method_name:
                time_limit_method = min(120, remaining_time - 5)
                _, candidate = method(data, initial_solution=best_solution, 
                                    num_iterations=500, time_limit=time_limit_method)
            
            # Check improvement
            if candidate and candidate.fitness_score > previous_score: # type: ignore
                improvement = candidate.fitness_score - previous_score
                best_solution = candidate
                improvements += 1
                selector.update(method_name, True, improvement)
                print(f" IMPROVED: {best_solution.fitness_score:,} (+{improvement:,})")
            else:
                selector.update(method_name, False)
                print(f" No improvement")
                
        except Exception as e:
            selector.update(method_name, False)
            print(f" Error: {str(e)}")
        
        # Show stats every 3 rounds
        if (round_num + 1) % 3 == 0:
            print(f"\n After round {round_num + 1}:")
            stats = selector.get_stats()
            for name, stat in stats.items():
                short_name = name.split('_')[0][:4].upper()
                print(f"  {short_name}: {stat['prob']:.1f}% prob, "
                      f"{stat['success_rate']:.1f}% success ({stat['calls']} calls)")
    
    total_time = time.time() - start_time
    print(f"\n COMPLETED: {improvements} improvements in {total_time:.1f}s")
    print(f"Final score: {best_solution.fitness_score:,}")
    
    return best_solution

def run_pipeline():
    print("---------- ADAPTIVE HYPERHEURISTIC ----------")
    
    input_dir, output_dir = './input', './output'
    os.makedirs(output_dir, exist_ok=True)
    
    solver = Solver()
    methods = [solver.simulated_annealing_hybrid_parallel,
               solver.iterated_local_search, 
               solver.hybrid_parallel_evolutionary_search]
    
    # Store final scores for all runs
    all_final_scores = {}
    
    # Run 5 times
    for run_id in range(1, 6):
        print(f'\n{"="*80}')
        print(f' RUN {run_id}/5')
        print(f'{"="*80}')
        
        run_scores = {}
        
        for file in os.listdir(input_dir):
            if file.endswith(('.txt', '.in')):
                print(f'\n{"="*60}\nProcessing: {file} (Run {run_id})\n{"="*60}')
                
                try:
                    parser = Parser(os.path.join(input_dir, file))
                    solution = run_adaptive_hypercycle(parser, solver, methods, rounds=10, time_limit=600)
                    
                    # Create unique output filename with run number
                    base_name = file.replace('.in', '').replace('.txt', '')
                    output_filename = f'{base_name}_{run_id}.txt'
                    output_path = os.path.join(output_dir, output_filename)
                    
                    solution.export(output_path)
                    run_scores[file] = solution.fitness_score
                    
                    print(f" Saved: {output_path}")
                    print(f" Score: {solution.fitness_score:,}")
                    
                except Exception as e:
                    print(f" Error: {e}")
                    run_scores[file] = 0
        
        all_final_scores[f'Run_{run_id}'] = run_scores
    
    # Print final scores summary
    print(f'\n{"="*80}')
    print(f' FINAL SCORES SUMMARY - ALL 5 RUNS')
    print(f'{"="*80}')
    
    for file in sorted(set().union(*[scores.keys() for scores in all_final_scores.values()])):
        print(f'\n File: {file}')
        scores_for_file = []
        for run_key in sorted(all_final_scores.keys()):
            score = all_final_scores[run_key].get(file, 0)
            scores_for_file.append(score)
            print(f"  {run_key}: {score:,}")
        
        if scores_for_file:
            best_score = max(scores_for_file)
            avg_score = sum(scores_for_file) / len(scores_for_file)
            print(f"   Best: {best_score:,}")
            print(f"   Average: {avg_score:,.0f}")
    
    print(f'\n All runs completed! Check ./output/ for numbered solution files.')

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_pipeline()
