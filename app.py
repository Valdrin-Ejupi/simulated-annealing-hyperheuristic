from models import Parser
from models import Solver
import random
import os
import time
import multiprocessing
solver = Solver()

directory = os.listdir('input')
solver = Solver()
input_dir = './input'
output_dir = './output'

def run_hypercycle(parser, solver, method_pool, rounds=10):
    print("Fillimi i ciklit hyperheuristik me", rounds, "itera.")

    data = parser.parse()

    # Fillimi me zgjidhje iniciale të gjeneruar
    solution = solver.generate_initial_solution_grasp(data, p=0.05, max_time=5)
    print(f"Zgjidhje fillestare → score: {solution.fitness_score}")

    for i in range(rounds):
        method = random.choice(method_pool)

        print(f"\nRaundi {i+1}/{rounds} → {method.__name__}")
        previous_score = solution.fitness_score

        if method.__name__ == "simulated_annealing_hybrid_parallel":
            _, candidate_solution = method(data, max_iterations=500, initial_solution=solution)
        else:
            candidate_solution = method(solution, data, iterations=500)

        # Garantojmë që s’bie score
        if candidate_solution.fitness_score >= previous_score:
            solution = candidate_solution
            print(f"Score pas raundit {i+1}: {solution.fitness_score:,}")
        else:
            print(f"{method.__name__} uli score: {candidate_solution.fitness_score:,} < {previous_score:,} → zgjidhja u injorua")
    return solution
def run_full_pipeline():
    print("---------- HYPERHEURISTIC CYCLE ----------")

    for file in os.listdir(input_dir):
        if file.endswith('.txt') or file.endswith('.in'):
            input_path = os.path.join(input_dir, file)
            print(f'Processing file: {input_path}')
            parser = Parser(input_path)

            # Metodat e disponueshme në cikël
            method_pool = [
                solver.simulated_annealing_hybrid_parallel,
                # solver.cpp_style_improvement,
                # solver.greedy_medium_approach,
            ]

            final_solution = run_hypercycle(parser, solver, method_pool, rounds=10)

            output_path = os.path.join(output_dir, f'sol_{file}')
            final_solution.export(output_path)
            print(f"\n Final solution written to: {output_path}\n")
# def run_parallel_sa():

#     print("---------- SIMULATED ANNEALING WITH MULTIPLE TEMPERATURE FUNCTIONS (PARALLEL) ----------")
#     for file in directory:
#         if file.endswith('.txt') or file.endswith('.in'):
#             print(f'Computing ./input/{file}')
#             parser = Parser(f'./input/{file}')
#             data = parser.parse()
#             score, solution = solver.simulated_annealing_hybrid_parallel(data, max_iterations=1000)
#             print(f'Best score from SA (parallel) for {file}: {score:,}')
#             output_file = f'./output/sa_hybrid_parallel_{file}'
#             solution.export(output_file)
#             print(f"Processing complete! Output written to: {output_file}")
           
if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_full_pipeline()
    # run_parallel_sa()