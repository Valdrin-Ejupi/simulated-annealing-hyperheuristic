import random
from collections import defaultdict
import threading
import time
from models.library import Library
import os
# from tqdm import tqdm
from models.solution import Solution
import copy
import random
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from typing import Tuple
from models.instance_data import InstanceData
#Stimulaited Annealing Cooling Functions
def cooling_exponential(temp, cooling_rate=0.003):
    return temp * (1 - cooling_rate)
def cooling_geometric(temp, alpha=0.95):
    return temp * alpha
def cooling_lundy_mees(temp, beta=0.001):
    return temp / (1 + beta * temp)
class Solver:
    def __init__(self):
        pass
    def generate_initial_solution(self, data):
        Library._id_counter = 0
        
        shuffled_libs = data.libs.copy()
        random.shuffle(shuffled_libs)

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        # for library in tqdm(shuffled_libs): # If the visualisation is needed
        for library in shuffled_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books, key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)

        solution.calculate_fitness_score(data.scores)

        return solution

    def crossover(self, solution, data):
        """Performs crossover by shuffling library order and swapping books accordingly."""
        new_solution = copy.deepcopy(solution) 

        old_order = new_solution.signed_libraries[:]
        library_indices = list(range(len(data.libs)))
        random.shuffle(library_indices)

        new_scanned_books_per_library = {}

        for new_idx, new_lib_idx in enumerate(library_indices):
            if new_idx >= len(old_order):
                break 

            old_lib_id = old_order[new_idx]
            new_lib_id = new_lib_idx

            if new_lib_id < 0 or new_lib_id >= len(data.libs):
                print(f"Warning: new_lib_id {new_lib_id} is out of range for data.libs (size: {len(data.libs)})")
                continue

            if old_lib_id in new_solution.scanned_books_per_library:
                books_to_move = new_solution.scanned_books_per_library[old_lib_id]

                existing_books_in_new_lib = {book.id for book in data.libs[new_lib_id].books}

                valid_books = []
                for book_id in books_to_move:
                    if book_id not in existing_books_in_new_lib and book_id not in [b for b in valid_books]:
                        valid_books.append(book_id)

                new_scanned_books_per_library[new_lib_id] = valid_books

        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_signed(self, solution, data):
        """
        Randomly swaps two libraries within the signed libraries list.
        This creates a new solution by exchanging the positions of two libraries
        while maintaining the feasibility of the solution.

        Args:
            solution: The current solution to tweak
            data: The problem data

        Returns:
            A new solution with two libraries swapped
        """
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)

        idx1, idx2 = random.sample(range(len(solution.signed_libraries)), 2)

        lib_id1 = solution.signed_libraries[idx1]
        lib_id2 = solution.signed_libraries[idx2]

        new_signed_libraries = solution.signed_libraries.copy()
        new_signed_libraries[idx1] = lib_id2
        new_signed_libraries[idx2] = lib_id1

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        for lib_id in new_signed_libraries:
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = []
            for book in library.books:
                if (
                    book.id not in scanned_books
                    and len(available_books) < max_books_scanned
                ):
                    available_books.append(book.id)

            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books

        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    # region Hill Climbing Signed & Unsigned libs
    def _extract_lib_id(self, libraries, library_index):
        return int(libraries[library_index][len("Library "):])

    def tweak_solution_swap_signed_with_unsigned(self, solution, data, bias_type=None, bias_ratio=2/3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        local_signed_libs = solution.signed_libraries.copy()
        local_unsigned_libs = solution.unsigned_libraries.copy()

        total_signed = len(local_signed_libs)

        # Bias
        if bias_type == "favor_first_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(0, total_signed // 2 - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(total_signed // 2, total_signed - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        unsigned_idx = random.randint(0, len(local_unsigned_libs) - 1)

        # signed_lib_id = self._extract_lib_id(local_signed_libs, signed_idx)
        # unsigned_lib_id = self._extract_lib_id(local_unsigned_libs, unsigned_idx)
        signed_lib_id = local_signed_libs[signed_idx]
        unsigned_lib_id = local_unsigned_libs[unsigned_idx]

        # Swap the libraries
        local_signed_libs[signed_idx] = unsigned_lib_id
        local_unsigned_libs[unsigned_idx] = signed_lib_id
        # print(f"swapped_signed_lib={unsigned_lib_id}")
        # print(f"swapped_unsigned_lib={unsigned_lib_id}")

        # Preserve the part before `signed_idx`
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        lib_lookup = {lib.id: lib for lib in data.libs}

        # Process libraries before the swapped index
        for i in range(signed_idx):
            # lib_id = self._extract_lib_id(solution.signed_libraries, i)
            lib_id = solution.signed_libraries[i]
            library = lib_lookup.get(lib_id)

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Recalculate from `signed_idx` onward
        new_signed_libraries = local_signed_libs[:signed_idx]

        for i in range(signed_idx, len(local_signed_libs)):
            # lib_id = self._extract_lib_id(local_signed_libs, i)
            lib_id = local_signed_libs[i]
            library = lib_lookup.get(lib_id)

            if curr_time + library.signup_days >= data.num_days:
                solution.unsigned_libraries.append(library.id)
                continue

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_signed_libraries.append(library.id)  # Not f"Library {library.id}"
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Update solution
        new_solution = Solution(new_signed_libraries, local_unsigned_libs, new_scanned_books_per_library, scanned_books)
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_same_books(self, solution, data):
        library_ids = [lib for lib in solution.signed_libraries if lib < len(data.libs)]

        if len(library_ids) < 2:
            return solution

        idx1 = random.randint(0, len(library_ids) - 1)
        idx2 = random.randint(0, len(library_ids) - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, len(library_ids) - 1)

        library_ids[idx1], library_ids[idx2] = library_ids[idx2], library_ids[idx1]

        ordered_libs = [data.libs[lib_id] for lib_id in library_ids]

        all_lib_ids = set(range(len(data.libs)))
        remaining_lib_ids = all_lib_ids - set(library_ids)
        for lib_id in sorted(remaining_lib_ids):
            ordered_libs.append(data.libs[lib_id])

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in ordered_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_last_book(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution  # No scanned or unsigned libraries, return unchanged solution

        # Pick a random library that has scanned books
        chosen_lib_id = random.choice(list(solution.scanned_books_per_library.keys()))
        scanned_books = solution.scanned_books_per_library[chosen_lib_id]

        if not scanned_books:
            return solution  # Safety check, shouldn't happen

        # Get the last scanned book from this library
        last_scanned_book = scanned_books[-1]  # Last book in the list

        # library_dict = {f"Library {lib.id}": lib for lib in data.libs}
        library_dict = {lib.id: lib for lib in data.libs}

        best_book = None
        best_score = -1

        for unsigned_lib in solution.unsigned_libraries:
            library = library_dict[unsigned_lib]  # O(1) dictionary lookup

            # Find the first unscanned book from this library
            for book in library.books:
                if book.id not in solution.scanned_books:  # O(1) lookup in set
                    if data.scores[book.id] > best_score:  # Only store the best
                        best_book = book.id
                        best_score = data.scores[book.id]
                    break  # Stop after the first valid book

        # Assign the best book found (or None if none exist)
        first_unscanned_book = best_book

        if first_unscanned_book is None:
            return solution  # No available unscanned books

        # Create new scanned books mapping (deep copy)
        new_scanned_books_per_library = {
            lib_id: books.copy() for lib_id, books in solution.scanned_books_per_library.items()
        }

        # Swap the books
        new_scanned_books_per_library[chosen_lib_id].remove(last_scanned_book)
        new_scanned_books_per_library[chosen_lib_id].append(first_unscanned_book)

        # Update the overall scanned books set
        new_scanned_books = solution.scanned_books.copy()
        new_scanned_books.remove(last_scanned_book)
        new_scanned_books.add(first_unscanned_book)

        # Create the new solution
        new_solution = Solution(
            signed_libs=solution.signed_libraries.copy(),
            unsigned_libs=solution.unsigned_libraries.copy(),
            scanned_books_per_library=new_scanned_books_per_library,
            scanned_books=new_scanned_books
        )

        # Recalculate fitness score
        new_solution.calculate_fitness_score(data.scores)

        return new_solution


    def iterated_local_search(self, data, time_limit=300, max_iterations=1000):
        """
        Implements Iterated Local Search (ILS) with Random Restarts
        Args:
            data: The problem data
            time_limit: Maximum time in seconds (default: 300s = 5 minutes)
            max_iterations: Maximum number of iterations (default: 1000)
        """
        min_time = 5
        max_time = min(60, time_limit)
        T = list(range(min_time, max_time + 1, 5))

        S = self.generate_initial_solution_grasp(data, p=0.05, max_time=20)
        
        print(f"Initial solution fitness: {S.fitness_score}")

        H = copy.deepcopy(S)
        Best = copy.deepcopy(S)
        
        # Create a pool of solutions to choose from as homebase
        solution_pool = [copy.deepcopy(S)]
        pool_size = 5  # Maximum number of solutions to keep in the pool

        start_time = time.time()
        total_iterations = 0

        while (
            total_iterations < max_iterations
            and (time.time() - start_time) < time_limit
        ):
            local_time_limit = random.choice(T)
            local_start_time = time.time()

            while (time.time() - local_start_time) < local_time_limit and (
                time.time() - start_time
            ) < time_limit:

                selected_tweak = self.choose_tweak_method()
                R = selected_tweak(copy.deepcopy(S), data)

                if R.fitness_score > S.fitness_score:
                    S = copy.deepcopy(R)

                if S.fitness_score >= data.calculate_upper_bound():
                    return (S.fitness_score, S)

                total_iterations += 1
                if total_iterations >= max_iterations:
                    break

            if S.fitness_score > Best.fitness_score:
                Best = copy.deepcopy(S)

            # Update the solution pool
            if S.fitness_score >= H.fitness_score:
                H = copy.deepcopy(S)
                # Add the improved solution to the pool
                solution_pool.append(copy.deepcopy(S))
                # Keep only the best solutions in the pool
                solution_pool.sort(key=lambda x: x.fitness_score, reverse=True)
                if len(solution_pool) > pool_size:
                    solution_pool = solution_pool[:pool_size]
            else:
                # Instead of random acceptance, choose a random solution from the pool
                if len(solution_pool) > 1:  # Only if we have more than one solution in the pool
                    H = copy.deepcopy(random.choice(solution_pool))
                # Add the current solution to the pool if it's not already there
                if S not in solution_pool:
                    solution_pool.append(copy.deepcopy(S))
                    # Keep only the best solutions in the pool
                    solution_pool.sort(key=lambda x: x.fitness_score, reverse=True)
                    if len(solution_pool) > pool_size:
                        solution_pool = solution_pool[:pool_size]

            S = self.perturb_solution(H, data)

            if Best.fitness_score >= data.calculate_upper_bound():
                break

        return (Best.fitness_score, Best)

    def perturb_solution(self, solution, data):
        """Helper method for ILS to perturb solutions with destroy-and-rebuild strategy"""
        perturbed = copy.deepcopy(solution)

        max_destroy_size = len(perturbed.signed_libraries)
        if max_destroy_size == 0:
            return perturbed

        destroy_size = random.randint(
            min(1, max_destroy_size), min(max_destroy_size, max_destroy_size // 3 + 1)
        )

        libraries_to_remove = random.sample(perturbed.signed_libraries, destroy_size)

        new_signed_libraries = [
            lib for lib in perturbed.signed_libraries if lib not in libraries_to_remove
        ]
        new_unsigned_libraries = perturbed.unsigned_libraries + libraries_to_remove

        new_scanned_books = set()
        new_scanned_books_per_library = {}

        for lib_id in new_signed_libraries:
            if lib_id in perturbed.scanned_books_per_library:
                new_scanned_books_per_library[lib_id] = (
                    perturbed.scanned_books_per_library[lib_id].copy()
                )
                new_scanned_books.update(new_scanned_books_per_library[lib_id])

        curr_time = sum(
            data.libs[lib_id].signup_days for lib_id in new_signed_libraries
        )

        lib_scores = []
        for lib_id in new_unsigned_libraries:
            library = data.libs[lib_id]
            available_books = [
                b for b in library.books if b.id not in new_scanned_books
            ]
            if not available_books:
                continue
            avg_score = sum(data.scores[b.id] for b in available_books) / len(
                available_books
            )
            score = library.books_per_day * avg_score / library.signup_days
            lib_scores.append((score, lib_id))

        lib_scores.sort(reverse=True)

        for _, lib_id in lib_scores:
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
                new_unsigned_libraries.remove(lib_id)

        rebuilt_solution = Solution(
            new_signed_libraries,
            new_unsigned_libraries,
            new_scanned_books_per_library,
            new_scanned_books,
        )
        rebuilt_solution.calculate_fitness_score(data.scores)

        return rebuilt_solution

    def build_grasp_solution(self, data, p=0.05):
        """
        Build a feasible solution using a GRASP-like approach:
        - Sorting libraries by signup_days ASC, then total_score DESC.
        - Repeatedly choosing from the top p% feasible libraries at random.

        Args:
            data: The problem data (libraries, scores, num_days, etc.)
            p: Percentage (as a fraction) for the restricted candidate list (RCL)

        Returns:
            A Solution object with the constructed solution
        """
        libs_sorted = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books)),
        )

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        candidate_libs = libs_sorted[:]

        while candidate_libs:
            rcl_size = max(1, int(len(candidate_libs) * p))
            rcl = candidate_libs[:rcl_size]

            chosen_lib = random.choice(rcl)
            candidate_libs.remove(chosen_lib)

            if curr_time + chosen_lib.signup_days >= data.num_days:
                unsigned_libraries.append(chosen_lib.id)
            else:
                time_left = data.num_days - (curr_time + chosen_lib.signup_days)
                max_books_scanned = time_left * chosen_lib.books_per_day

                available_books = sorted(
                    {book.id for book in chosen_lib.books} - scanned_books,
                    key=lambda b: -data.scores[b],
                )[:max_books_scanned]

                if available_books:
                    signed_libraries.append(chosen_lib.id)
                    scanned_books_per_library[chosen_lib.id] = available_books
                    scanned_books.update(available_books)
                    curr_time += chosen_lib.signup_days
                else:
                    unsigned_libraries.append(chosen_lib.id)

        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        solution.calculate_fitness_score(data.scores)
        return solution

    def generate_initial_solution_grasp(self, data, p=0.05, max_time=60):
        """
        Generate an initial solution using a GRASP-like approach:
        1) Sort libraries by (signup_days ASC, total_score DESC).
        2) Repeatedly pick from top p% of feasible libraries at random.
        3) Optionally improve with a quick local search for up to max_time seconds.

        :param data:      The problem data (libraries, scores, num_days, etc.).
        :param p:         Percentage (as a fraction) for the restricted candidate list (RCL).
        :param max_time:  Time limit (in seconds) to repeat GRASP + local search.
        :return:          A Solution object with the best found solution.
        """
        start_time = time.time()
        best_solution = None
        Library._id_counter = 0

        while time.time() - start_time < max_time:
            candidate_solution = self.build_grasp_solution(data, p)

            improved_solution = self.local_search(
                candidate_solution, data, time_limit=1.0
            )

            if (best_solution is None) or (
                improved_solution.fitness_score > best_solution.fitness_score
            ):
                best_solution = improved_solution

        return best_solution

    def local_search(self, solution, data, time_limit=1.0):
        """
        A simple local search/hill-climbing method that randomly selects one of the available tweak methods.
        Uses choose_tweak_method to select the tweak operation based on defined probabilities.
        Runs for 'time_limit' seconds and tries small random modifications.
        """
        start_time = time.time()
        best = copy.deepcopy(solution)

        while time.time() - start_time < time_limit:
            selected_tweak = self.choose_tweak_method()

            neighbor = selected_tweak(copy.deepcopy(best), data)
            if neighbor.fitness_score > best.fitness_score:
                best = neighbor

        return best

    def choose_tweak_method(self):
        """Randomly chooses a tweak method based on the defined probabilities."""
        tweak_methods = [
            (self.tweak_solution_swap_signed_with_unsigned, 0.5),
            (self.tweak_solution_swap_same_books, 0.1),
            (self.crossover, 0.2),
            (self.tweak_solution_swap_last_book, 0.1),
            (self.tweak_solution_swap_signed, 0.1),
        ]

        methods, weights = zip(*tweak_methods)

        selected_method = random.choices(methods, weights=weights, k=1)[0]
        return selected_method

    def generate_initial_solution_sorted(self, data):
        """
        Generate an initial solution by sorting libraries by:
        1. Signup time in ascending order (fastest libraries first)
        2. Total book score in descending order (highest scoring libraries first)
        
        This deterministic approach prioritizes libraries that can be signed up quickly
        and have high total book scores.
        
        Args:
            data: The problem data containing libraries, books, and scores
            
        Returns:
            A Solution object with the constructed solution
        """
        Library._id_counter = 0
        # Sort libraries by signup time ASC and total book score DESC
        sorted_libraries = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books))
        )
        
        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0
        
        for library in sorted_libraries:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(library.id)
        
        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books
        )
        solution.calculate_fitness_score(data.scores)
        
        return solution


    def tweak_solution_insert_library(self, solution, data, target_lib=None):
        if not solution.unsigned_libraries and target_lib is None:
            return solution

        new_solution = copy.deepcopy(solution)
        curr_time = sum(data.libs[lib_id].signup_days for lib_id in new_solution.signed_libraries)
        
        if target_lib is not None and target_lib not in new_solution.signed_libraries:
            lib_to_insert = target_lib
        else:
            if not new_solution.unsigned_libraries:
                return solution
            insert_idx = random.randint(0, len(new_solution.unsigned_libraries) - 1)
            lib_to_insert = new_solution.unsigned_libraries[insert_idx]
            new_solution.unsigned_libraries.pop(insert_idx)

        if curr_time + data.libs[lib_to_insert].signup_days >= data.num_days:
            return solution
            
        time_left = data.num_days - (curr_time + data.libs[lib_to_insert].signup_days)
        max_books_scanned = time_left * data.libs[lib_to_insert].books_per_day
        
        available_books = sorted(
            {book.id for book in data.libs[lib_to_insert].books} - new_solution.scanned_books,
            key=lambda b: -data.scores[b]
        )[:max_books_scanned]
        
        if available_books:
            best_pos = len(new_solution.signed_libraries)
            best_score = 0
            best_solution = None
            
            for pos in range(len(new_solution.signed_libraries) + 1):
                test_solution = copy.deepcopy(new_solution)
                test_solution.signed_libraries.insert(pos, lib_to_insert)
                test_solution.scanned_books_per_library[lib_to_insert] = available_books
                test_solution.scanned_books.update(available_books)
                test_solution.calculate_fitness_score(data.scores)
                
                if test_solution.fitness_score > best_score:
                    best_score = test_solution.fitness_score
                    best_pos = pos
                    best_solution = test_solution
            
            return best_solution if best_solution else solution
        
        return solution

    def tweak_solution_swap_neighbor_libraries(self, solution, data):
        """Swaps two adjacent libraries in the signed list to create a neighbor solution."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)
        swap_pos = random.randint(0, len(new_solution.signed_libraries) - 2)
        
        # Swap adjacent libraries
        new_solution.signed_libraries[swap_pos], new_solution.signed_libraries[swap_pos + 1] = \
            new_solution.signed_libraries[swap_pos + 1], new_solution.signed_libraries[swap_pos]
        
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        
        # Process libraries before swap point
        for i in range(swap_pos):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Safety check
                continue
            library = data.libs[lib_id]
            curr_time += library.signup_days
            
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
        
        # Re-process from swap point
        i = swap_pos
        while i < len(new_solution.signed_libraries):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Skip invalid library IDs
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
                continue
                
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.extend(new_solution.signed_libraries[i:])
                new_solution.signed_libraries = new_solution.signed_libraries[:i]
                break
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
                i += 1
            else:
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
        
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution
    def simulated_annealing_core_mp_optimized(self, initial_solution, data, cooling_func, iterations, shared_best, lock, name):
        current_solution = copy.deepcopy(initial_solution)
        current_solution.calculate_fitness_score(data.scores)
        best_solution = copy.deepcopy(current_solution)
        current_temp = 1000.0
        start_time = time.time()  # Koha e fillimit për kontrollin 10 minutësh

        # Operatorët kryesorë
        operators = [
            self.tweak_solution_swap_signed,
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books,
            self.tweak_solution_swap_last_book,
            self.tweak_solution_insert_library,
            self.tweak_solution_swap_neighbor_libraries
        ]
        operator_names = [
            "swap_signed",
            "swap_signed_with_unsigned",
            "swap_same_books",
            "swap_last_book",
            "insert_library",
            "swap_neighbor"
        ]

        #Inicializo peshat për secilin operator
        stats = {
            name: {"gain": 1.0, "count": 1} for name in operator_names
        }
        weights = [1.0 for _ in operators]

        for iteration in range(iterations):
            # Kontrolli i kohës për ndalje pas 10 minutash (600 sekonda)
            elapsed_time = time.time() - start_time
            if elapsed_time >= 600:
                print(f"[{name.upper()}] ⏱️ Time limit reached (10 minutes). Breaking at iteration {iteration}")
                break

            # Zgjedh operatorin sipas peshave
            operator = random.choices(operators, weights=weights, k=1)[0]
            op_name = operator_names[operators.index(operator)]

            try:
                new_solution = operator(copy.deepcopy(current_solution), data)
                new_solution.calculate_fitness_score(data.scores)

                delta = new_solution.fitness_score - current_solution.fitness_score
                acceptance_prob = math.exp(delta / current_temp) if delta < 0 else 1.0

                if delta > 0 or random.random() < acceptance_prob:
                    current_solution = new_solution
                    if current_solution.fitness_score > best_solution.fitness_score:
                        best_solution = copy.deepcopy(current_solution)
                    stats[op_name]["gain"] += max(0, delta)

                stats[op_name]["count"] += 1

            except Exception:
                continue

            # Ftohja e temperaturës
            current_temp = cooling_func(current_temp)

            # Sinkronizimi dhe përditësimi i peshave çdo 100 iterime
            if iteration % 100 == 0:
                with lock:
                    if best_solution.fitness_score > shared_best["score"]:
                        shared_best["score"] = best_solution.fitness_score
                        shared_best["solution"] = copy.deepcopy(best_solution)
                    else:
                        current_solution = copy.deepcopy(shared_best["solution"])
                        current_solution.calculate_fitness_score(data.scores)

                #Përditëso peshat sipas mesatares së "gain"
                weights = [
                    stats[name]["gain"] / stats[name]["count"]
                    for name in operator_names
                ]
        with lock:
            if best_solution.fitness_score > shared_best["score"]:
                shared_best["score"] = best_solution.fitness_score
                shared_best["solution"] = best_solution
    def simulated_annealing_hybrid_parallel(self, data, max_iterations=500,initial_solution=None):
            #Generate initial solution using GRASP
            if initial_solution is None:
                initial_solution = self.generate_initial_solution_grasp(data, p=0.05, max_time=5)

            # Shared dictionary and lock for process synchronization
            manager = multiprocessing.Manager()
            shared_best = manager.dict()
            shared_best["score"] = initial_solution.fitness_score
            shared_best["solution"] = initial_solution
            lock = manager.Lock()

            # Launch three paralell processes with different cooling strategies

            processes = [
                multiprocessing.Process(target=self.simulated_annealing_core_mp_optimized,
                                        args=(initial_solution, data, cooling_exponential, max_iterations, shared_best, lock, "exp")),
                multiprocessing.Process(target=self.simulated_annealing_core_mp_optimized,
                                        args=(initial_solution, data, cooling_geometric, max_iterations, shared_best, lock, "geo")),
                multiprocessing.Process(target=self.simulated_annealing_core_mp_optimized,
                                        args=(initial_solution, data, cooling_lundy_mees, max_iterations, shared_best, lock, "lundy"))
            ]

            # Start all processes

            for p in processes:
                p.start()
            
            # Wait for all to finish

            for p in processes:
                p.join()

            return shared_best["score"], shared_best["solution"]
    def validate_solution(self, solution, data):
        # Validates and corrects a solution by removing excess books from libraries that exceed scanning limits.
        id_map = {lib.id: lib for lib in data.libs}
        curr_time = 0

        # Validate each library in the signed list

        for lib_id in list(solution.signed_libraries):
            library = id_map.get(lib_id)
            if library is None:
                continue  # # Skip if ID is invalid
            
            # Calculate how many books this library can scan within the time limit

            if curr_time + library.signup_days >= data.num_days:
                max_books = 0
            else:
                time_left = data.num_days - (curr_time + library.signup_days)
                max_books = time_left * library.books_per_day
           
            # Get currently planned books
            
            scanned_list = solution.scanned_books_per_library.get(lib_id, [])
            actual_count = len(scanned_list)

            # Remove excess books if over limit

            if actual_count > max_books:
                # Remove all books
                if max_books <= 0:
        
                    removed_books = set(scanned_list)
                    solution.scanned_books_per_library.pop(lib_id, None)
                else:
                    # Remove lowest-scoring books
                    sorted_books = sorted(scanned_list, key=lambda b: data.scores[b])
                    remove_count = actual_count - max_books
                    removed_books = set(sorted_books[:remove_count])
                    kept_books = [b for b in scanned_list if b not in removed_books]
                    solution.scanned_books_per_library[lib_id] = kept_books

                # Remove from global scanned books

                solution.scanned_books.difference_update(removed_books)
             
            # Advance time with registered date of this library
            
            curr_time += library.signup_days
        return solution
    # def cpp_style_improvement(self, solution, data, iterations=500):
    #     import random
    #     from copy import deepcopy

    #     new_solution = deepcopy(solution)

    #     book_used = set()
    #     for books in new_solution.scanned_books_per_library.values():
    #         book_used.update(books)

    #     for _ in range(iterations):
    #         if not new_solution.signed_libraries:
    #             break

    #         lib_id = random.choice(list(new_solution.signed_libraries))
    #         books = new_solution.scanned_books_per_library.get(lib_id, [])
    #         if not books:
    #             continue

    #         lib = data.libs[lib_id]
    #         books_sorted = sorted(books, key=lambda b: data.scores[b])
    #         for b_out in books_sorted:
    #             #lib.books contains Book objects
    #             candidates = [b for b in lib.books if b.id not in book_used]
    #             if not candidates:
    #                 continue
    #             b_in = max(candidates, key=lambda b: data.scores[b.id], default=None)

    #             if b_in and data.scores[b_in.id] > data.scores[b_out]:
    #                 books.remove(b_out)
    #                 books.append(b_in.id)
    #                 book_used.remove(b_out)
    #                 book_used.add(b_in.id)
    #                 break

    #         new_solution.scanned_books_per_library[lib_id] = books

    #     new_solution.calculate_fitness_score(data.scores)
    #     return new_solution
    # def greedy_medium_approach(self, solution, data, iterations=1):
    #     from copy import deepcopy
    #     current_solution = deepcopy(solution)
    #     assigned_books = set(current_solution.scanned_books)
    #     assigned_libraries = set(current_solution.signed_libraries)

    #     # Rikalkulojmë current_day bazuar në libraritë ekzistuese
    #     id_map = {lib.id: lib for lib in data.libs}
    #     curr_time = 0
    #     for lib_id in current_solution.signed_libraries:
    #         lib = id_map[lib_id]
    #         curr_time += lib.signup_days
    #         if curr_time >= data.num_days:
    #             break

    #     remaining_libraries = [lib for lib in data.libs if lib.id not in assigned_libraries]
    #     selected_libraries = [
    #         (lib_id, current_solution.scanned_books_per_library.get(lib_id, []))
    #         for lib_id in current_solution.signed_libraries
    #     ]

    #     while curr_time < data.num_days and remaining_libraries:
    #         best_lib = None
    #         best_score = float('-inf')
    #         best_books = []

    #         for lib in remaining_libraries:
    #             days_left = data.num_days - curr_time - lib.signup_days
    #             if days_left <= 0:
    #                 continue

    #             max_books = days_left * lib.books_per_day
    #             available_books = [b for b in lib.books if b.id not in assigned_books]
    #             sorted_books = sorted(available_books, key=lambda b: data.scores[b.id], reverse=True)
    #             top_books = sorted_books[:max_books]
    #             score_sum = sum(data.scores[b.id] for b in top_books)

    #             lib_score = score_sum / lib.signup_days if lib.signup_days else score_sum

    #             if lib_score > best_score and top_books:
    #                 best_score = lib_score
    #                 best_lib = lib
    #                 best_books = top_books

    #         if not best_lib:
    #             break

    #         curr_time += best_lib.signup_days
    #         if curr_time >= data.num_days:
    #             break

    #         # Llogarit sasinë reale të librave që mund të skanohen
    #         days_left = data.num_days - curr_time
    #         max_books = min(days_left * best_lib.books_per_day, len(best_books))
    #         final_books = best_books[:max_books]

    #         assigned_books.update(b.id for b in final_books)
    #         selected_libraries.append((best_lib.id, [b.id for b in final_books]))
    #         assigned_libraries.add(best_lib.id)
    #         remaining_libraries = [l for l in remaining_libraries if l.id != best_lib.id]

    #     # Krijimi i zgjidhjes së re
    #     new_solution = deepcopy(current_solution)
    #     new_solution.signed_libraries = [lib_id for lib_id, _ in selected_libraries]
    #     new_solution.scanned_books_per_library = {lib_id: books for lib_id, books in selected_libraries}
    #     new_solution.scanned_books = set()
    #     for books in new_solution.scanned_books_per_library.values():
    #         new_solution.scanned_books.update(books)

    #     new_solution.calculate_fitness_score(data.scores)
    #     self.validate_solution(new_solution, data)

    #     return new_solution