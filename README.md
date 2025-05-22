# **Simulated Annealing Hybrid Parallel – Book Scanning (Google Hash Code 2020)**
<table>
  <tr>
   <td>
     <img src="Images/uni-logo.png" alt="University Logo" width="200" >
    </td>
    <td>
      <h2>University of Prishtina "HASAN PRISHTINA"</h2>
      <p><strong>Faculty of Electronic and Computer Engineering</p>
      <p><strong>Major: </strong> Software and Computer Engineering</p>
      <p><strong>Kodra e Diellit, p.n. - 10000 Prishtinë, Kosova</string>
    </td>
   
  </tr>
</table>
  

---

## Subject Details
- **Program**: Master
- **Topic**: Algorithms Inspired by Nature
- **Proffesor**: Prof.Dr.Kadri Sylejmani
- **Students**: Valdrin Ejupi, Diana Beqiri and Diare Daqi
- **Year:** 2024/2025
---

This repository presents our solution for the **Book Scanning Problem** from the Google Hash Code 2020 competition.

---
## Simulated Annealing

**Simulated Annealing (SA)** is a probabilistic optimization algorithm inspired by the physical annealing process, where materials are slowly cooled to reach a stable, low-energy state. In the context of the Book Scanning problem, SA starts from an initial valid solution and explores neighboring solutions by applying small changes (like swapping books or reordering libraries).

The algorithm accepts not only improvements but occasionally worse solutions, based on a probability that decreases over time—controlled by a "cooling schedule." This helps avoid getting stuck in local optima and encourages broader exploration early on, gradually shifting toward fine-tuning and exploitation as the temperature lowers.

Through iterative refinement and controlled randomness, SA searches for a more optimal library signup and book scanning sequence that maximizes the total score within the deadline.

---

## Problem Overview

Given:
- A set of books with scores
- Multiple libraries (each with books, signup time, and scanning rate)
- A deadline in days

The objective is to schedule the signup of libraries and selection of books in order to **maximize the total score** of scanned books within the allowed time.

---

## Algorithm Description

Our approach combines:

- **GRASP (Greedy Randomized Adaptive Search Procedure)** for initial solution generation
- **Simulated Annealing (SA) with 3 Cooling Functions** for metaheuristic optimization
- **Parallel Execution (multiprocessing)** with 3 SA processes using different cooling functions
- **Adaptive Operator Selection** based on past performance (gain tracking)
- **Shared Memory Synchronization** for best global solution exchange
- **Timeout-Aware Execution** (10-minute time limit)
- **Simulated Annealing Hypercycle** utilizing different approaches to improve our current solution.

---

##  Simulated Annealing Parallel Setup

Three independent processes are launched:

| Process | Cooling Function         |
|---------|--------------------------|
| `exp`   | Exponential Cooling      |
| `geo`   | Geometric Cooling        |
| `lundy` | Lundy-Mees Cooling       |

Each process runs up to **1000 iterations** or **stops earlier if 10 minutes are exceeded**.

At every 100 iterations:
- Processes synchronize using `multiprocessing.Manager().dict()` and `Lock`
- The globally best solution is adopted by each thread if it outperforms the local one

---
### Cooling Strategies
In Simulated Annealing, cooling strategies control how the algorithm gradually lowers the “temperature” to reduce the chance of accepting worse solutions as it progresses. At high temperatures, the algorithm allows uphill moves (worse solutions) to escape local optima; as temperature decreases, it focuses more on refining the current solution.

![Cooling Functions](Images/Figure_1.png)

Our approach executes all three cooling strategies in parallel, allowing the best-performing strategy to dominate without committing to a single path. This improves robustness and convergence across diverse problem instances.

---
##  Adaptive Operator Selection Strategy

We implemented **reward-based adaptive selection** of mutation operators.  
The algorithm tracks the *gain* (score improvement) produced by each operator:

- Operators are selected with probabilities proportional to their **gain / usage count**
- This ensures that well-performing operators are preferred but all operators retain a non-zero chance

At every 100 iterations, operator weights are updated dynamically:

### 🔧 Core Logic

```python
# 1. Initialize gain and usage count
stats = {
    name: {"gain": 1.0, "count": 1} for name in operator_names
}
weights = [1.0 for _ in operators]

# 2. Select operator based on current weights
operator = random.choices(operators, weights=weights, k=1)[0]
op_name = operator_names[operators.index(operator)]

# 3. Update weights every 100 iterations
weights = [
    stats[name]["gain"] / stats[name]["count"]
    for name in operator_names
]
```

---

## Simulated Annealing Hypercycle
To further enhance solution quality, we developed a **hyperheuristic cycle** that chains multiple metaheuristic strategies in a sequential and probabilistic manner. Each method is applied on top of the best-known solution so far, forming a pipeline of refinements.
We perform:
- **10 sequential rounds**
- **500 iterations per method per round**
- **Random selection** of the next method from a pool
- Only **score-improving solutions are accepted**

```python
method_pool = [
    solver.simulated_annealing_hybrid_parallel,
    solver.cpp_style_improvement,
    solver.greedy_medium_approach
]
```
---

## Operators used:
#
| Name                     |
|--------------------------|
| `swap_signed`            |
| `swap_signed_with_unsigned`|
| `swap_same_books`        |
| `swap_last_book`         |
| `insert_library`         |
| `swap_neighbor_libraries`|

---

## Termination Conditions

The algorithm terminates **when either of the following is true**:
- 1000 iterations per process are completed
- 10 minutes of wall-clock execution time have passed

Upon termination:
- Only the **globally best solution** is validated and returned
---

## Resources Used

##### Algorithmic Concepts
- [An Introduction to a Powerful Optimization Technique: Simulated Annealing](https://medium.com/data-science/an-introduction-to-a-powerful-optimization-technique-simulated-annealing-87fd1e3676dd) – Medium
- [Adaptive Operator Selection in Heuristics](https://hal.science/hal-00349087v2/document)

## Python Multiprocessing
- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
---

## Credits & References

Some methods and heuristics in this project were adapted and converted to Python from existing open-source and educational resources. We acknowledge and thank the authors for their contributions:

- C++ implementation of book scanning logic from:  
  [indjev99/Optimizational-Problems-Solutions](https://github.com/indjev99/Optimizational-Problems-Solutions/blob/master/book_scanning.cpp)

- Greedy heuristic strategy based on the analysis and approach described in:  
  [Google Hash Code 2020 – A Greedy Approach (Medium)](https://medium.com/data-science/google-hash-code-2020-a-greedy-approach-2dd4587b6033)

We restructured and integrated these ideas into our own **multi-strategy Python framework**, applying parallel simulated annealing, adaptive operator selection, and hyperheuristic orchestration.

---
## Problem Statement:
- [Google Hash Code 2020 – Book Scanning](https://github.com/Elzawawy/hashcode-book-scanning/blob/master/BookScanningProblem.pdf)
