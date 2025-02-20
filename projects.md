## 0. A project of your choice, that might be useful for your research!

In this case get in touch with the instructor and the assistant before starting developing your project.

---

## 1. Parallelized Monte Carlo estimation of $\pi$

### Problem statement
Use Monte Carlo methods to estimate $\pi$ efficiently using **parallel computing**.

### Mathematical formulation
Generate $N$ random points $(x,y) \in [0,1] \times [0,1]$. The fraction of points inside the quarter-circle $x^2 + y^2 \leq 1$ estimates $\pi$ as:

$$
\pi \approx 4 \times \frac{\text{points inside quarter-circle}}{\text{total points}}
$$

### Implementation steps
1. Naïve NumPy implementation.
2. Parallelized version with Numba (`@njit(parallel=True)`).
3. GPU-accelerated version using Numba CUDA.
4. Compare performance (runtime, efficiency, and error convergence).

### Expected output
- Speedup plots (NumPy vs. CPU parallel vs. GPU).
- Error analysis vs. number of samples.

---

## 2. (Complex) High-Performance eigenvalue solver

### Problem statement
Solve the **eigenvalue problem** efficiently for a large sparse matrix:

$$
Ax = \lambda x
$$

where $A$ is a symmetric matrix.

### Implementation steps
1. Generate a large random sparse matrix using `scipy.sparse`.
2. Compare three approaches:
   - Baseline: NumPy's `numpy.linalg.eig`.
   - Optimized: SciPy's `scipy.sparse.linalg.eigs`.
   - High-performance: Custom Numba-based iterative solver.
3. Profile runtime and memory usage.

### Expected output
- Performance comparison graphs (runtime vs. matrix size).
- Trade-off analysis between accuracy and efficiency.

---

## 3. (Complex) Large-Scale data processing: profiling and optimization

### Problem statement
Optimize **large-scale data processing** operations for a dataset with $10^8$ rows.

### Implementation steps
1. Load and analyze a dataset (e.g., financial data, sensor logs).
2. Profile performance using `cProfile`, `line_profiler`, and `memory_profiler`.
3. Optimize bottlenecks:
   - Replace loops with NumPy vectorization.
   - Use Numba for fast computations.
   - Optimize storage: compare CSV, HDF5, Parquet.
4. Benchmark before and after optimizations.

### Expected output
- Performance graphs (before vs. after optimization).
- Report on bottleneck analysis and applied optimizations.

---

## 4. (Complex) Parallel K-Means clustering on HPC

### Problem statement
Optimize **K-Means clustering** for large datasets using parallelization.

### Mathematical formulation
Given data points $x_1, x_2, ..., x_N$, partition them into $K$ clusters by minimizing:

$$
J = \sum_{i=1}^{N} \min_{k} \| x_i - \mu_k \|^2
$$

where $\mu_k$ are cluster centroids.

### Implementation steps
1. Baseline: Naïve K-Means with NumPy.
2. Parallelized version using Numba (`prange`).
3. GPU-accelerated version using CuPy or PyTorch.
4. Test on large datasets (e.g., MNIST, synthetic Gaussian blobs).

### Expected output
- Speedup graphs (serial vs. CPU parallel vs. GPU).
- Clustering performance comparison (runtime vs. dataset size).

---

## 5. Parallel sorting algorithm benchmark

### Problem statement
Compare different **parallel sorting algorithms**.

### Implementation steps
1. Implement different sorting algorithms:
   - Merge Sort (NumPy baseline)
   - Parallel Merge Sort (Numba `prange`)
   - Quicksort with parallel partitioning
2. Compare performance for large random datasets.
3. Profile memory usage and scalability.

### Expected output
- Performance benchmarks (runtime vs. input size).
- Profiling report on efficiency.

---

## 6. (Complex) Parallel matrix multiplication using MPI

### Problem statement
Implement a **parallel matrix multiplication algorithm** to compute $C=A×B$ efficiently for large matrices.

### Implementation steps
- Distribute rows of $A$ across MPI processes.
- Use `mpi4py` to communicate required portions of $B$.
- Gather results into the final matrix $C$.

### Expected output
- Implement a serial matrix multiplication.
- Implement an MPI-parallelized version.
- Compare runtimes for increasing matrix sizes (e.g., $256 \times 256$ to $4096 \times 4096$).

---

## 7. Parallelized numerical integration using trapezoidal rule

### Problem statement
Implement a **parallelized numerical integration** method using the trapezoidal rule to compute the integral of a function over a given interval.

### Mathematical formulation
Given a function $f(x)$, approximate the integral over $[a, b]$ as:

$$
\int_{a}^{b} f(x) \, dx \approx \frac{h}{2} \left( f(a) + 2 \sum_{i=1}^{n-1} f(x_i) + f(b) \right)
$$

where $h = \frac{b-a}{n}$ and $x_i = a + i \cdot h$.

### Implementation steps
1. Implement a serial version using NumPy.
2. Parallelize the computation using Numba (`prange`).
3. Compare performance for different numbers of intervals $n$.
4. Test on functions like $f(x) = \sin(x)$ or $f(x) = e^{-x^2}$.

### Expected output
- Speedup plots (serial vs. parallel).
- Error analysis vs. number of intervals.

---

## 8. (Complex) Parallelized PageRank Algorithm

### Problem statement
Implement a **parallelized version of the PageRank algorithm** to rank web pages in a large graph.

### Mathematical formulation
Given a graph $G = (V, E)$, the PageRank $PR(v)$ of a node $v$ is computed iteratively as:

$$
PR(v) = \frac{1-d}{N} + d \sum_{u \in M(v)} \frac{PR(u)}{L(u)}
$$

where $d$ is the damping factor, $N$ is the total number of nodes, $M(v)$ is the set of nodes pointing to $v$, and $L(u)$ is the number of outbound links from $u$.

### Implementation steps
1. Implement a serial version using NumPy.
2. Parallelize the iterative updates using Numba or MPI.
3. Test on synthetic graphs or real-world datasets (e.g., Stanford Web Graph).
4. Compare convergence rates and runtime.

### Expected output
- Performance benchmarks (serial vs. parallel).
- Convergence analysis for different graph sizes.

---

## 9. Parallelized gradient descent for machine learning

### Problem statement
Implement a **parallelized gradient descent algorithm** for optimizing a machine learning model (e.g., linear regression).

### Mathematical formulation
Given a loss function $L(\theta)$, update the parameters $\theta$ iteratively as:

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

where $\eta$ is the learning rate.

### Implementation steps
1. Implement a serial gradient descent using NumPy.
2. Parallelize the gradient computation using Numba or MPI.
3. Test on synthetic datasets or real-world datasets (e.g., Boston Housing).
4. Compare convergence rates and runtime.

### Expected output
- Performance benchmarks (serial vs. parallel).
- Convergence analysis for different dataset sizes.

---

## 10. Parallelized sparse matrix-vector multiplication

### Problem statement
Implement a **parallelized sparse matrix-vector multiplication** for large sparse matrices.

### Mathematical formulation
Given a sparse matrix $A$ and a vector $x$, compute $y = A \cdot x$.

### Implementation steps
1. Generate a large sparse matrix using `scipy.sparse`.
2. Implement a serial matrix-vector multiplication.
3. Parallelize the computation using Numba or MPI.
4. Compare performance for different matrix sizes.

### Expected output
- Speedup plots (serial vs. parallel).
- Memory usage analysis.
