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
- Implement a serial matrix multiplication.
- Distribute rows of $A$ across MPI processes.
- Use `mpi4py` to communicate required portions of $B$.
- Gather results into the final matrix $C$.

### Expected output
- Performance benchmarks (serial vs. parallel) for increasing matrix sizes (e.g., $256 \times 256$ to $4096 \times 4096$).
- Memory usage analysis.

---

## 7. Parallelized numerical integration using trapezoidal rule

### Problem statement
Implement a **parallelized numerical integration** method using the trapezoidal rule to compute the integral of a function over a given interval.

### Mathematical formulation
Given a function $f(x)$, approximate the integral over $[a, b]$ as:

$$
\int_{a}^{b} f(x) \mathrm{d}x \approx \frac{h}{2} \left( f(a) + 2 \sum_{i=1}^{n-1} f(x_i) + f(b) \right)
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

---

## 11. Efficient Causal Convolutions for Time-Series Forecasting

### Problem statement
Implement efficient **causal convolutions** for time-series forecasting, optimizing for large input sequences. Causal convolutions ensure that the model only uses past information for predictions, making them suitable for tasks where future data points cannot be accessed. See [WaveNet](https://arxiv.org/abs/1609.03499) for a possible application.

### Mathematical formulation
In a **causal convolution**, the output at time $t$ depends only on inputs from time $t$ and earlier. For a 1D causal convolution with kernel $k$, the output $y_t$ is computed as:

$$
y_t = \sum_{i=0}^{d} k_i \cdot x_{t-i}
$$

where $d$ is the filter size (or receptive field) and $x_{t-i}$ are the past inputs. This ensures no information from the future is used when predicting $y_t$.

### Implementation steps
1. Implement a basic 1D/2D/3D causal convolution layer in PyTorch.
2. Extend dilated causal convolutions to increase the receptive field without increasing the computational complexity.
3. Benchmark the model's performance on long input sequences (training time), such as time-series data or raw audio as in [WaveNet](https://arxiv.org/abs/1609.03499).

---

## 12. Efficient LSTM/GRU Implementations

### Problem statement
Implement efficient versions of **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**, focusing on reducing computational cost and improving runtime performance for sequence modelling tasks. 

### Mathematical formulation

- **LSTM** has the following key equations:

$$
i_t = \sigma(W_{ii}x_t + W_{hi}h_{t-1} + b_i) \\
f_t = \sigma(W_{if}x_t + W_{hf}h_{t-1} + b_f) \\
o_t = \sigma(W_{io}x_t + W_{ho}h_{t-1} + b_o) \\
g_t = \tanh(W_{ig}x_t + W_{hg}h_{t-1} + b_g) \\
c_t = f_t \cdot c_{t-1} + i_t \cdot g_t \\
h_t = o_t \cdot \tanh(c_t)
$$

- **GRU** simplifies the gates as:

$$
z_t = \sigma(W_{iz}x_t + W_{hz}h_{t-1} + b_z) \\
r_t = \sigma(W_{ir}x_t + W_{hr}h_{t-1} + b_r) \\
n_t = \tanh(W_{in}x_t + r_t \cdot (W_{hn}h_{t-1}) + b_n) \\
h_t = (1 - z_t) \cdot n_t + z_t \cdot h_{t-1}
$$

### Implementation steps
1. Implement baseline LSTM and GRU using PyTorch.
2. Optimize LSTM/GRU by reducing the number of matrix multiplications and shared weights by vectorization.
3. Investigate fused operations and scripting. 
4. Compare the optimized version regarding runtime, memory usage, and training stability. You can train on your preferable language task, or consider the experiments in [this paper](https://arxiv.org/abs/2410.01201v1) as suggestions.

### Expected output
- Performance comparison (runtime vs. sequence length).
- Memory usage profile for different implementations.

---

## 13. Efficient minLSTM and minGRU Implementations

### Problem statement
Implement efficient **minLSTM** and **minGRU**, which are minimalistic versions of LSTM and GRU designed to reduce computational complexity while maintaining similar performance for sequence modelling.

### Mathematical formulation
- **minLSTM**, **minGRU** mathematical implementation are reported in [this paper](https://arxiv.org/abs/2410.01201v1).

### Implementation steps
1. Implement the minLSTM and minGRU architectures in PyTorch.
2. Optimize their implementations by vectorizing an efficient parallel scan.
3. Compare minLSTM/minGRU against standard LSTM/GRU in terms of training speed, convergence, and accuracy on datasets like sequential MNIST, time-series data, or NLP tasks.

### Expected output
- Performance comparison (runtime vs. sequence length).
- Memory usage profile for different implementations.

---

## 14. Efficient State-Space-Models Implementations

### Problem statement
Implement efficient **S4** and **S6** layers, which are minimalistic versions of LSTM and GRU designed to reduce computational complexity while maintaining similar performance for sequence modelling.

### Mathematical formulation
- **minLSTM**, **minGRU** mathematical implementation are reported in [this paper](https://arxiv.org/abs/2410.01201v1).

### Implementation steps
1. Implement the minLSTM and minGRU architectures in PyTorch.
2. Optimize their implementations by vectorizing an efficient parallel scan.
3. Compare minLSTM/minGRU against standard LSTM/GRU in terms of training speed, convergence, and accuracy on datasets like sequential MNIST, time-series data, or NLP tasks.

### Expected output
- Performance comparison (runtime vs. sequence length).
- Memory usage profile for different implementations.

---

## 15. Implementing Structured State Space Models (S4)

### Problem statement
Implement the **Structured State Space (S4/S6) model**, which is an efficient and scalable variant of SSMs designed for modelling long-range dependencies in sequential data. Refer to [this paper](https://arxiv.org/abs/2312.00752) for an overview.

### Mathematical formulation
The S4/S6 model can be expressed as a specific type of SSM that uses efficient parameterization of the state-space matrices to reduce computational complexity while maintaining the ability to model long-range sequences.

- **State evolution (base)**:

$$
h_t = A h_{t-1} + B x_t
$$

### Implementation steps
1. Implement the baseline version of S4 by following the mathematical formulation using PyTorch or TensorFlow.
2. Implement optimizations such as low-rank matrix approximations and using efficient convolutional operations to handle long-range dependencies.
3. Test the implementation on benchmark datasets like sequential MNIST or long-range forecasting tasks.
4. Compare the performance (speed, memory, accuracy) of S4 with vanilla SSMs and other sequence models like LSTMs and Transformers.

### Expected output
- Performance comparison charts (accuracy, training time, memory usage) between S4, SSM, and other models.
- Analysis of how well the model handles long-range dependencies.

---

