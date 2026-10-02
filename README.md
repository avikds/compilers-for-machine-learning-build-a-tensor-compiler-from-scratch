# Compilers for Machine Learning: Build a Tensor Compiler from Scratch

Follow George Hotz's 'Compilers for Machine Learning' syllabus and build the compiler it describes, in pure Python, with tinygrad as the reference you measure against. Start with the UOp graph and a pattern-matching rewrite engine that folds constants and simplifies symbolic index arithmetic, and render loopless programs to C that you compile and run. Add views: reshape, expand, permute, flip, pad and shrink become index arithmetic pushed down to buffer loads, so matmul and convolution fall out of movement ops and a reduce. Rangeify tensor graphs into loop nests with accumulators, split a model into kernels with staged buffers, and compile any model to C, slowly. Then make it fast: loop tiling, reordering and unrolling into register blocks turn a naive GEMM into one that runs an order of magnitude faster, and a fused online-softmax kernel is flash attention. Map loops onto GPU threads and render CUDA from the same IR. Finish with reverse-mode autodiff on the graph and train a tiny GPT entirely through your compiler, matching tinygrad's loss curve step for step.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** UOp
- [x] **2.** bounds
- [x] **3.** UPat
- [x] **4.** graph_rewrite
- [x] **5.** exec_alu
- [x] **6.** fold_div
- [x] **7.** make_symbolic
- [x] **8.** render_sink
- [x] **9.** eval_sink
- [x] **10.** compile_c
- [x] **11.** shape_of
- [x] **12.** Tensor
- [x] **13.** eval_tensor
- [x] **14.** conv2d
- [x] **15.** Kernel
- [x] **16.** Lowerer
- [x] **17.** ReduceLowerer
- [x] **18.** run_kernel_np
- [x] **19.** render_kernel
- [x] **20.** schedule
- [x] **21.** Compiled
- [x] **22.** split_range
- [x] **23.** unroll_output
- [x] **24.** unroll_reduce
- [x] **25.** optimize_gemm
- [x] **26.** flash_attention_kernel
- [x] **27.** to_gpu
- [x] **28.** render_cuda
- [x] **29.** grad_alu
- [x] **30.** grad_movement
- [x] **31.** backward
- [x] **32.** gpt_forward
- [x] **33.** build_train_program
- [x] **34.** train_gpt

## Results

```
1. Symbolic rewrites
   ((r0*4 + r1)*3 + r2) // 3 % 4 has 12 nodes; after simplification: 2 node (r1)
   loopless program: 10 lines of C before rewriting, 5 after

2. Compile any model to C
   MLP + attention block: 14 kernels (13 unique sources), compiled in 0.50s, max |C - NumPy| = 1.2e-07
   3x3 convolution from movement ops: one kernel, 5 loops, 18 loads per step, max err 0.0e+00

3. Fast kernels
   naive triple loop        1.9 GFLOP/s  (max err 1.9e-05)
   4x4 tile, K by 4        27.4 GFLOP/s  (max err 1.9e-05)
   4x8 tile, K by 8        56.2 GFLOP/s  (max err 1.9e-05)
   tinygrad CPU backend     8.9 GFLOP/s  (its own codegen and search, same compiler)
   speedup of the blocked kernel over the naive loop nest: 29.8x
   flash attention (N=128, d=32): 1 fused kernel, 34 accumulators, 0.44 ms, max err 3.6e-07; unfused schedule: 5 kernels, 2.10 ms

4. GPU lowering
   64x48 @ 48x32 as a CUDA kernel: grid (1, 16, 1), block (32, 4, 1), 21 lines, emulated max err 5.7e-06
   __global__ void gemm(float* data0, const float* data1, const float* data2) {
   if (gidx1 < 64 && gidx0 < 32) {

5. Autodiff and a tiny GPT
   gradient of an MLP loss vs central differences: max |diff| = 2.1e-04
   GPT (V=16, T=8, d=16): 67 kernels per step (62 unique), compiled in 1.80s, 3.9 ms/step
   loss every 5 steps: 2.708 2.263 0.859 0.324 0.224 0.136 0.115 0.091
   tinygrad comparison unavailable: AssertionError: 
   The whole training step, forward, loss, backward and SGD, is a program of compiled C kernels produced by this compiler.
```
