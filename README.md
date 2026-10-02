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

---

Built on Deep-ML.
