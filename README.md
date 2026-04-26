# Access Tracking in GPUs for NVIDIA UVM Based Workflows

## Group Members

| Name                  | Roll No | Contributions |
| --------------------- | ------- | ------------- |
| Ravi Arora            | 230846  | 20%           |
| Ashik Stenny          | 230224  | 20%           |
| Pothuganti Nikhil     | 230760  | 20%           |
| Spandan Pati          | 231031  | 20%           |
| Ruthvik Tunuguntla    | 220924  | 20%           |

## Declaration and Undertaking

All project members own and give their consent on everything submitted as part of this submission. The work done and submitted in the project are done by the members of the project group mentioned above. Any use of external sources such as source code from other repositories, GPT generated source code/documentation/figures etc. are properly attributed.

## External Sources

- [NVIDIA Open GPU Kernel Modules](https://github.com/NVIDIA/open-gpu-kernel-modules) – used as reference and for the induced page fault approach (nvidia-470.256.02 Open Source UVM GPU Kernel Modules).
- CUDA Oversubscribed Benchmarks – provided by TA Pranjal, used to test performance.

## AI Assistant Usage

We have used LLMs and Agentic AI to speed up development by offloading redundant tasks such as generating boilerplate code, documentation, compilation scripts, shell scripts, writing tests, generating plots, and plot analysis.  
LLMs were also occasionally used for debugging and understanding difficult concepts, as well as analyzing flaws in our logic and architecture. In no instance did we use AI without first developing our own logic and architecting the solution.

A good example is the use of AI in setting up files like `benchmark.py`, `tests/*/test.py`, and similar files in the Compiler Pass Approach.  
For core project files, AI usage was minimal (e.g., inline completions to speed up syntax writing, documenting code, or minor rewrites) and never relied upon to solve problems for us.

## Project Overview

This project implements two complementary approaches for tracking memory access patterns in NVIDIA GPUs under Unified Virtual Memory (UVM):

1. **Fault‑Based Approach** – kernel module modification to intercept page faults.
2. **Compiler Pass Approach** – static instrumentation of CUDA kernels.

## Fault-Based Approach

To run the fault-based approach:

```bash
cd fault_based_approach
./pre.sh   # install driver
./post.sh  # patch the code
```

Source code: [https://github.com/nikhilpothuganti/uvm_access_tracker](https://github.com/nikhilpothuganti/uvm_access_tracker)

---

## Compiler Pass Approach

Source code: [https://github.com/PhantomzBack/uvm_access_tracking_1](https://github.com/PhantomzBack/uvm_access_tracking_1)