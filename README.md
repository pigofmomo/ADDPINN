# ADD-PINN

This repository contains the official research code for **ADD-PINN: Adaptive Domain Decomposition based Physics-Informed Neural Networks via Spatial Clustering**. The paper has been accepted by **Communications in Computational Physics (CICP)** and is awaiting publication.

ADD-PINN extends standard physics-informed neural networks (PINNs) with an adaptive spatial domain-decomposition workflow. The code supports training a single global PINN, XPINN-style decomposed models, and ADD-PINN variants that use clustering, banded interfaces, balanced initialization, gradient-analysis-based synchronization control, and optional subdomain adjustment during training.

## Highlights

The implementation follows the main contributions of the ADD-PINN paper:

- **Subdomain shape manipulation via spatial clustering.** Polygonal blocks are autonomously grouped into irregular subdomains, and sampling points are generated inside those subdomains so the method can accommodate solution domains with complex shapes.
- **Banded interface condition with improved subnetwork coupling.** ADD-PINN uses banded interface regions, an improved loss design, and synchronization-frequency control based on gradient analysis to make information exchange between subnetworks more efficient.
- **Graph-optimization-based initialization and dynamic adjustment.** Domain initialization and adaptive subdomain adjustment are formulated with graph optimization to improve convergence speed and prediction accuracy.
- **Reproducible benchmark scripts.** The repository includes examples for 2-D function approximation, Poisson-Boltzmann-type problems, heat equations, lid-driven cavity resources, and complex-shape domains.

## Repository layout

```text
pinn_add/
├── pinnadd.py                 # Main PINNADD model orchestration class
├── submodel.py                # Per-subdomain model wrapper
├── subdata.py                 # PDE/function data abstractions
├── pointset2d.py              # 2-D point-set sampling and filtering helpers
├── optimize_graph.py          # Graph-based domain-decomposition optimization
├── config_pinnadd.py          # Global training and decomposition configuration
├── function2d/                # 2-D function approximation experiment
├── poisson2d/                 # Poisson-Boltzmann experiment and reference data
├── heat2d/                    # Heat-equation experiment
├── ldc/                       # Lid-driven cavity resources
└── complexshape/              # Complex-domain resources and visualizations
```

## Installation

> The project is research code and currently does not include a packaged `requirements.txt`. The dependency list below is inferred from the source files.

Create and activate a Python environment, then install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install deepxde torch numpy scipy scikit-learn shapely networkx matplotlib profilehooks
```

If you use a CUDA-enabled PyTorch build, install `torch` according to the instructions for your platform before installing the remaining packages.

## Quick start

Run an example from its own directory so that local reference-data paths resolve correctly.

### 2-D function approximation

```bash
cd pinn_add/function2d
python -m pinn_add.function2d.function2d
```

The script defines a synthetic target function, creates domain samples on `[-1, 1] x [-1, 1]`, and exposes helpers for standard PINN, XPINN, and ADD-PINN training.

### Poisson-Boltzmann example

```bash
cd pinn_add/poisson2d
python poisson2d.py
```

This example uses a rectangular domain with circular holes, Dirichlet boundary conditions, and reference data from `poisson_boltzmann2d.dat`.

### Heat-equation example

```bash
cd pinn_add/heat2d
python heat2d.py
```

This example solves a time-dependent heat equation over a rectangular spatial domain with initial and boundary conditions supplied through DeepXDE.

## Typical workflow

1. **Define the geometry and point set** with DeepXDE geometry objects and `pinn_add.PointSets2D`.
2. **Define the problem** as either:
   - a callable target function through the `"function"` key, or
   - a PDE residual through the `"pde"` key, plus boundary/initial conditions in `"icbc"`.
3. **Choose subdomain networks** by passing one network layer list for a global PINN or multiple layer lists for decomposed models.
4. **Configure decomposition behavior** with options such as:
   - `interface_type="line"` for XPINN-style interfaces,
   - `interface_type="band"` for ADD-PINN interface bands,
   - `initial_subdomain_balance=True` for balanced initialization,
   - `adjust_subdomain=True` for adaptive subdomain updates.
5. **Train, save, restore, and test** through the `PINNADD` class.

Minimal sketch:

```python
import deepxde as dde
import pinn_add

spatial_domain = dde.geometry.Rectangle([-1, -1], [1, 1])
pointset = pinn_add.PointSets2D(spatial_domain, divide_config=None)

nets = [
    [2, 50, 50, 50, 50, 1],
    [2, 50, 50, 50, 50, 1],
    [2, 50, 50, 50, 50, 1],
]

data = {
    "function": target_function,
    "pointset": pointset,
    "num_domain": 10000,
    "n_clusters": 500,
    "interface_type": "band",
    "initial_subdomain_balance": True,
    "adjust_subdomain": True,
}

model = pinn_add.PINNADD(nets, data, save_dir="./pinnadd_result/0/")
result = model.train(iterations=20000, print_every=100)
model.save("./pinnadd_result/0/", result)
```

## Configuration

Global defaults are stored in `pinn_add/config_pinnadd.py`. Common knobs include:

- random seed control (`random_seed`, `set_random_seed`)
- learning-rate decay (`lr_decay`)
- pretraining length (`pretrain_iterations`)
- adaptive subdomain update schedule (`adjust_subdomain_start_iteration`, `adjust_subdomain_end_iteration`, `adjust_subdomain_period`)
- graph-optimization tolerances and balancing parameters

Example scripts override these values near the top of each experiment file.

## Outputs

Training scripts save results under experiment-specific folders such as:

- `pinn_result/<seed>/`
- `xpinn_result/<seed>/`
- `pinnadd_result/<seed>/`
- `pinnadd_result2/<seed>/`
- `pinnadd_result3/<seed>/`

Typical outputs include serialized model state, metrics, figures, and decomposition artifacts, depending on the example.

## Notes

- Run long experiments on a machine with sufficient CPU/GPU resources; the default examples use thousands of collocation points and up to 20,000 training iterations.
- Some example scripts assume they are launched from their own directory because they load local `.dat` reference files.
- The codebase is intended to accompany the ADD-PINN paper and may evolve as the manuscript and experiments are finalized.

## Paper and citation

The ADD-PINN paper has been accepted by **Communications in Computational Physics (CICP)** and is currently awaiting publication. Please cite the paper if you use this repository in academic work. The final bibliographic information and paper URL will be filled in after publication.

```bibtex
@article{addpinn_cicp_todo,
  title   = {ADD-PINN: Adaptive Domain Decomposition based Physics-Informed Neural Networks via Spatial Clustering},
  author  = {TODO},
  journal = {Communications in Computational Physics},
  year    = {TODO},
  volume  = {TODO},
  number  = {TODO},
  pages   = {TODO},
  doi     = {TODO},
  url     = {TODO}
}
```

- Paper URL: TODO
- Project page / documentation URL: TODO
