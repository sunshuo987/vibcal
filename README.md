# vibcal

Research sandbox for computing vibrational spectra of polyatomic molecules with tree tensor network techniques. The repository bundles the upstream `PyTreeNet` library together with experiment scripts and reference data used to benchmark Lobpcg-based solvers on realistic molecular potentials.

## Overview

- Uses tree tensor network states/operators to solve high-dimensional vibrational Hamiltonians.
- Provides reproducible experiments for coupled harmonic oscillators and the acetonitrile (CH₃CN) benchmark problem.
- Stores selected numerical outputs and profiler traces for later inspection.

## Repository Layout

- `PyTreeNet/` – vendored copy of the `pytreenet` library (EUPL v1.2) powering all tensor network routines.
- `Experiment/` – runnable scripts, potentials, and utilities for setting up vibrational calculations (`run_logpcg_block.py`, `run_logpcg_block_harmonic-oscillator.py`, etc.).
- `Results_final/` – example outputs (energies, plots, profiler dumps) created by the experiment scripts.
- `.venv/` – optional local Python virtual environment (ignored in version control).
- `LICENSE` – licensing information inherited from `PyTreeNet`.

## Getting Started

1. **Prerequisites**  
   Python 3.8.10 or newer (per `PyTreeNet`), plus system BLAS/LAPACK for NumPy/SciPy.

2. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd PyTreeNet/
   pip install -e .
   cd ..
   pip install numpy scipy pandas matplotlib tqdm h5py 
   ```
   Add any extra packages you need (e.g., `ipykernel` for notebooks).

## Running Experiments

- **CH₃CN vibrational spectrum**
  ```bash
  python Experiment/run_logpcg_block.py
  ```
  Produces optimized states, energy traces, and profiler output under `Results_final/ch3cn/`.

- **Coupled harmonic oscillator benchmark**
  ```bash
  python Experiment/run_logpcg_block_harmonic-oscillator.py
  ```

Each script accepts optional parameters (see source) to adjust topology (`mps`, `threetree`, `leafonly`), maximum bond dimension, and output folders.

## Reference Data

- `Experiment/ch3cn_ref.csv` holds target energy levels used to validate the simulations.
- Potential definitions for CH₃CN, formaldehyde, and generic harmonic models live in `Experiment/potentials/`.

## Documentation & Help

- The vendored `PyTreeNet` documentation is available online at [pytreenet.readthedocs.io](https://pytreenet.readthedocs.io/).
- For tensor network theory and algorithmic background, see the references listed in `PyTreeNet/README.rst`.

## Contributing

Open issues or submit pull requests against this repository if you extend the experiments. For changes to the core tensor network library, contribute upstream to [PyTreeNet](https://github.com/Drachier/PyTreeNet).

## License

This project carries the EUPL v1.2 license through the bundled `PyTreeNet` library. See `LICENSE` for full details.
