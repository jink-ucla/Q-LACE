# Environment record

CPU tests require Python >=3.10, NumPy, pytest and Pillow. They do not invoke
model loading, remote services, or restricted inputs. Package verification uses
offline environment flags and an extracted copy outside the research tree.

Observed new inference environment: Python 3.12; Torch 2.9.0+cu128; PEFT 0.17.1;
Accelerate 1.11.0; Pillow 12.3.0. The imported Transformers source identifies as
4.57.0.dev0, while installed distribution metadata identifies as 5.0.0.dev0.
This discrepancy is retained; installing either version string alone is **not**
a demonstrated reconstruction of that executing environment. H100 NVL and L40S
were both used, including different hardware for the two base preprocessing
arms. Numerical parity across hardware was not established.

GPU dependencies are listed as an inventory in `requirements-gpu.txt`, not a
validated lockfile. The archive does not vendor installed libraries or a
container. Consult model documentation for a compatible runtime, then validate
inputs, smoke outputs and source hashes before a new run. No promise is made
that a fresh pip installation reproduces historical arithmetic.
