# Known Constraints

## Current primary split

The current `pancreas_seed42` split is label-stratified. It is appropriate for the main same-distribution benchmark, but it is not a leave-one-batch-out split and should not be described as cross-batch generalization.

## scFoundation device behavior

The upstream `scFoundation` loader expected CUDA by default. The local copy now resolves to CPU automatically when CUDA is unavailable so that model loading can succeed on this machine.

The GitHub upload does not include the cloned `third_party/scFoundation` repository. A full scFoundation rerun requires cloning `https://github.com/biomap-research/scFoundation.git` into `third_party/scFoundation`. On a CPU-only machine, the upstream loader may need the same CPU-compatible loading adjustment before `scripts/10_run_scfoundation_linear_probe.py` can run end-to-end.
