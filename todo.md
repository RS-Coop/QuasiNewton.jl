
# Symmetric tridiagonal eigensolve
stegr!, which is the default symmetric tridiagonal eigensolver, can often fail due to some LAPACK error. This can be somewhat mitigated by using stev! instead.

Doing the diagonal perturbation seems to mostly help, but it can alter the result. Seems to be error introduced by magnitude of the perturbation.

With noise 1e-6, there are only a couple of problem failures.

# Improvements
- Make sure typing and casting is all good
- Cannonical/Unified regularization calculation?
- Centralized fast norm?
- Typing of M?

# Documentation
- Docstrings, Julia->Documentation->Documentation
- Non-square least squares problem for README example?

# Features
- ADHvpOperator mul! without having to convert to Vector
- Custom Newton linesearch
- Alternate M update from Mishchenko
  - Or more robust M estimate with multiple random perturbations
- Regularization backtrack without skipping updates
- What happens if krylov processes for Newton or RN haven't converged?
- It would be nice to only use LinearOperator.jl, but they have no way of updating the operator in place
- Krylov subspace recycling
- Block Lanczos
- Better reorthogonalization?

# Method Additions
- Randomized coordinate projection (ARC and/or R-SFN)
- SketchySGD (how is their stepsize theoretically motivated)
