# Improvements
- Make sure typing and casting is all good
- Cannonical/Unified regularization calculation?
- Centralized fast norm?
- Typing of M?
- Should we be reorthogonalizing in Lanczos?
  - Does this let us use `stegr`?
- Do we need the tolerance we are asking for in LFA?

# Documentation
- Docstrings, Julia->Documentation->Documentation
- Non-square least squares problem for README example?

# Features
- ADHvpOperator mul! without having to convert to Vector
- Custom Newton linesearch
- Alternate M update from Mishchenko
- Regularization backtrack without skipping updates
- What happens if krylov processes for Newton or RN haven't converged?
- It would be nice to only use LinearOperator.jl, but they have no way of updating the operator in place
- Krylov subspace recycling

# Method Additions
- Randomized coordinate projection (ARC and/or R-SFN)
- SketchySGD (how is their stepsize theoretically motivated)
