# Imporvements/Features
- ADHvpOperator mul! without having to convert to Vector
- Custom Newton linesearch
- Regularization backtrack without skipping updates
- What happens if krylov processes for Newton or RN haven't converged?
- It would be nice to only use LinearOperator.jl, but they have no way of updating the operator in place
- Krylov subspace recycling
- Should we be reorthogonalizing in Lanczos?
  - Does this let us use `stegr`?
- Do we need the tolerance we are asking for in LFA?
- Probably should be passing `M=missing` when we want autoestimation or something other than `M=NaN`

# Method Additions
- Randomized coordinate projection (ARC and/or R-SFN)
- SketchySGD (how is their stepsize theoretically motivated)