# Improvements/Features
- What happens if krylov processes for Newton or RN haven't converged?
- What should be the (default) Krylov tolerance?
- Krylov subspace recycling
- Should we be reorthogonalizing in Lanczos?
- Do we need the tolerance we are asking for in LFA?
- Probably should be passing `M=missing` when we want autoestimation or something other than `M=NaN`
    - Same thing with how we pass `tol` in some places
- Another linesearch for regularized Newton in the convex case
- Correct tolerance for Newton?
- Extension for other modeling packages
    - SciML Optimization.jl
    - JuliaSmoothOptimizers NLPModels.jl)
- Be better about type rigidty
  - When promoting (i.e., M::R=10 and then R(1e16)), how to deal with possible Int
  - ARC solver shifts ignores type
- Can the mutable but mostly const structs (thinking LFASolver) be made immutable with some cleverness
- Can methods be less allocating
  - Does multiplication by negative, e.g., Newton -g allocated
  - ARC
- Instead of discarding computed values when `history=false` we could just track the running max/mean somehow
- Make sure evals are being tracked
  - I think this is just the operator now
- How should I name the QuasiNewtonStats update functions and/or deal with that in general
- The `reset!` call in `minimize!` potentially reruns the AD prep, maybe there is a better way of doing this?

# Method Additions
- Randomized coordinate projection (ARC and/or R-SFN)
- SketchySGD (how is their stepsize theoretically motivated)
- LBFGS
