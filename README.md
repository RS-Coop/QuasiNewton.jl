# QuasiNewton.jl

A collection of quasi-Newton optimization algorithms.

### Author: [Cooper Simpson](https://rs-coop.github.io/)

**DISCLAIMER**: This package is still very much a work in progress and certainly not everything has been tested thoroughly. For example, the AD functionality has only been tested (in a limited capacity) with the Enzyme.jl backend.

## License & Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

This repository can be cited using the GitHub action in the sidebar, or using the metadata in `CITATION.cff`. See [Publications](#publications) for a full list of publications related to R-SFN and influencing this package. If any of these are useful to your own work, please cite them individually.

## Contributing
Feel free to open issues, ask questions, or otherwise contribute!

## Methods
Our use of the term *quasi* should not be taken in the traditional sense, i.e. only in reference to methods such as BFGS, but instead via its true definition to describe algorithms that are *similar* to or *resemble* Newton's method. As such, our algorithms are intended for second-order unconstrained convex or non-convex optimization, i.e. we consider problems of the following form
```math
\min_{\mathbf{x}\in \mathbb{R}^n}f(\mathbf{x})
```
where $`f:\mathbb{R}^n\to\mathbb{R}`$ is a twice continuously differentiable function, possibly with a Lipschitz continuous Hessian. We then consider updates of the following form:
```math
\mathbf{x}_{(k+1)} = \mathbf{x}_{(k)} - \eta_{(k)}\mathbf{B}_{(k)}^{-1}\nabla f(\mathbf{x}_{(k)})
```
where $`\mathbf{B}_{(k)}`$ is some matrix constructed as a function of the Hessian $`\mathbf{H}_{(k)} = \nabla^2 f(\mathbf{x}_{(k)})`$, recovering different algorithms.

All algorithms are implemented matrix-free using Krylov subspace methods to compute the update and can leverage mixed-mode automatic differentiation to compute cheap Hessian-vector products.

### (Regularized) Newton
```math
\mathbf{B}_{(k)} = \mathbf{H}_{(k)} + \lambda_{(k)}\mathbf{I}
```
where choosing $`\lambda_{(k)}=0`$ recovers vanilla Newton's method and $`\lambda_{(k)}\propto\|\nabla f(\mathbf{x}_{(k)})\|^{1/2}`$ recovers the [regularized Newton's method](https://epubs.siam.org/doi/10.1137/22M1488752).

### Regularized Saddle-Free Newton (R-SFN)
```math
\mathbf{B}_{(k)} = \left(\mathbf{H}_{(k)}^2 + \lambda_{(k)}\mathbf{I}\right)^{1/2}
```
where the regularization term is $`\lambda_{(k)}\propto\|\nabla f(\mathbf{x}_{(k)})\|`$. The matrix function in question is a smooth approximation to the absolute value and is computed via the Lanczos process. See [Publications](#publications) for more details.

### Adaptive Regularization with Cubics (ARC)
```math
\mathbf{B}_{(k)} = \mathbf{H}_{(k)} + \lambda_{(k)}\mathbf{I}
```
where the analytic choice of regularization is $`\lambda_{(k)}\propto\|\mathbf{x}_{(k+1)} - \mathbf{x}_{(k)}\|`$, but in practice this is chosen adaptively. This particular implementation is based of [ARCqK](https://link.springer.com/article/10.1007/s10107-023-02007-6).

## Installation
This package can be installed just like any other Julia package. From the terminal, after starting the Julia REPL, run the following:
```julia
using Pkg
Pkg.add("QuasiNewton")
```
which will install the package and its direct dependencies. For automatic differentiation (AD), we use [DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/), which is agnostic to the backend. Backends such as `Enzyme.jl` or `Zygote.jl` can be installed by the user and specified in the optimization call. See [JuliaDiff](https://juliadiff.org/) for current details on the Julia AD ecosystem.

### Testing
To test the package, run the following command in the REPL:
```julia
using Pkg
Pkg.test(test_args=[<"optional specific tests">], coverage=true)
```

## Usage
Loading the package as usual
```julia
using QuasiNewton
```
will export four in-place optimization routines: `minimize!`, `newton!`, `rsfn!`, and `arc!`. We will only cover the details of the first, as it covers the others as well. The optimization algorithm can be specified via a symbol and depending on the function information provided, the method is dispatched to an AD or [LinearOperator.jl](https://jso.dev/LinearOperators.jl/latest/) based execution. All methods return a `Stats` object containing relevant information about the procedure, with some information only being stored if `history=true`.

First, we consider the following example of minimizing a simple least-squares problem with Newton's method.
```julia
using LinearAlgebra #for norm
using LinearOperators #for Hessian

m = 100; n = 50 #problem size

A = randn(m,n) #random data
b = A*randn(n) #random rhs

f(x) = 0.5*norm(A*x-b)^2 #objective function
function fg!(g, x) #computes gradient in-place and returns objective function
	g .= A'*(A*x - b)
	return f(x)
end
H(x) = LinearOperator(A'*A) #computes Hessian

stats = minimize!(randn(n), f, fg!, H, Val(:newton);
					max_iter=100,
					max_time=60,
					posdef=true,
					M=0.0,
					history=true)

show(stats)
#=
Converged:                    true
Iterations:                      1
Runtime (s):             6.72e-02
Minimum:                  1.56e-12
Gradient Norm:            1.52e-05

Evaluations:
      Total:                    36
   Function:                     3
   Gradient:                     2
    Hessian:                    31

Residual Norm:
          Max:            0.00e+00
          Avg:            0.00e+00

Regularization:
          Max:            0.00e+00
          Avg:            0.00e+00

Krylov Iterations:
          Max:            3.10e+01
          Avg:            3.10e+01

Status:                  Nominal
=#
```

Second, we consider the following example of minimizing the non-convex Rosenbrock function with R-SFN.
```julia
using DifferentiationInterface #for AD
import Enzyme #for AD

function rosenbrock(x) #objective function
	res = 0.0
	for i = 1:size(x,1)-1
		res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
	end

	return res
end

stats = minimize!(zeros(10), rosenbrock, Val(:rsfn), AutoEnzyme();
					max_iter=100,
					max_time=60,
					M=1e-8,
					history=true)

show(stats)
#=
Converged:                    true
Iterations:                     30
Runtime (s):             1.68e-04
Minimum:                  7.04e-14
Gradient Norm:            1.58e-06

Evaluations:
      Total:                   315
   Function:                    80
   Gradient:                    31
    Hessian:                   204

Residual Norm:
          Max:            6.30e+00
          Avg:            1.03e+00

Regularization:
          Max:            4.79e-07
          Avg:            9.81e-08

Krylov Iterations:
          Max:            1.00e+01
          Avg:            6.80e+00

Status:                  Nominal
=#
```

## Publications
More information about R-SFN in can be found in the following publications. Please cite them if they are useful for your own work!

### [Regularized Saddle-Free Newton: Saddle Avoidance and Efficient Implementation](https://rs-coop.github.io/projects/research/rsfn)
```bibtex
@mastersthesis{rsfn,
	title = {{Regularized Saddle-Free Newton: Saddle Avoidance and Efficient Implementation}},
	author = {Cooper Simpson},
	school = {Dept. of Applied Mathematics, CU Boulder},
	year = {2022},
	type = {{M.S.} Thesis}
}
```
