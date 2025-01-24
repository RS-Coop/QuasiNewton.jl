# QuasiNewton

A collection of Newton-type optimization algorithms.

### Authors: [Cooper Simpson](https://rs-coop.github.io/)

## License & Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

This repository can be cited using the GitHub action in the sidebar, or using the metadata in `CITATION.cff`. See [Publications](#publications) for a full list of publications related to R-SFN and influencing this package. If any of these are useful to your own work, please cite them individually.

## Methods

### R-SFN
A Julia implementation of the R-SFN algorithm: a second-order method for unconstrained non-convex optimization. To that end, we consider a problem of the following form
$$\min_{\mathbf{x}\in \mathbb{R}^n}f(\mathbf{x})$$
where $f:\mathbb{R}^n\to\mathbb{R}$ is a twice continuously differentiable function. Each iteration applies an update of the following form:
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}-\Big(\big(\nabla^2f(\mathbf{x}^{(k)})\big)^2+\lambda^{(k)}\mathbf{I}\Big)^{-1/2} \nabla f(\mathbf{x}^{(k)})$$
where the regularization term is $\lambda^{(k)}\propto||\nabla^2f(\mathbf{x}^{(k)})||$. The matrix inverse square root is computed via a quadrature approximation of the following integral:
$$\mathbf{A}^{-1/2}=\frac{2}{\pi}\int_{0}^{\infty}\big(t^2\mathbf{I}+\mathbf{A}\big)^{-1}\ dt$$
where $\mathbf{A}\in\mathbb{R}^{n\times n}$ has strictly positive spectrum, i.e $\sigma(\mathbf{A})\subset\mathbb{R}_{+ +}$.

### ARC
...

## Installation
This package can be installed just like any other Julia package. From the terminal, after starting the Julia REPL, run the following:
```julia
using Pkg
Pkg.add("QuasiNewton")
```
This will install the package and its direct dependencies, but in order to use the package you must install one of the following sets of packages for automatic differentiation (AD):
- `Enzyme.jl`
- `ReverseDiff.jl` and `ForwardDiff.jl`
- `Zygote.jl` and `ForwardDiff.jl`

### Testing
To test the package, run the following command in the REPL:
```julia
using Pkg
Pkg.test(test_args=["optional specific tests"])
```

## Usage
Load the package as usual:
```julia
using QuasiNewton
```
which will export the struct `RSFNOptimizer` and the `minimize!` function. Then load your AD packages, which will export a subtype of `HvpOperator`. Say you load Enzyme:
```julia
using Enzyme
```
then the `EHvpOperator` will be available.

Let's look at a two dimensional Rosenbrock example:
```julia
function rosenbrock(x)

	res = 0.0
	for i = 1:size(x,1)-1
		res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
	end

	return res

end

x = [0.0, 0.0]

opt = RSFNOptimizer(size(x,1))

minimize!(opt, x, rosenbrock, itmax=10)
```

## Publications

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
