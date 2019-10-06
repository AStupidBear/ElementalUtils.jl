# Extensions to Elemental

## Installation

```julia
julia>]
pkg> add https://github.com/AStupidBear/ElementalUtils.jl.git
```

You may find this [script](https://gist.github.com/AStupidBear/2e1d0b21c1516b8d3624ded22f445dc3) useful if you encounter errors when building Elemental.


## Usage

```
using Elemental, ElementalUtils
```

Copy Julia's Array to Elemental's DistMatrix
```
A = Elemental.DistMatrix(Float32)
B = Elemental.DistMatrix(Float32)

copyto!(A, Float32[2 1; 1 2])
copyto!(B, Float32[4, 5])
```

Run distributed ridge regression ` ½|A*X-B|₂² + λ|X|₂²`
```
X = ElementalUtils.ridge(A, B, 0f0)
@assert isapprox(Array(X), [1, 2])
```

Run distributed lasso regression ` ½|A*X-B|₂² + λ|X|₁` (only supported in recent version of Elemental)
```
X = ElementalUtils.bpdn(A, B, 0.1f0)
```
