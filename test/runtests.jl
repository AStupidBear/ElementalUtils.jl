using Elemental
using ElementalUtils
using Test

A = Elemental.DistMatrix(Float32)
B = Elemental.DistMatrix(Float32)
X = Elemental.DistMatrix(Float32)

copyto!(A, Float32[2 1; 1 2])
copyto!(B, Float32[4, 5])

Elemental.leastSquares!(A, B, X)
@test isapprox(Array(X), [1, 2])

ElementalUtils.ridge!(A, B, 0f0, X)
@test isapprox(Array(X), [1, 2])
