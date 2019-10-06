module ElementalUtils

using Elemental

using Elemental: MC, MR, comm
using Elemental: ElementalMatrix, DistMatrix, DistSparseMatrix, DistMultiVec
using Elemental: libEl, Orientation, NORMAL, ElError
using Elemental: zeros!, queueUpdate, processQueues, queuePull, processPullQueue
using Elemental: leastSquares, leastSquares!
using Elemental.MPI: commRank

export bpdn!, bpdn, ridge!, ridge

for (elty, relty, ext) in ((:Float32, :Float32, :s),
                           (:Float64, :Float64, :d))
    for (matA, matB, sym) in ((:(Elemental.Matrix), :(Elemental.Matrix), "_"),
                              (:DistMatrix, :DistMatrix, "Dist_"),
                              (:(Elemental.SparseMatrix), :(Elemental.Matrix), "Sparse_"),
                              (:DistSparseMatrix, :DistMultiVec, "DistSparse_"))
        @eval begin
            function bpdn!(A::$matA{$elty}, B::$matB{$elty}, lambda::$elty, X::$matB{$elty})
                ElError(ccall(($(string("ElBPDNX", sym, ext)), libEl), Cuint,
                    (Ptr{Cvoid}, Ptr{Cvoid}, $elty, Ptr{Cvoid}, Cuint),
                    A.obj, B.obj, lambda, X.obj, 0))
                return X
            end
        end
    end
end

function bpdn(A::DistMatrix{T}, B::DistMatrix{T}, lambda::T) where {T}
    X = DistMatrix(T, MC, MR, A.g)
    return bpdn!(A, B, lambda, X)
end

function bpdn(A::DistSparseMatrix{T}, B::DistMultiVec{T}, lambda) where {T}
    X = DistMultiVec(T, comm(A))
    return bpdn!(A, B, lambda, X)
end

@enum RidgeAlg RIDGE_CHOLESKY RIDGE_QR RIDGE_SVD

for (elty, relty, ext) in ((:Float32, :Float32, :s),
                           (:Float64, :Float64, :d))
    for (matA, matB, sym) in ((:(Elemental.Matrix), :(Elemental.Matrix), "_"),
                              (:DistMatrix, :DistMatrix, "Dist_"),
                              (:(Elemental.SparseMatrix), :(Elemental.Matrix), "Sparse_"),
                              (:DistSparseMatrix, :DistMultiVec, "DistSparse_"))
        @eval begin
            function ridge!(A::$matA{$elty}, B::$matB{$elty}, gamma::$elty, X::$matB{$elty};
                orientation::Orientation = NORMAL, alg::RidgeAlg = RIDGE_CHOLESKY)
                ElError(ccall(($(string("ElRidge", sym, ext)), libEl), Cuint,
                    (Cuint, Ptr{Cvoid}, Ptr{Cvoid}, $elty, Ptr{Cvoid}, Cuint),
                    orientation, A.obj, B.obj, gamma, X.obj, alg))
                return X
            end
        end
    end
end

function ridge(A::DistMatrix{T}, B::DistMatrix{T}, gamma::T; ka...) where {T}
    X = DistMatrix(T, MC, MR, A.g)
    return ridge!(A, B, gamma, X; ka...)
end

function ridge(A::DistSparseMatrix{T}, B::DistMultiVec{T}, gamma::T; ka...) where {T}
    X = DistMultiVec(T, comm(A))
    return ridge!(A, B, gamma, X; ka...)
end

function Base.copyto!(dest::DistMatrix{T}, src::Base.AbstractVecOrMat) where {T}
    m, n = size(src, 1), size(src, 2)
    zeros!(dest, m, n)
    if commRank(comm(dest)) == 0
        for j = 1:n
            for i = 1:m
                queueUpdate(dest, i, j, src[i, j])
            end
        end
    end
    processQueues(dest)
    return dest
end

Base.copyto!(dest::DistMatrix, src::ElementalMatrix) = Elemental._copy!(src, dest)

function Base.copyto!(dest::Base.VecOrMat, src::DistMatrix{T}) where {T}
    m, n = size(src, 1), size(src, 2)
    if commRank(comm(src)) == 0
        for j = 1:n
            for i = 1:m
                queuePull(src, i, j)
            end
        end
    end
    dest_mat = ndims(dest) == 1 ? reshape(dest, :, 1) : dest
    processPullQueue(src, dest_mat)
    return dest
end

function Base.convert(::Type{Array}, xd::DistMatrix{T}) where {T}
    x = zeros(T, size(xd))
    copyto!(x, xd)
end

Base.Array(xd::DistMatrix) = convert(Array, xd)

end # module
