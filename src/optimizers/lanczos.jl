#=
Author: Cooper Simpson
=#

#########################################################
# Scalar Lanczos
#########################################################

struct LanczosWorkspace{R<:AbstractFloat}
    Q::Matrix{R} # orthogonal basis
    d::Vector{R} # diagonal elements
    dl::Vector{R} # lower (and upper) diagonal elements
end

function LanczosWorkspace{R}(n::Int, max_depth::Int) where {R<:AbstractFloat}
    Q = Matrix{R}(undef, n, max_depth+1)
    d = Vector{R}(undef, max_depth)
    dl = Vector{R}(undef, max_depth)

    return LanczosWorkspace(Q, d, dl)
end

"""
"""
function lanczos(Z::M, ω::S, k::Int; reorthogonalize::Bool=false) where {R, S<:AbstractVector{R}, M<:AbstractMatrix{R}}
    m, n = size(Z)
	m == n || throw(DimensionMismatch("Lanczos requires a square operator"))

    Q = Matrix{R}(undef, n, k+1)
	d = Vector{R}(undef, k) # diagonal elements
	dl = Vector{R}(undef, k) # lower (and upper) diagonal elements

    workspace = LanczosWorkspace(Q, d, dl)

    return lanczos(workspace, Z, ω, k; reorthogonalize=reorthogonalize)
end

"""
Scalar Lanczos process.

# Arguments
- `workspace::LanczosWorkspace`: Preallocated workspace
- `Z::Matrix`: Symmetric matrix
- `ω::Vector`: Vector
- `k::Int`: Krylov subspace depth
- `reorthogonalize::Bool`: Whether to perform partial reorthogonalization.
"""
function lanczos(workspace::LanczosWorkspace{R}, Z::M, ω::S, k::Int; reorthogonalize::Bool=false) where {R, S<:AbstractVector{R}, M<:AbstractMatrix{R}}
	m, n = size(Z)
	m == n || throw(DimensionMismatch("Lanczos requires a square operator"))

    # Setup
	β₁ = zero(R)

	Q = @view workspace.Q[:,1:k+1]

	d = @view workspace.d[1:k]
	dl = @view workspace.dl[1:k]

    # Initialize
    β₁ = twonorm(ω)
    β₁ == 0 && error("Exact breakdown β₁ == 0.")
    @views q₁ = Q[:,1]
    copyto!(q₁, ω)
    rmul!(q₁, inv(β₁))

	@views for i = 1:k
		qᵢ = Q[:,i]
		qᵢ₊₁ = Q[:,i+1]

		mul!(qᵢ₊₁, Z, qᵢ)

		if i ≥ 2
			qᵢ₋₁ = Q[:,i-1]
			βᵢ = dl[i-1]
			axpy!(-βᵢ, qᵢ₋₁, qᵢ₊₁)
		end

		αᵢ = dot(qᵢ, qᵢ₊₁)
		axpy!(-αᵢ, qᵢ, qᵢ₊₁)

		# Selective reorthogonalization against last two vectors.
		if reorthogonalize
			if i ≥ 2
				βtmp = dot(qᵢ₋₁, qᵢ₊₁)
				dl[i-1] += βtmp
				axpy!(-βtmp, qᵢ₋₁, qᵢ₊₁)
			end

			αtmp = dot(qᵢ, qᵢ₊₁)
			αᵢ += αtmp
			axpy!(-αtmp, qᵢ, qᵢ₊₁)
		end

		d[i] = αᵢ
		βᵢ₊₁ = twonorm(qᵢ₊₁)

		if βᵢ₊₁ ≤ eps(R)
			fill!(qᵢ₊₁, zero(R))
		else
            rmul!(qᵢ₊₁, inv(βᵢ₊₁))
		end

		dl[i] = βᵢ₊₁
	end

	return @views Q[:,1:k], SymTridiagonal(d, dl[1:k-1]), dl[end]
end

#########################################################
# Block Lanczos
#########################################################

"""
Block Lanczos process.

# Arguments
- `Z::Matrix`: Symmetric matrix
- `Ω::Vector`: Matrix
- `k::Int`: Krylov subspace depth
- `reorthogonalize::Bool`: Whether to perform partial reorthogonalization.
"""
function block_lanczos(Z::M1, Ω::M2, k::Int; reorthogonalize::Bool=false) where {R<:AbstractFloat, M1<:AbstractMatrix{R}, M2<:AbstractMatrix{R}}
    m, n = size(Z)
	p, b = size(Ω)
    m == n || throw(DimensionMismatch("Lanczos requires a square operator"))
    n == p || throw(DimensionMismatch("Lanczos requires a compatible block"))

	# Preallocate
    Q = zeros(R, n, (k+1)*b)
	T = zeros(R, k*b, k*b) # dense block-tridgiagonal

    A_i = zeros(R, b, b)
    B_i = zeros(R, b, b)
    B_ip1 = zeros(R, b, b)

	if reorthogonalize
		ABtmp = zeros(R, b, b)
	end

	QAi = zeros(R, n, b)

    # Initial block normalization; identical to scalar case when b=1
    V, R1 = qr(Ω)  # reduced QR
	Q[:,1:b] .= Matrix(V)
    B1 = UpperTriangular(R1)

    for i = 1:k
        # Location of blocks in Q and T
        blk = (i-1)*b+1 : i*b
        blk_next = i*b+1 : (i+1)*b

        V_i = view(Q, :, blk)
        V_next = view(Q, :, blk_next)

		mul!(QAi, Z, V_i)

        # Subtract V_{i-1}*B_i
        if i ≥ 2
            V_prev = view(Q, :, blk .- b)
            QAi .-= V_prev*B_i
        end

		mul!(A_i, V_i', QAi)
        QAi .-= V_i*A_i

        # Selective reorthogonalization against last two blocks
        if reorthogonalize
            if i > 1
                mul!(ABtmp, V_prev', QAi)
                B_i .+= ABtmp
                QAi .-= V_prev * ABtmp
            end

			mul!(ABtmp, V_i', QAi)
            A_i .+= ABtmp
            QAi .-= V_i * ABtmp
        end

        T[blk, blk] .= A_i

        # Orthogonalize
        V, Rnew = qr(QAi)
		V_next[:,:] .= Matrix(V)
        B_ip1 .= UpperTriangular(Rnew)

        if i < k
            blk_below = blk_next
            T[blk_below, blk] .= B_ip1
            T[blk, blk_below] .= B_ip1'
        end

        B_i .= B_ip1
    end

    return Q[:,1:k*b], Symmetric(T), B1
end

"""
Potentially an improved Block Lanczos function. Not tested or guaranteed to work at all.
"""
function block_lanczos_fa(Z::M1, Ω::M2, k::Int; reorthogonalize::Bool=false) where {R<:AbstractFloat, M1<:AbstractMatrix{R}, M2<:AbstractMatrix{R}}

    m, n = size(Z)
    p, b = size(Ω)
    @assert m == n && n == p

    # Preallocate
    Q = zeros(R, n, (k+1)*b)

    AB = [zeros(R, b, b) for _ in 1:k]   # diagonal blocks A_i
    BB = [zeros(R, b, b) for _ in 1:k]   # off-diagonal blocks B_{i+1}

    if reorthogonalization
        ABtmp = zeros(R, b, b)
    end

    # === Normalize initial block ===
    V, R1 = qr(Ω)
    Q[:,1:b] .= Matrix(V)
    B1 = UpperTriangular(R1)

    # Allocate W once
    W = zeros(R, n, b)

    # === Lanczos loop ===
    for i = 1:k
        blk      = (i-1)*b+1 : i*b
        blk_next = i*b+1 : (i+1)*b

        V_i = view(Q, :, blk)

        # W = Z * V_i
        mul!(W, Z, V_i)

        # subtract previous block
        if i > 1
            V_prev = view(Q, :, blk .- b)
            W .-= V_prev * BB[i-1]      # <-- corrected sign
        end

        # A_i = V_i' * W
        mul!(AB[i], V_i', W)

        # subtract projection onto current block
        W .-= V_i * AB[i]

        # optional reorthogonalization
        if reorthogonalization
            if i > 1
                mul!(ABtmp, V_prev', W)
                W .-= V_prev * ABtmp
            end
            mul!(ABtmp, V_i', W)
            W .-= V_i * ABtmp
        end

        # QR → next block
        if i < k
            Vn, RR = qr(W)
            Q[:, blk_next] .= Matrix(Vn)
            BB[i] .= UpperTriangular(RR)
        end
    end

    return Q[:,1:k*b], (AB, BB), B1
end