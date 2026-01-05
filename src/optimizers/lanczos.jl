#=
Author: Cooper Simpson
=#

#########################################################
# Scalar Lanczos
#########################################################

"""
Scalar Lanczos process.

NOTE: Adapted from Krylov.jl (src/krylov_processes.jl)
"""
function lanczos(A::M, b::S, k::Int; allow_breakdown::Bool=false, reorthogonalization::Bool=false) where {R, S<:AbstractVector{R}, M<:AbstractMatrix{R}}
	m, n = size(A)
	m == n || throw(DimensionMismatch("Lanczos requires a square operator"))

	β₁ = zero(R)
	Q = Matrix{R}(undef, n, k+1)

	d = zeros(R, k)
	dl = zeros(R, k)

	for i = 1:k
		qᵢ = view(Q,:,i)
		qᵢ₊₁ = q = view(Q,:,i+1)

		if i == 1
            β₁ = norm2(b)
			if β₁ == 0
				!allow_breakdown && error("Exact breakdown β₁ == 0.")
				fill!(qᵢ, zero(R))
			else
                copyto!(qᵢ, b)
                rmul!(qᵢ, inv(β₁))
			end
		end

		mul!(q, A, qᵢ)

		if i ≥ 2
			qᵢ₋₁ = view(Q,:,i-1)
			βᵢ = dl[i-1] #βᵢ = Tᵢ.ᵢ₋₁
			axpy!(-βᵢ, qᵢ₋₁, q)
		end

		αᵢ = dot(qᵢ, q)
		axpy!(-αᵢ, qᵢ, q)

		"""
		Selective reorthogonalization against last two vectors.
		"""
		if reorthogonalization
			if i ≥ 2
				qᵢ₋₁ = view(Q,:,i-1)
				βtmp = dot(qᵢ₋₁, q)
				dl[i-1] += βtmp
				axpy!(-βtmp, qᵢ₋₁, q)
			end

			αtmp = dot(qᵢ, q)
			αᵢ += αtmp
			axpy!(-αtmp, qᵢ, q)
		end

		d[i] = αᵢ # Tᵢ.ᵢ = αᵢ
		βᵢ₊₁ = norm2(q)

		if βᵢ₊₁ ≤ eps(R)
			!allow_breakdown && error("Breakdown βᵢ₊₁ ≤ eps at iteration i = $i.")
			fill!(qᵢ₊₁, zero(R))
		else
            copyto!(qᵢ₊₁, q)
            rmul!(qᵢ₊₁, inv(βᵢ₊₁))
		end

		dl[i] = βᵢ₊₁ # Tᵢ₊₁.ᵢ = βᵢ₊₁
	end

	return Q, SymTridiagonal(d, dl[1:end-1]), dl[end]
end

#########################################################
# Block Lanczos
#########################################################

"""
Block Lanczos process.

NOTE: Adapted from Krylov.jl (src/block_krylov_processes.jl)
"""
function block_lanczos(Z::M1, Ω::M2, k::Int; reorthogonalization::Bool=false) where {R<:AbstractFloat, M1<:AbstractMatrix{R}, M2<:AbstractMatrix{R}}
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

	if reorthogonalization
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

		mul!(QAi, Z, V_i) # q = A * V_i

        # Subtract V_{i-1}*B_i   (does the right thing for b=1)
        if i ≥ 2
            V_prev = view(Q, :, blk .- b)
            QAi .-= V_prev*B_i
        end

        # compute A_i = V_i' * QAi
		mul!(A_i, V_i', QAi)

        # q -= V_i * A_i
        QAi .-= V_i*A_i

        # selective reorthogonalization (same as scalar)
        if reorthogonalization
            if i > 1
                mul!(ABtmp, V_prev', QAi)
                B_i .+= ABtmp
                QAi .-= V_prev * ABtmp
            end
			mul!(ABtmp, V_i', QAi)
            A_i .+= ABtmp
            QAi .-= V_i * ABtmp
        end

        # save A_i in T
        T[blk, blk] .= A_i

        # orthogonalize QAi → V_next, B_ip1 = QR factor
        V, Rnew = qr(QAi)
		V_next[:,:] .= Matrix(V)
        B_ip1 .= UpperTriangular(Rnew)

        # save B_ip1 in T
        if i < k
            blk_below = blk_next
            T[blk_below, blk] .= B_ip1
            T[blk, blk_below] .= B_ip1'
        end

        # shift B_i ← B_ip1
        B_i .= B_ip1
    end

    return Q[:,1:k*b], Symmetric(T), B1
end

"""
Potentially an improved Block Lanczos function. Not tested or guaranteed to work at all.
"""
function block_lanczos_fa(Z::M1, Ω::M2, k::Int; reorthogonalization::Bool=false) where
        {R<:AbstractFloat, M1<:AbstractMatrix{R}, M2<:AbstractMatrix{R}}

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