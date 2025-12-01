#=
Author: Cooper Simpson
=#

#########################################################

"""
Adapted from Krylov.jl (src/krylov_processes.jl)
"""
function lanczos(A::M, b::S, k::Int; allow_breakdown::Bool=false, reorthogonalization::Bool=false) where {R<:AbstractFloat, S<:AbstractVector{R}, M<:AbstractMatrix{R}}
	m, n = size(A)

	β₁ = zero(R)
	Q = Matrix{R}(undef, n, k+1)

	d = zeros(R, k)
	dl = zeros(R, k)

	for i = 1:k
		qᵢ = view(Q,:,i)
		qᵢ₊₁ = q = view(Q,:,i+1)

		if i == 1
			β₁ = norm(b)
			if β₁ == 0
				!allow_breakdown && error("Exact breakdown β₁ == 0.")
				fill!(qᵢ, zero(R))
			else
				@. qᵢ = b/β₁
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

		d[i] = αᵢ #Tᵢ.ᵢ = αᵢ
		βᵢ₊₁ = norm(q)

		if βᵢ₊₁ == 0
			!allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
			fill!(qᵢ₊₁, zero(R))
		else
			@. qᵢ₊₁ = q/βᵢ₊₁
		end

		dl[i] = βᵢ₊₁ #Tᵢ₊₁.ᵢ = βᵢ₊₁
	end

	return Q, SymTridiagonal(d, dl[1:end-1]), dl[end]
end