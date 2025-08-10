#=
Author: Cooper Simpson

Adapted from Krylov.jl (src/krylov_processes.jl)
=#
function lanczos(A::M, b::S, k::Int; allow_breakdown::Bool=false, reorthogonalization::Bool=false) where {R<:AbstractFloat, S<:AbstractVector{R}, M<:AbstractMatrix{R}}
	m, n = size(A)

	β₁ = zero(R)
	V = Matrix{R}(undef, n, k+1)

	d = zeros(R, k)
	dl = zeros(R, k)

	for i = 1:k
		vᵢ = view(V,:,i)
		vᵢ₊₁ = q = view(V,:,i+1)

		if i == 1
			β₁ = norm(b)
			if β₁ == 0
				!allow_breakdown && error("Exact breakdown β₁ == 0.")
				fill!(vᵢ, zero(R))
			else
				@. vᵢ = b/β₁
			end
		end

		mul!(q, A, vᵢ)

		if i ≥ 2
			vᵢ₋₁ = view(V,:,i-1)
			βᵢ = dl[i-1] #βᵢ = Tᵢ.ᵢ₋₁
			axpy!(-βᵢ, vᵢ₋₁, q)
		end

		αᵢ = dot(vᵢ, q)
		axpy!(-αᵢ, vᵢ, q)

		if reorthogonalization
			if i ≥ 2
				vᵢ₋₁ = view(V,:,i-1)
				βtmp = dot(vᵢ₋₁, q)
				dl[i-1] += βtmp
				axpy!(-βtmp, vᵢ₋₁, q)
			end
			αtmp = dot(vᵢ, q)
			αᵢ += αtmp
			axpy!(-αtmp, vᵢ, q)
		end

		d[i] = αᵢ #Tᵢ.ᵢ = αᵢ
		βᵢ₊₁ = norm(q)

		if βᵢ₊₁ == 0
			!allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
			fill!(vᵢ₊₁, zero(R))
		else
			@. vᵢ₊₁ = q/βᵢ₊₁
		end

		dl[i] = βᵢ₊₁ #Tᵢ₊₁.ᵢ = βᵢ₊₁
	end

	return V, SymTridiagonal(d, dl[1:end-1]), dl[end]
end