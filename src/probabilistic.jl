# computes log marginal likelihood of θ given (x, y) based on the Laplace approximation
# NOTE: the input θ should be an (approximate) local minimum with respect to θ
# mean_θ, std_θ are the mean and standard deviation of the prior Gaussian distribution of θ
function approximate_negative_log_evidence(node::Node, θ::AbstractVector, x::AbstractVector,
								 y::AbstractVector, std_noise::RealOrVec,
								 mean_θ::RealOrVec, std_θ::RealOrVec, objective::String, λ::Real = 1e-6,
								 verbose::Bool = false)
	mean_log_θ = log.(mean_θ)
	f = if objective == "LS"
			function (log_θ)
				ls_objective(node.phase_model, log_θ, x, y, std_noise, mean_log_θ, std_θ)
			end
		elseif objective == "KL"
			function (log_θ)
				kl_objective(node.phase_model, log_θ, x, y, mean_log_θ, std_θ, λ)
			end
		end

	θ[1:get_param_nums(node.phase_model.CPs)]= log.(θ[1:get_param_nums(node.phase_model.CPs)]) # tramsform to log space for better conditioning
	log_θ = θ
	# newton!(f, log_θ)
	return approximate_negative_log_evidence(f, log_θ, verbose)
end

# NOTE assumes θ is stationary point of θ (i.e. ∇f = 0)
function approximate_negative_log_evidence(f, θ, verbose::Bool = false)
	# calculate marginal likelihood
	d = length(θ)
	val = f(θ)
	H = ForwardDiff.hessian(f, θ)
	# display(Matrix(H))
	Σ = inverse(H) # reinterpret Hessian of minimization problem as inverse of covariance matrix
	# if verbose
	# println("in approximate_negative_log_evidence")
	# display(eigvals(Matrix(Σ)))
	# display(Matrix(Σ))
	# end
	# println("val: $(val)")
	# println("rest: $(logdet(Σ) + d * log(2π))")
	try
		logdet(Σ)
	catch DomainError
		# println("negative det")
		return 1e8
	end
	# println(val) #- (logdet(Σ) - d * log(2π)))
	return val - (logdet(Σ) - d * log(2π)) #
end

approximate_evidence(x...) = exp(-approximate_negative_log_evidence(x...))

# undamped newton algorithm for fine-tuning
function newton!(f, θ::AbstractVector; min_step::Real = 1e-10, maxiter::Int = 5)
	N = OptimizationAlgorithms.SaddleFreeNewton(f, θ)
	d = OptimizationAlgorithms.direction(N, θ)
	i = 1
	while maximum(abs, d) > min_step * maximum(abs, θ) && i < maxiter
		OptimizationAlgorithms.update!(N, θ)
		d = N.d
		i += 1
	end
	return θ
end

# TODO: move these into optimize.jl in CrystalShift, to make everything cleaner
function ls_objective(PM::PhaseModel, log_θ::AbstractVector,
					x::AbstractVector, y::AbstractVector, std_noise::RealOrVec,
					mean_log_θ::AbstractVector, std_θ::AbstractVector)
	ls_residual(PM, log_θ, x, y, std_noise) + ls_regularizer(PM, log_θ, mean_log_θ, std_θ)
end

function ls_residual(PM::PhaseModel, log_θ::AbstractVector, x::AbstractVector,
	                 y::AbstractVector, std_noise::RealOrVec)
	r = zeros(promote_type(eltype(log_θ), eltype(x), eltype(y)), length(x))
	r = _residual!(PM, log_θ, x, y, r, std_noise)
	return sum(abs2, r)
end

#################################### KL Divergence #############################
function kl_objective(phases::AbstractVector, log_θ::AbstractVector,
	                  x::AbstractVector, y::AbstractVector,
					  mean_log_θ::AbstractVector, std_θ::AbstractVector, λ::Real)
	θ = exp.(log_θ)
	r_θ = evaluate!(phases, θ, x) # reconstruction of phases, IDEA: pre-allocate result (one for Dual, one for Float)
	r_θ ./= exp(1) # since we are not normalizing the inputs, this rescaling has the effect that kl(α*y, y) has the optimum at α = 1
	kl(r_θ, y) + λ * ls_regularizer(log_θ, mean_log_θ, std_θ)
end

function ls_regularizer(PM::PhaseModel, log_θ::AbstractVector, mean_log_θ::AbstractVector, std_θ::AbstractVector)
	p = zero(eltype(log_θ))
	bg_param_num = get_param_nums(PM.background)
	θ_cp = log_θ[1:end - bg_param_num ]
	θ_bg = log_θ[end - bg_param_num + 1 : end]
	@inbounds @simd for i in eachindex(θ_cp)
		p += ((log_θ[i] - mean_log_θ[i]) / (sqrt(2)*std_θ[i]))^2
	end
	p += _prior(PM.background, θ_bg)
	return p
end