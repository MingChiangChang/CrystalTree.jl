# computes log marginal likelihood of θ given (x, y) based on the Laplace approximation
# NOTE: the input θ should be an (approximate) local minimum with respect to θ
# mean_θ, std_θ are the mean and standard deviation of the prior Gaussian distribution of θ
# std_noise is the standard deviation of the noise, can be real number or vector
# if std_noise is a vector, each element corresponds the std of an element in y
function get_probabilities(results::AbstractVector{<:Node},
							x::AbstractVector{<:Real},
							y::AbstractVector{<:Real},
							std_noise::RealOrVec,
							mean_θ::AbstractVector{<:Real},
							std_θ::AbstractVector{<:Real};
							objective::AbstractObjective = LeastSquares(),
							renormalize::Bool = true)

	neg_log_prob = zeros(length(results))

	for i in 1:length(results)
		θ = get_free_params(results[i].phase_model)
		full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, results[i].phase_model.CPs)
		neg_log_prob[i] = approximate_negative_log_evidence(results[i], θ, x, y,
								std_noise, full_mean_θ, full_std_θ, objective)
	end

	if renormalize
		neg_log_prob ./= minimum(neg_log_prob) * std_noise # Renormalize
	end
	log_normalization = logsumexp(-neg_log_prob)  # numerically stable computation
	return @. exp(-(neg_log_prob + log_normalization)) # i.e. prob / sum(prob)
end

# λ controls the weight of the least-squares regularization term for KL objective
function approximate_negative_log_evidence(node::Node, θ::AbstractVector, x::AbstractVector,
								 y::AbstractVector, std_noise::RealOrVec,
								 mean_θ::RealOrVec, std_θ::RealOrVec,
								 objective::AbstractObjective, λ::Real = 1e-6,
								 verbose::Bool = false)
	mean_log_θ = log.(mean_θ)

	f = if objective isa LeastSquares
			function (log_θ)
				ls_objective(node.phase_model, log_θ, x, y, std_noise, mean_log_θ, std_θ)
			end
		elseif objective isa KullbackLeilber
			function (log_θ)
				kl_objective(node.phase_model, log_θ, x, y, mean_log_θ, std_θ, λ)
			end
		end

	# tramsform to log space for better conditioning
	θ[1:get_param_nums(node.phase_model.CPs)]= log.(θ[1:get_param_nums(node.phase_model.CPs)])
	log_θ = θ
	# IDEA: chek norm of gradient. If it exceeds a threshold, apply fine-tuning.
	# newton!(f, log_θ)
	return approximate_negative_log_evidence(f, log_θ, verbose)
end

# NOTE assumes θ is stationary point of θ (i.e. ∇f = 0)
# computes approximation to \int exp(-f(θ)) dθ via Laplace approximation
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
		return val - (logdet(Σ) + d * log(2π)) / 2
	catch DomainError
		return 1e8
	end
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
