# computes log marginal likelihood of θ given (x, y) based on the Laplace approximation
# NOTE: the input θ should be the local minimum with respect to θ
# mean_θ, std_θ are the mean and standard deviation of the prior Gaussian distribution of θ
function log_marginal_likelihood(node::Node, θ::AbstractVector, x::AbstractVector,
								 y::AbstractVector, std_noise::RealOrVec,
								 mean_θ::RealOrVec, std_θ::RealOrVec, objective::String)
	log_θ = log.(θ)

	H = if objective == "LS"
		hessian_of_objective_wrt_log(node, log_θ, x, y, std_noise, mean_θ, std_θ)
	else objective = "KL"
		hessian_of_kl_objective(node.current_phases, log_θ, x, y, mean_θ, std_θ)
	end

	# calculate marginal likelihood
	Σ = inverse(H) # reinterpret Hessian of minimization problem as inverse of covariance matrix
	negative_log_marginal_likelihood(Σ, log_θ) # TODO: do we need to scale by "height" of distribution?
end

# computes Hessian of objective function - including regularzation - w.r.t. θ
# useful for Laplace approximation
# NOTE: if we want to use a separate std_noise for each q value, need to
# modify the residual function
function hessian_of_objective(node::Node, θ::AbstractVector, x::AbstractVector,
							  y::AbstractVector, std_noise::RealOrVec,
							  mean_θ::RealOrVec, std_θ::RealOrVec)
	function f(θ)
		sos_objective(node, θ, x, y, std_noise) + regularizer(θ, mean_θ, std_θ)
	end

	ForwardDiff.hessian(f, θ)
end

# sos = sum of squares
function sos_objective(node::Node, θ::AbstractVector, x::AbstractVector,
	y::AbstractVector, std_noise::RealOrVec)
	r = zeros(promote_type(eltype(θ), eltype(x), eltype(y)), length(x))
	r = _residual!(node.current_phases, log.(θ), x, y, r, std_noise)
	return sum(abs2, r)
end

# regularizer in log space
function regularizer(θ::AbstractVector, mean_θ::RealOrVec, std_θ::RealOrVec)
	p = zero(θ)
	sum(abs2, _prior(p, log.(θ), mean_θ, std_θ))
end

function hessian_of_objective_wrt_log(node::Node, log_θ::AbstractVector, x::AbstractVector,
									  y::AbstractVector, std_noise::RealOrVec,
									  mean_θ::RealOrVec, std_θ::RealOrVec)
	function f(log_θ)
		sos_log_objective(node, log_θ, x, y, std_noise) + log_regularizer(log_θ, mean_θ, std_θ)
	end
    # println("sos: $()")
	H = ForwardDiff.hessian(f, log_θ)
	# display(eigvals(H))
	H
end

function sos_log_objective(node::Node, log_θ::AbstractVector, x::AbstractVector,
	                       y::AbstractVector, std_noise::RealOrVec)
	r = zeros(promote_type(eltype(log_θ), eltype(x), eltype(y)), length(x))
	r = _residual!(node.current_phases, log_θ, x, y, r, std_noise)
	return sum(abs2, r)
end

function log_regularizer(log_θ::AbstractVector, mean_θ::RealOrVec, std_θ::RealOrVec)
	p = zero(log_θ)
	sum(abs2, _prior(p, log_θ, mean_θ, std_θ))
end

function hessian_of_kl_objective(phases::AbstractVector, log_θ::AbstractVector,
	                             x::AbstractVector, y::AbstractVector,
								 mean_θ::AbstractVector, std_θ::AbstractVector, λ::Real = 0)
    μ = log.(mean_θ)
	function f(log_θ)
		newton_objective(phases, log_θ, x, y, μ, std_θ, λ)
	end
	H = ForwardDiff.hessian(f, log_θ)
	# display(eigvals(H))
	H
end

function newton_objective(phases::AbstractVector, log_θ::AbstractVector,
	                      x::AbstractVector, y::AbstractVector,
						  μ::AbstractVector, std_θ::AbstractVector, λ::Real)
	θ = exp.(log_θ)
	r_θ = reconstruct!(phases, θ, x) # reconstruction of phases, IDEA: pre-allocate result (one for Dual, one for Float)
	r_θ ./= exp(1) # since we are not normalizing the inputs, this rescaling has the effect that kl(α*y, y) has the optimum at α = 1
	kl(r_θ, y) + λ * prior(log_θ, μ, std_θ)
end

function prior(log_θ::AbstractVector, μ::AbstractVector, std_θ::AbstractVector)
	p = zero(eltype(log_θ))
	@inbounds @simd for i in eachindex(log_θ)
		p += (log_θ[i] - μ[i]) / (sqrt(2)*std_θ[i])
	end
	return p
end

function negative_log_marginal_likelihood(Σ, y)
	d = length(y)
	return 1/2 * (dot(y, inverse(Σ), y) + logdet(Σ) + d * log(2π))
end

marginal_likelihood(x...) = exp(-negative_log_marginal_likelihood(x...))
