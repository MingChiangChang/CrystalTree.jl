#const RealOrVec = Union{Real, AbstractVector{<:Real}}
# computes log marginal likelihood of θ given (x, y) based on the Laplace approximation
# NOTE: the input θ should be the local minimum with respect to θ
# mean_θ, std_θ are the mean and standard deviation of the prior Gaussian distribution of θ
function log_marginal_likelihood(node::Node, θ::AbstractVector, x::AbstractVector,
								 y::AbstractVector, std_noise::RealOrVec,
								 mean_θ::RealOrVec, std_θ::RealOrVec)
	H = hessian_of_objective(node, θ, x, y, std_noise, mean_θ, std_θ)
	# calculate marginal likelihood
	Σ⁻¹ = H # reinterpret Hessian of minimization problem as inverse of covariance matrix
	negative_log_marginal_likelihood(Σ⁻¹, θ) # TODO: do we need to scale by "height" of distribution?
end

using ForwardDiff
using CrystalShift: res!, remove_act_from_θ
# computes Hessian of objective function - including regularzation - w.r.t. θ
# useful for Laplace approximation
# NOTE: if we want to use a separate std_noise for each q value, need to
# modify the residual function
function hessian_of_objective(node::Node, θ::AbstractVector, x::AbstractVector,
							  y::AbstractVector, std_noise::RealOrVec,
							  mean_θ::RealOrVec, std_θ::RealOrVec)
	function f(θ)
		sos_objective(node, θ, x, y, std_noise) + regularizer(node, θ, mean_θ, std_θ)
	end
	ForwardDiff.hessian(f, θ)
end

# sos = sum of squares
function sos_objective(node::Node, θ::AbstractVector, x::AbstractVector,
				  	   y::AbstractVector, std_noise::RealOrVec)
	residual = copy(y)
	cps = node.current_phases
	res!(cps, θ, x, residual)
	residual ./= sqrt(2) * std_noise
	return sum(abs2, residual)
end

function regularizer(node::Node, θ::AbstractVector, mean_θ::RealOrVec, std_θ::RealOrVec)
    θ_c = remove_act_from_θ(θ, node.current_phases)
	par = @. (θ_c - mean_θ) / (sqrt(2) * std_θ)
	sum(abs2, par)
end

# regularizer in log space
function log_regularizer(node::Node, θ::AbstractVector, mean_θ::RealOrVec, std_θ::RealOrVec)
    θ_c = remove_act_from_θ(θ, node.current_phases)
	par = @. (log(θ_c) - log(mean_θ)) / (sqrt(2) * std_θ)
	sum(abs2, par)
end

function negative_log_marginal_likelihood(Σ⁻¹, y)
	d = length(y)
	try
		logdet(Σ⁻¹)
		log(dot(y, Σ⁻¹, y))
	catch DomainError
		println("Σ⁻¹ is not spd..")
		return 10000
	end
	return logdet(Σ⁻¹) + d/2 * log(dot(y, Σ⁻¹, y)) # + constant
end

marginal_likelihood(x...) = exp(-negative_log_marginal_likelihood(x...))
