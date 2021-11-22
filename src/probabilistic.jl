#const RealOrVec = Union{Real, AbstractVector{<:Real}}
# computes log marginal likelihood of θ given (x, y) based on the Laplace approximation
# NOTE: the input θ should be the local minimum with respect to θ
# mean_θ, std_θ are the mean and standard deviation of the prior Gaussian distribution of θ
function log_marginal_likelihood(node::Node, θ::AbstractVector, x::AbstractVector,
								 y::AbstractVector, std_noise::RealOrVec,
								 mean_θ::RealOrVec, std_θ::RealOrVec)
	H = hessian_residual(node, θ, x, y, std_noise, mean_θ, std_θ)
	# calculate marginal likelihood
	Σ⁻¹ = H # reinterpret Hessian of minimization problem as inverse of covariance matrix
	log_marginal_likelihood(Σ⁻¹, θ) # TODO: do we need to scale by "height" of distribution?
end

using ForwardDiff
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
	residual = copy(y)
	for phase in node.current_phases
	    full_θ = get_eight_params(phase, θ)
	    residual .-= CrystalPhase(phase, θ).(x)
	end
	residual ./= std_noise
	return sum(abs2, residual)
end

function regularizer(θ::AbstractVector, mean_θ::RealOrVec, std_θ::RealOrVec)
	par = @. θ - mean_Θ / std_θ
	sum(abs2, par)
end

function negative_log_marginal_likelihood(Σ⁻¹, y)
	d = length(y)
	logdet(Σ⁻¹) + d/2 * log(dot(y, Σ⁻¹, y)) # + constant
end

marginal_likelihood(x...) = exp(-negative_log_marginal_likelihood(x...))
