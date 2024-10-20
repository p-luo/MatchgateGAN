# Constructs a quantum circuit g with parameters θ, then differentiates the recursive algorithm given in Section 5.1 of https://arxiv.org/abs/1112.2184 to obtain the gradient of p_θ(x) wrt θ, where x is a measurement of g|0>. The differentiation takes polynomial time due to memoization.
# We then compare our results to the finite difference gradient
using Yao, FLOYao
using LinearAlgebra

function create_circuit(nq::Int)
    layers = 2 #Number of brick-wall layers in the circuit
    g = chain(nq)
    for _ in 1:layers
        for i in 1:nq-1
            push!(g, rot(kron(nq, i => X, i+1 => X), 0.)) #Nearest-neighbor XX rotation gates
        end
        for i in 1:nq-1
            push!(g, rot(kron(nq, i => X, i+1 => Y), 0.)) #Nearest-neighbor XY rotation gates
        end
        for i in 1:nq
            push!(g, put(nq, i => Rz(0.))) #Single qubit Z rotation gates
        end
    end
    return g
end

# #Set g to have random parameters
# p = rand(nparameters(g)).*2π
# dispatch!(g, p)
# nparams = nparameters(g)
# dim = 2*nq
# println("number of parameters: ", nparams)

⊗ = kron

function covariance_matrix(reg::MajoranaReg)
    nq = nqubits(reg)
    G = I(nq) ⊗ [0 1; -1 0]
    return reg.state * G * reg.state'
end

function majoranaindices2kron(nq, i, j) #Returns (im/2)γ_iγ_j, assuming that i≠j
    p = []
    c = (i % 2 == j % 2) ? 1 : -1
    a = min(i, j)
    b = max(i, j)
    first = (a+1) ÷ 2 
    last = (b+1) ÷ 2 
    if first == last #This means i=j-1 and j is even
        c = 1
        push!(p, first => Z)
    else
        if a % 2 == 0
            push!(p, first => X)
            c *= 1
        else
            push!(p, first => Y)
            c *= -1
        end
        for k in first+1:last-1
            push!(p, k => Z)
            c *= -1
        end
        if b % 2 == 0
            push!(p, last => Y)
        else
            push!(p, last => X)
        end
    end
    if i > j
        c *= -1
    end
    return c*kron(nq, p...)
end

function majorana_commutator(nq, i, j) #Returns [γ_i,γ_j]=2γ_iγ_j, due to the anti-commutation of Majorana operators. It needs to be an 'Add' object so that the Yao.expect' function can take it in as input.
    return Add(majoranaindices2kron(nq, i, j)) 
end

function update_opt!(reg::MajoranaReg, theta, b, temp_m, temp_grad_m, probabilities, grad_probabilities) #Evolves all matrices and probabilities and gradients by nq steps, in-place and optimally
    t_tot = 0
    dim = 2*nq
    for i in 1:nq
        t = time()
        if i > 1
            ni = b[i-1]
            cur_prob = probabilities[i-1]
            cur_grad_prob = grad_probabilities[:, i-1]
            cur_prefactor = (-1)^ni / (2*cur_prob)
            cur_grad_prefactor = (-1)^ni / (2*cur_prob^2)
            #@show size(temp_grad_m) size(cur_grad_prefactor) size(cur_grad_prob) size(temp_m) size(cur_prob) size(cur_prefactor)
            @inbounds for p in 2*(i-1)+1:dim
                for q in p+1:dim
                    for s in size(temp_grad_m, 1)
                        temp_grad_m[s,p,q] -= cur_grad_prefactor * ((-cur_grad_prob[s] * temp_m[2*(i-1)-1,p] * temp_m[2*(i-1),q]) + (cur_prob * (temp_grad_m[s, 2*(i-1)-1,p] * temp_m[2*(i-1),q] + temp_m[2*(i-1)-1,p] * temp_grad_m[s,2*(i-1),q])))
                        temp_grad_m[s,p,q] += cur_grad_prefactor * ((-cur_grad_prob[s] * temp_m[2*(i-1)-1,q] * temp_m[2*(i-1),p]) + (cur_prob * (temp_grad_m[s, 2*(i-1)-1,q] * temp_m[2*(i-1),p] + temp_m[2*(i-1)-1,q] * temp_grad_m[s,2*(i-1),p])))
                    end
                end
            end
            for p in 2*(i-1)+1:dim
                for q in p+1:dim
                    temp_m[p,q] -= cur_prefactor * (temp_m[2*(i-1)-1,p] * temp_m[2*(i-1),q])
                    temp_m[p,q] += cur_prefactor * (temp_m[2*(i-1)-1,q] * temp_m[2*(i-1),p])
                end
            end
            ni = b[i]
            probabilities[i] = (1+(-1)^ni * temp_m[2*i-1, 2*i]) / 2
            grad_probabilities[:, i] = (-1)^ni * temp_grad_m[:,2*i-1, 2*i] / 2
        else
            dispatch!(g, theta)
            temp_m = covariance_matrix(apply(reg, g))
            ni = b[i]
            probabilities[i] = (1+(-1)^ni * temp_m[2*i-1, 2*i]) / 2
            for p in 1:dim
                for q in p+1:dim
                    ham = majorana_commutator(nq, p, q) #tr([γ_i,γ_j])
                    #profiler.jl
                    temp_grad_m[:,p,q] = expect'(ham, reg => g)[2]
                end
            end
            grad_probabilities[:, i] = (-1)^ni * temp_grad_m[:,2*i-1, 2*i] / 2
        end
        diff = (time() - t) * 10^6
        t_tot += diff
        println("iteration $i: $diff")
    end
    println("total time: $t_tot")
end

function log_grad_opt(reg::MajoranaReg, theta, b, temp_m, temp_grad_m, probabilities, grad_probabilities) #Returns ∇_θlog(p_θ(b)), evaluated at 'theta' (parameters of circuit) and 'b' (measurement result); 'reg' is the initial register and must be of type MajoranaReg (e.g. FLOYao.zero_state(nq)). This uses the optimal updating function which is more efficient but still outputs the same thing as the original update! function.
    update_opt!(reg, theta, b, temp_m, temp_grad_m, probabilities, grad_probabilities)
    s = zeros(length(theta))
    for i in 1:nq
        s += grad_probabilities[:, i] / probabilities[i]
    end
    asdf = probabilities
    return asdf, s
end

# reg = apply(FLOYao.zero_state(nq), g)
# bitstr = measure(reg, nshots = 1)[1] #Random measurement of g|0>
# println("measured outcome: $bitstr")
# println("probability of measuring the above outcome: ", FLOYao.bitstring_probability(reg, bitstr)) #Uses FLOYao.bitstring_probability(reg, bitstr) which is known to be correct. We check this number against our algorithm output, to verify correctness.

# T = Float64 #Can also be BigFloat, may experiment with other data types later
# println("data type used in calculations: $T") 
# println("note: the time (μs) taken for 'iteration i' refers to the time required for the algorithm to compute p_θ(x_i|x_1,...x_{i-1}) and ∇_θ(p_θ(x_i|x_1,...x_{i-1}))")

# #Initializing temporary matrices and vectors.
# temp_m = Matrix{T}(undef, dim, dim)
# temp_grad_m = Array{T}(undef, nparams, dim, dim)
# probabilities = Vector{T}(undef, nq)
# grad_probabilities = Matrix{T}(undef, nparams, nq)

# optimized_prob, optimized = log_grad_opt(FLOYao.zero_state(nq), p, bitstr, temp_m, temp_grad_m, probabilities, grad_probabilities)
# println("The ith entry in the following vector is p_θ(x_i|x_1,...x_{i-1})")
# println(optimized_prob)
# println("the product of all entries in the above vector, should match the earlier probability computed using FLOYao.bitstring_probability: ", prod(optimized_prob))
# println("The following vector is ∇_θ(log(p_θ(x))), evaluated at x = measured outcome")
# optimized

using Yao.BitBasis
using Flux

function postprocess(g_output::Vector) #turns output of measure  into an Int vector
    result = []
    for i in 1:nq
        push!(result, g_output[1][end - i + 1])
    end
    Int.(result)
end
function d_postprocess(measurement::Vector, nbatch = batchsize)
    aa = breflect.(measurement)
    ret = Matrix(undef, nq, nbatch)
    for i in 1:nbatch
        ret[:,i] = [aa[i]...]
    end
    return ret
end

function g_loss(reg, g, theta, nbatch)
    nq = nqubits(g)
    dispatch!(g, theta)
    measurements = measure(reg, nshots = nbatch)
    discriminator_output = log.(d(d_postprocess(measurements, nbatch)))
    probs = Vector{Float64}(undef, nbatch)
    for i in 1:nbatch
        probs[i] = FLOYao.bitstring_probability(reg, measurements[i])    
    end
    return -discriminator_output * probs
end

function reinforce_grad_loss(reg, theta, nbatch)
    dispatch!(g, theta)
    T = Float64
    sampled = Dict{BitStr{nq, BigInt}, Vector{T}}()
    measurements = measure(reg, nshots = nbatch)
    discriminator_output = log.(d(d_postprocess(measurements, nbatch)))
    print(length(discriminator_output))
    #Initializing temporary matrices and vectors for the optimized version of the algorithm. Note: Do NOT need to reset these temporary matrices at the end of each iteration of the for loop.
    dim = 2*nq
    nparams = nparameters(g)
    temp_m = Matrix{T}(undef, dim, dim)
    temp_grad_m = Array{T}(undef, nparams, dim, dim)
    probabilities = Vector{T}(undef, nq)
    grad_probabilities = Matrix{T}(undef, nparams, nq)
    grad_p = Matrix{T}(undef, nparams, nbatch)

    for i in 1:nbatch
        cur_bitstr = measurements[i]
        if haskey(sampled, cur_bitstr)
            # println("SAMPLED AGAIN")
            grad_p[:,i] = sampled[cur_bitstr]
        else
            _, log_grad = log_grad_opt(FLOYao.zero_state(nq), theta, cur_bitstr, temp_m, temp_grad_m, probabilities, grad_probabilities)
            grad_p[:,i] = log_grad
            sampled[cur_bitstr] = log_grad
        end
    end
    return vec(mean(discriminator_output.*grad_p, dims = 2))
end
mean(x; dims) = sum(x; dims)/length(x)

nq = 20 #Number of qubits
d = Chain(Dense(nq, 10, relu), Dense(10, 1, sigmoid))
nparams = sum(length, Flux.params(d))
println("Number of parameters in critic: $nparams")
g = create_circuit(nq)
p = rand(nparameters(g)).*2π
reg = FLOYao.zero_state(nq)
reinforce_grad_loss(reg, p, 10)