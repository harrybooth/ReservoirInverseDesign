# ------------------------------------------------------------
# 1. KdV MODEL
# ------------------------------------------------------------

struct KdVParams
    λ::Float64
    dx::Float64
    N::Int
end

function Dx(u, dx)
    N = length(u)
    du = similar(u)
    @inbounds for i in 1:N
        du[i] = (
            -u[mod1(i+2,N)] + 8u[mod1(i+1,N)]
            - 8u[mod1(i-1,N)] + u[mod1(i-2,N)]
        ) / (12dx)
    end
    return du
end

function Dxxx(u, dx)
    N = length(u)
    d3u = similar(u)
    @inbounds for i in 1:N
        d3u[i] = (
            -u[mod1(i+3, N)] + 8u[mod1(i+2, N)] - 13u[mod1(i+1, N)] +
             13u[mod1(i-1, N)] - 8u[mod1(i-2, N)] + u[mod1(i-3, N)]
        ) / (8 * dx^3)
    end
    return d3u
end

function kdv_fd!(du, u, p, t)
    @inbounds for i in 1:p.N
        du[i] = (-u[i] * (
            (-u[mod1(i+2,p.N)] + 8u[mod1(i+1,p.N)] - 8u[mod1(i-1,p.N)] + u[mod1(i-2,p.N)]) / (12p.dx)
        )) - p.λ * (
            (-u[mod1(i+3,p.N)] + 8u[mod1(i+2,p.N)] - 13u[mod1(i+1,p.N)] +
              13u[mod1(i-1,p.N)] - 8u[mod1(i-2,p.N)] + u[mod1(i-3,p.N)]) / (8p.dx^3)
        )
    end
    return nothing
end

# ------------------------------------------------------------
# 2. SOLITON / CNOIDAL BUILDING BLOCKS
# ------------------------------------------------------------

struct SolitonParams
    U0::Float64
    a::Float64
    k::Float64
    v::Float64
end

function SolitonParams(r::KdVParams, k, U0)
    a = 12 * r.λ * k^2
    v = U0 + a / 3
    return SolitonParams(U0, a, k, v)
end

function soliton(x, t, p::SolitonParams)
    return p.U0 .+ p.a .* (sech.(p.k .* (x .- p.v * t))).^2
end

struct CnoidalParams
    a::Float64
    k::Float64
    delta::Float64
end

function cnoidal(x, p::CnoidalParams)
    return p.a .* cos.(p.k .* x).^2 .* exp.(-((2 .* x ./ p.delta).^8))
end

# ------------------------------------------------------------
# 3. XOR / XNOR INPUT ENCODING
#
# theta = [ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta]
# ------------------------------------------------------------

const X_BITS = [[0,0], [0,1], [1,0], [1,1]]
const y_XOR  = Float64[0.0, 1.0, 1.0, 0.0]
const y_XNOR = Float64[1.0, 0.0, 0.0, 1.0]

"""
    encode_initial_condition_7(xbits, xgrid, θ, p_soliton)

θ = [ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta]
"""

function encode_initial_condition_7(xbits, xgrid, θ, p_soliton::SolitonParams)
    @assert length(θ) == 7
    ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta = θ

    b1, b2 = xbits
    e1 = b1 == 0 ? ϵ10 : ϵ11
    e2 = b2 == 0 ? ϵ20 : ϵ21

    return soliton(xgrid, 0.0, p_soliton) .+
           cnoidal(xgrid, CnoidalParams(e1, k1, delta)) .+
           cnoidal(xgrid, CnoidalParams(e2, k2, delta))
end

# ------------------------------------------------------------
# 4. LATIN HYPERCUBE SAMPLING
# ------------------------------------------------------------

"""
    latin_hypercube(n, lb, ub; rng=Random.default_rng())

Generate `n` Latin hypercube samples in [lb, ub].
Returns an n × d matrix.
"""
function latin_hypercube(n::Integer, lb::AbstractVector, ub::AbstractVector; rng=Random.default_rng())
    d = length(lb)
    @assert length(ub) == d
    @assert all(ub .> lb)

    X = Matrix{Float64}(undef, n, d)
    for j in 1:d
        pts = ((0:n-1) .+ rand(rng, n)) ./ n
        X[:, j] = lb[j] .+ shuffle(rng, pts) .* (ub[j] - lb[j])
    end
    return X
end

# ------------------------------------------------------------
# 5. GROUND-TRUTH PDE SOLVER
#
# surrogate target: u0(x) -> u(x, t_end)
# ------------------------------------------------------------

"""
    simulate_final_field_from_u0(u0, prob, t_end; solver=AutoVern7(RadauIIA5()), kwargs...)

Solve from initial condition u0 and return u(x, t_end).
"""
function simulate_final_field_from_u0(
    u0,
    prob,
    t_end;
    solver = AutoVern7(RadauIIA5()),
    kwargs...
)
    prob_s = remake(prob; u0=u0, tspan=(first(prob.tspan), t_end))
    sol = solve(prob_s, solver; saveat=[t_end], kwargs...)
    return Array(sol.u[end]),sol.retcode
end

"""
    simulate_final_field(bits, xgrid, θ, prob, t_end, p_soliton; kwargs...)
"""
function simulate_final_field(
    bits,
    xgrid,
    θ,
    prob,
    t_end,
    p_soliton;
    kwargs...
)
    u0 = encode_initial_condition_7(bits, xgrid, θ, p_soliton)
    return simulate_final_field_from_u0(u0, prob, t_end; kwargs...)
end

# ------------------------------------------------------------
# 6. DATASET CONSTRUCTION FOR NO TRAINING
#
# input  X : (Nx, 1, batch) = u0(x)
# target Y : (Nx, 1, batch) = u(x, t_end)
# ------------------------------------------------------------

"""
    build_no_dataset(param_samples, xgrid, prob, t_end, p_soliton;
                     bit_patterns=X_BITS, kwargs...)

param_samples must be n × 7 with columns:
[ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta]
"""
function build_no_dataset(
    param_samples::AbstractMatrix,
    xgrid,
    prob,
    t_end,
    p_soliton;
    bit_patterns = X_BITS,
    kwargs...
)
    @assert size(param_samples, 2) == 7

    Nx = length(xgrid)
    batch = size(param_samples, 1) * length(bit_patterns)

    X = Array{Float32}(undef, Nx, 1, batch)
    Y = Array{Float32}(undef, Nx, 1, batch)
    meta = Vector{NamedTuple}(undef, batch)

    b = 1
    for i in 1:size(param_samples, 1)
        θ = vec(param_samples[i, :])
        for bits in bit_patterns
            u0 = encode_initial_condition_7(bits, xgrid, θ, p_soliton)
            uf,retcode = simulate_final_field_from_u0(u0, prob, t_end; kwargs...)

            X[:, 1, b] .= Float32.(u0)
            Y[:, 1, b] .= Float32.(uf)
            meta[b] = (sample_id=i, bits=bits, θ=copy(θ),retcode = retcode == SciMLBase.ReturnCode.Success)
            b += 1
        end
    end

    return X, Y, meta
end

function build_no_dataset_parallel(
    param_samples::AbstractMatrix,
    xgrid,
    prob,
    t_end,
    p_soliton;
    bit_patterns = X_BITS,
    kwargs...
)
    @assert size(param_samples, 2) == 7

    Nx = length(xgrid)
    batch = size(param_samples, 1) * length(bit_patterns)

    X = Array{Float32}(undef, Nx, 1, batch)
    Y = Array{Float32}(undef, Nx, 1, batch)
    meta = Vector{NamedTuple}(undef, batch)

    b = 1

    for bits in bit_patterns

        X_vec = pmap(θ -> encode_initial_condition_7(bits, xgrid, θ, p_soliton), eachrow(param_samples))
        Y_vec = pmap(u0->simulate_final_field_from_u0(u0, prob, t_end; kwargs...),X_vec)

        for (i,θ) in enumerate(eachrow(param_samples))

            X[:, 1, b] .= Float32.(X_vec[i])
            Y[:, 1, b] .= Float32.(Y_vec[i][1])
            meta[b] = (sample_id=i, bits=bits, θ=copy(θ),retcode = Y_vec[i][2] == SciMLBase.ReturnCode.Success)
            b += 1
        end
    end

    return X, Y, meta
end


# ------------------------------------------------------------
# 7. TRAIN FNO
#
# Uses current stable NeuralOperators / Lux pattern:
#   fno = FourierNeuralOperator((modes,), in_ch, out_ch, width; ...)
#   ps, st = Lux.setup(rng, fno)
# ------------------------------------------------------------

"""
    train_fno(X, Y; width=32, modes=16, epochs=500, lr=3f-3, rng=Random.default_rng())

Train a 1D FNO on:
    X :: (Nx, 1, batch)
    Y :: (Nx, 1, batch)

Returns:
    fno, ps, st, losses
"""
function train_fno(
    X::Array{Float32,3},
    Y::Array{Float32,3};
    width::Int = 32,
    modes::Int = 16,
    epochs::Int = 500,
    lr::Float32 = 3f-3,
    rng = Random.default_rng(),
    xdev = reactant_device(; force=true),
)
    @assert size(X, 1) == size(Y, 1)
    @assert size(X, 2) == 1
    @assert size(Y, 2) == 1
    @assert size(X, 3) == size(Y, 3)

    fno = FourierNeuralOperator(
        gelu;
        chs = (1, width, width, 2width, 1),
        modes = (modes,),
    )

    ps, st = Lux.setup(rng, fno) |> xdev
    x_data = X |> xdev
    y_data = Y |> xdev
    data = [(x_data, y_data)]

    losses = Float32[]
    tstate = Training.TrainState(fno, ps, st, Optimisers.Adam(lr))

    for epoch in 1:epochs
        for (x, y) in data
            (_, loss, _, tstate) = Training.single_train_step!(
                AutoEnzyme(),
                MSELoss(),
                (x, y),
                tstate;
                return_gradients = Val(false),
            )
            push!(losses, Float32(loss))
        end

        if epoch % 50 == 0
            @info "epoch = $epoch, loss = $(losses[end])"
        end
    end

    return fno, tstate.parameters, tstate.states, losses
end

"""
    predict_final_field(fno, ps, st, u0)

Predict u(x, t_end) from u0(x).
Returns a Vector{Float32}.
"""
function predict_final_field(fno, ps, st, u0::AbstractVector)
    xin = reshape(u0, :, 1, 1)
    yhat, _ = fno(xin, ps, st)
    return vec(yhat[:, 1, 1])
end

# ------------------------------------------------------------
# 8. SMOOTH DIFFERENTIABLE SPATIAL READOUT
#
# Gaussian kernel interpolation:
#   readout(u, xr) = sum_i w_i(xr) u_i / sum_i w_i(xr)
#
# This is smooth in xr and AD-friendly.
# ------------------------------------------------------------

"""
    smooth_readout(u, xgrid, x_readout; sigma=nothing)

Differentiable spatial readout using Gaussian kernel interpolation.
"""
function smooth_readout(u::AbstractVector, xgrid::AbstractVector, x_readout; sigma=nothing)
    @assert length(u) == length(xgrid)
    σ = isnothing(sigma) ? 1.5 * (xgrid[2] - xgrid[1]) : sigma

    w = exp.(-0.5 .* ((xgrid .- x_readout) ./ σ).^2)
    return sum(w .* u) / (sum(w) + eps(eltype(w)))
end

"""
    multiple_smooth_readouts(u, xgrid, x_readouts; sigma=nothing)
"""
function multiple_smooth_readouts(u::AbstractVector, xgrid::AbstractVector, x_readouts::AbstractVector; sigma=nothing)
    return [smooth_readout(u, xgrid, xr; sigma=sigma) for xr in x_readouts]
end

# ------------------------------------------------------------
# 9. BUILD SURROGATE READOUT MATRIX
#
# R[i, j] = readout of pattern i at spatial location x_readouts[j]
# ------------------------------------------------------------

"""
    surrogate_readout_matrix(fno, ps, st, θ, x_readouts, xgrid, p_soliton;
                             bit_patterns=X_BITS, sigma=nothing)

Returns matrix R of size (n_patterns, n_readouts).
"""

function surrogate_readout_matrix(
    fno,
    ps,
    st,
    θ,
    x_readouts,
    xgrid,
    p_soliton;
    bit_patterns = X_BITS,
    sigma = nothing,
)
    rows = map(bit_patterns) do bits
        u0 = encode_initial_condition_7(bits, xgrid, θ, p_soliton)
        uf = predict_final_field(fno, ps, st, u0)
        multiple_smooth_readouts(uf, xgrid, x_readouts; sigma=sigma)
    end

    return reduce(vcat, permutedims.(rows))
end

"""
    true_readout_matrix(θ, x_readouts, xgrid, prob, t_end, p_soliton;
                        bit_patterns=X_BITS, sigma=nothing, kwargs...)

Ground-truth PDE-based readout matrix.
"""
function true_readout_matrix(
    θ,
    x_readouts,
    xgrid,
    prob,
    t_end,
    p_soliton;
    bit_patterns = X_BITS,
    sigma = nothing,
    kwargs...
)
    n_patterns = length(bit_patterns)
    n_readouts = length(x_readouts)

    R = Array{Float64}(undef, n_patterns, n_readouts)

    for i in 1:n_patterns
        uf,retcode = simulate_final_field(bit_patterns[i], xgrid, θ, prob, t_end, p_soliton; kwargs...)
        R[i, :] .= multiple_smooth_readouts(uf, xgrid, x_readouts; sigma=sigma)
    end

    return R
end

# ------------------------------------------------------------
# 10. DETERMINANT / GRAM-DETERMINANT SEPARABILITY OBJECTIVE
#
# If R is square: log(|det(R)| + eps)
# Else: 0.5 * logdet(R*R' + eps*I)
# ------------------------------------------------------------

"""
    separability_score(R; ϵ=1f-6)

Returns a scalar score to maximize.
"""
function separability_score(R::AbstractMatrix; ϵ=1f-6)
    m, n = size(R)

    if m == n
        return log(abs(det(Matrix(R))) + ϵ)
    else
        G = Matrix(R * R') + ϵ * I(m)
        return 0.5f0 * logdet(Hermitian(G))
    end
end

"""
    readout_repulsion_penalty(x_readouts; δ=1f-4)

Discourages collapse of readout locations.
"""
function readout_repulsion_penalty(x_readouts::AbstractVector; δ=1f-4)
    pen = zero(eltype(x_readouts))
    for i in 1:length(x_readouts)-1
        for j in i+1:length(x_readouts)
            pen += 1 / ((x_readouts[i] - x_readouts[j])^2 + δ)
        end
    end
    return pen
end

# ------------------------------------------------------------
# 11. INVERSE DESIGN OBJECTIVE
#
# z = [θ; x_readouts]
# θ has length 7:
#   [ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta]
# x_readouts has length n_readouts
# ------------------------------------------------------------

"""
    inverse_design_objective(fno, ps, st, z, xgrid, p_soliton; λ_rep=1f-3, sigma=nothing)

Minimize:
    - separability_score(R) + λ_rep * repulsion_penalty
"""
function inverse_design_objective(
    fno,
    ps,
    st,
    z,
    xgrid,
    p_soliton;
    λ_rep = 1f-3,
    sigma = nothing,
)
    θ = z[1:7]
    x_readouts = z[8:end]

    R = surrogate_readout_matrix(
        fno, ps, st, θ, x_readouts, xgrid, p_soliton;
        sigma = sigma,
    )

    sep = separability_score(R)
    rep = readout_repulsion_penalty(x_readouts)

    return -sep + λ_rep * rep
end

"""
    optimize_theta_and_readouts(fno, ps, st, θ0, xreadout0, lbθ, ubθ, xgrid, p_soliton;
                                maxiters=300, λ_rep=1f-3, sigma=nothing)

Optimize the 7 encoding parameters and n readout locations.
"""
function optimize_theta_and_readouts(
    fno,
    ps,
    st,
    θ0,
    xreadout0,
    lbθ,
    ubθ,
    xgrid,
    p_soliton;
    maxiters = 300,
    λ_rep = 1f-3,
    nstarts = 1,
    sigma = nothing,
)
    @assert length(θ0) == 7
    @assert length(lbθ) == 7
    @assert length(ubθ) == 7

    z0 = Float32.(vcat(θ0, xreadout0))
    lb = Float32.(vcat(lbθ, fill(xgrid[1], length(xreadout0))))
    ub = Float32.(vcat(ubθ, fill(xgrid[end], length(xreadout0))))

    lossfun = (u, p) -> inverse_design_objective(
        fno, ps, st, u, xgrid, p_soliton;
        λ_rep = λ_rep,
        sigma = sigma,
    )

    optf = OptimizationFunction(lossfun, AutoZygote())
    prob = OptimizationProblem(optf, z0; lb=lb, ub=ub)

    return solve(prob,TikTak(nstarts), LBFGS(); maxiters=maxiters)
end

"""
    optimize_theta_and_readouts_BBO(
        prob,
        t_end,
        θ0,
        xreadout0,
        lbθ,
        ubθ,
        xgrid,
        p_soliton;
        maxiters=300,
        maxevals=nothing,
        λ_rep=1e-3,
        sigma=nothing,
        alg=BBO_adaptive_de_rand_1_bin_radiuslimited(),
        callback=nothing,
    )

Black-box optimization of:
    z = [θ; x_readouts]

where
    θ = [ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta]

Uses the true PDE objective.
"""
function optimize_theta_and_readouts_BBO(
    prob,
    t_end,
    θ0,
    xreadout0,
    lbθ,
    ubθ,
    xgrid,
    p_soliton;
    MaxFuncEvals = nothing, 
    maxiters = nothing,
    λ_rep = 1e-3,
    sigma = nothing,
    alg = BBO_adaptive_de_rand_1_bin_radiuslimited(),
    meta_alg = false,
    callback = nothing,
    sort_readouts = true,
    nthreads = 1)

    @assert length(θ0) == 7
    @assert length(lbθ) == 7
    @assert length(ubθ) == 7

    # Use Float64 for PDE-based objective
    z0 = Float64.(vcat(θ0, xreadout0))
    lb = Float64.(vcat(lbθ, fill(xgrid[1], length(xreadout0))))
    ub = Float64.(vcat(ubθ, fill(xgrid[end], length(xreadout0))))

    lossfun = function (z, p)
        θ = z[1:7]
        x_readouts = z[8:end]

        # Optional symmetry breaking (recommended for BBO)
        if sort_readouts
            x_readouts = sort(x_readouts)
        end

        R = true_readout_matrix(
            θ,
            x_readouts,
            xgrid,
            prob,
            t_end,
            p_soliton;
            sigma = sigma,
        )

        sep = separability_score(R)
        rep = readout_repulsion_penalty(x_readouts)

        return -sep + λ_rep * rep
    end

    optf = OptimizationFunction(lossfun)
    optprob = OptimizationProblem(optf, z0; lb=lb, ub=ub)

    # --------------------------------------------------
    # Build keyword arguments for solver
    # --------------------------------------------------
    solve_kwargs = Dict{Symbol,Any}()

    if MaxFuncEvals !== nothing 
        if meta_alg
            solve_kwargs[:f_calls_limit] = MaxFuncEvals 
        else
            solve_kwargs[:MaxFuncEvals] = MaxFuncEvals 
        end
    end

    if maxiters !== nothing
        solve_kwargs[:maxiters] = maxiters
    end

    if callback !== nothing
        solve_kwargs[:callback] = callback
    end

    if nthreads !== nothing
        solve_kwargs[:NThreads] = nthreads
    end

    return solve(optprob, alg; solve_kwargs...)
end

# ------------------------------------------------------------
# 12. LINEAR READOUT FIT FOR VALIDATION
#
# Once R is found, fit y ≈ R w by least squares.
# ------------------------------------------------------------

"""
    fit_linear_readout(R, y; ridge=1e-8)

Returns:
    w, yhat, mse
"""
function fit_linear_readout(R::AbstractMatrix, y::AbstractVector; ridge=1e-8)
    nread = size(R, 2)
    A = Matrix(R' * R + ridge * I(nread))
    b = Vector(R' * y)
    w = A \ b
    yhat = R * w
    mse = mean((yhat .- y).^2)
    return w, yhat, mse
end