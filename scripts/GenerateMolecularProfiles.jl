# This code is expected to be run from an sbatch script after a module load julia command has been run.
# It starts the remote processes with srun within an allocation specified in the sbatch script.

using Pkg

Pkg.activate("..")
Pkg.instantiate()
# Pkg.precompile()

using DrWatson
using Distributed
using SlurmClusterManager

projectdir_static = dirname(Base.active_project())

cluster_calc = true

if !cluster_calc
    @quickactivate "ReservoirInverseDesign"
end

if cluster_calc
    n_tasks = parse(Int, ENV["SLURM_NTASKS"])
    addprocs(SlurmManager())
    @everywhere using Pkg
    @everywhere Pkg.activate("..")
end

@everywhere begin
    using DrWatson
    using JLD2
    using Printf
    using Base.Threads
    using Base.Threads: @spawn
    using Distributions
end

@everywhere projectdirx(args...) = joinpath($projectdir_static, args...)

for dir_type ∈ ("data", "src", "plots", "scripts", "papers")
    function_name = Symbol(dir_type * "dirx")
    @everywhere @eval begin
        $function_name(args...) = projectdirx($dir_type, args...)
    end
end

@everywhere include(srcdir("kdV_Reservoir_Surrogate.jl"))
@everywhere include(srcdir("experiments/Reservoir_1.jl"))

@everywhere begin 

    function lambda_gaussian(x; λ0=1.0, A=1.0, x0=0.0, σ=1.0)
        return λ0 .+ A .* exp.(-((x .- x0).^2) ./ (2σ^2))
    end

    lambda_toxic(seed) = lambda_gaussian(x; λ0=1.0, A=rand(MersenneTwister(seed),Uniform(0.,1.)), x0=0.0, σ=rand(MersenneTwister(seed),Uniform(4.,5.)))
    lambda_good(seed) = lambda_gaussian(x; λ0=1.0, A=rand(MersenneTwister(seed),Uniform(0.,1.)), x0=0.0, σ=rand(MersenneTwister(seed),Uniform(1.,2.)))

    function return_measurment(seed,trial,toxic_threshold; θ = [0.05, 0.5,30], delivery_noise = 1.)

        θn = θ .* rand(LogNormal(0,delivery_noise),3)

        u0 = encode_initial_condition_single(x, θn, p_soliton);

        λ = seed <= toxic_threshold ? lambda_good(seed) : lambda_toxic(seed)

        p_set = KdVParamsE(λ,p_reservoir.dx,p_reservoir.N)

        prob = ODEProblem(kdv_fd!, u0, (0.0, t_end),p_set)

        sol = solve(prob, AutoVern7(RadauIIA5()),saveat = [1.,t_end]);

        return sol.retcode,sol.u
    end

    n_molecules = 5000
    toxic_threshold = Int(n_molecules /2)
    n_trials = 10

    delivery_noise = 1.
    θ = [0.05, 0.5,30]

    molecule_test_schedule = [(seed,trial) for seed in 1:n_molecules for trial in 1:n_trials]
end

measurements = pmap(p->return_measurment(p[1],p[2],toxic_threshold; θ = θ, delivery_noise = 1.),molecule_test_schedule)

data_dict = Dict()

data_dict["mol_id"] = molecule_test_schedule  
data_dict["profiles"] = last.(measurements)
data_dict["numeric_status"] = first.(measurements)

save(datadirx("exp_raw","GoodToxicMol_" * string(n_molecules) * "_" * string(n_trials) * ".jld2"),data_dict)

