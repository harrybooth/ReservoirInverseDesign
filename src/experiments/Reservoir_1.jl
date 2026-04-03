# ------------------------
# Reservoir / PDE setup
# ------------------------
λ = 0.333
N = 256
L = 40.0

x = range(-L/2, L/2, length=N)[1:end-1]
dx = step(x)

p_reservoir = KdVParams(λ, dx, N - 1)

ks = 0.5
U0 = 1.0
p_soliton = SolitonParams(p_reservoir, ks, U0)

u0_base = soliton(x, 0.0, p_soliton)
prob = ODEProblem(kdv_fd!, u0_base, (0.0, 10.0), p_reservoir)

t_end = 10.0


# ------------------------
# Design parameter bounds
#
# θ = [ϵ10, ϵ11, ϵ20, ϵ21, k1, k2, delta]
# ------------------------

lb_theta = [0.01, 0.01, 0.01, 0.01, 0.20, 0.20,  10.]
ub_theta = [0.35, 0.35, 0.35, 0.35, 0.90, 0.90, 30.0]