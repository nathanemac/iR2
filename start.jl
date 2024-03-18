using Pkg
Pkg.activate(".")
using ShiftedProximalOperators
using RegularizedOptimization
using ProximalOperators

x0 = [1.0, 2.0, 3.0]
s = [0.0, 0.0, 0.0]
σ0 = 1e-1

# Norme l0
h0 = NormL0(1.0)
ψ0 = ShiftedNormL0(h0, x0, s, false) # q -> l0(x0 + s + q)
q = [-2.0, 0.3, 0.05]
y = similar(x0)
prox!(y, ψ0, q, σ0)


# Norme l1
h1 = NormL1(1.0)
ψ1 = ShiftedNormL1(h1, x0, s, false) # l1(x0 + s + q)

y = similar(x0)
q = [-2.0, 0.3, 0.0]
prox!(y, ψ1, q, σ0)





################## 
#### Test R2 #####
using ADNLPModels
using NLPModels

nlp = ADNLPModel(x -> x[2]+x[1].^2 -11, [10., 20.])
options = ROSolverOptions(maxIter=1000, verbose=1)
h1 = NormL1(1.0)

res = R2(nlp, h1, options)

