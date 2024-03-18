using LinearAlgebra
using Distributions
using NLPModels
using ADNLPModels
using ShiftedProximalOperators
using RegularizedOptimization
using ProximalOperators
using Printf
using SolverCore
using Plots

include("MPR2_Alg.jl")

####################

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1., 2.])
h = NormL1(1.0)
options = ROSolverOptions(verbose=2)

####################
jso_res = R2(nlp, h, options)
my_res = MPR2(nlp, h, options)

####################

K = [k for k=1:my_res.iter]
P = my_res.solver_specific[:p_hist]
E = [callback_precision(P[i])[2] for i=1:my_res.iter]

plot(K, P, label="Niveau de précision selon les itérations", xlabel="itération", ylabel="niveau de précision")
plot(K, E, label="Erreur maximale admise selon les itérations", xlabel="itération", ylabel="erreur", yscale=:log10)


# test

