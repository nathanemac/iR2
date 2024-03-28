using LinearAlgebra
using NLPModels
using ADNLPModels
using ShiftedProximalOperators
using RegularizedOptimization 
using ProximalOperators
using Printf
using SolverCore
using Plots
using JSOSolvers

include("iR2_Alg MP.jl")
include("utils.jl")

####################

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1.55, 2.345])
h = NormL1(1.0)
options = ROSolverOptions(verbose=1, maxIter=2000)

####################
jso_res = RegularizedOptimization.R2(nlp, h, options)
my_res = MPR2(nlp, h, options, verb=true)

####################

K = [k for k=1:my_res.iter]
P1 = my_res.solver_specific[:p_hist]
P2 = abs.(my_res.solver_specific[:Fhist] .- my_res.solver_specific[:Fhist][end])
E = [callback_precision(P1[i])[2] for i=1:my_res.iter]

plot(K, P1, label="Niveau de précision selon les itérations", xlabel="itération", ylabel="niveau de précision")
plot(K, E, label="Erreur maximale admise selon les itérations", xlabel="itération", ylabel="erreur", yscale=:log10)
plot!(K, P2,  ylabel="", label="Distance a la solution", yscale=:log10)


# test sur 1 probleme de OptimizationProblems
prob = woods(n=1000)
options = ROSolverOptions(verbose=2)
my = MPR2(prob, h, options)
jso = RegularizedOptimization.R2(prob, h, options)




# Benchmark : 
using ADNLPModels
using SolverBenchmark
using OptimizationProblems.ADNLPProblems
problems = (eval(Meta.parse(problem))() for problem ∈ OptimizationProblems.meta[!, :name])

h = NormL1(1.0)
options = ROSolverOptions(verbose=0)

solvers = Dict(
  :iR2_Reg => model -> MPR2(model, h, options),
  :R2_Reg => model -> RegularizedOptimization.R2(model, h, options),
)

stats = bmark_solvers(
  solvers, problems,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 10),
)


