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
using OptimizationProblems.ADNLPProblems
using OptimizationProblems

include("iR2_Alg MP.jl")
include("utils.jl")

####################

nlp=woods(;n=100, type = Val(Float64), backend=:generic)
nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1.2, -1.345], backend=:generic)
nlp = BOX3(;n=100, backend=:generic)
h = NormL1(1.0)
options = ROSolverOptions(verbose=1, maxIter = 10000)

####################

params = iR2RegParams([Float16, Float32, Float64], verb=true, activate_mp=true)

jso_res = RegularizedOptimization.R2(nlp, h, options)
my_res = MPR2(nlp, h, options, params) # launches vanilla R2-Reg (one might add verbose=1 for more verbosity)

####################
# Benchmark : 
using SolverBenchmark
using CSV
options = ROSolverOptions(verbose=0)
h = NormL1(1.0)

## sans MP ##
problems_sans_mp = (eval(Meta.parse(problem))() for problem ∈ OptimizationProblems.meta[!, :name])
params_sans_mp = iR2RegParams([Float64], verb=false, activate_mp=false)
solvers_sans_mp = Dict(
  :R2_Reg => model -> RegularizedOptimization.R2(model, h, options),
  :iR2_Reg => model -> MPR2(model, h, options, params_sans_mp),
)

stats_sans_mp = bmark_solvers(
  solvers_sans_mp, problems_sans_mp,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 4),
)


## avec MP ##
problems_mp = (eval(Meta.parse(problem))(;  type = Val(Float16), backend = :generic) for problem ∈ OptimizationProblems.meta[!, :name])
params_mp = iR2RegParams([Float16, Float32, Float64], verb=true, activate_mp=true)

solvers_avec_mp = Dict(
  :iR2_Reg => model -> MPR2(model, h, options, params_mp)
)
stats_avec_mp = bmark_solvers(
  solvers_avec_mp, problems_mp,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 4),
)

solvers_avec_mp2 = Dict(
  :R2_Reg => model -> RegularizedOptimization.R2(model, h, options)
)
stats_avec_mp2 = bmark_solvers(
  solvers_avec_mp2, problems_sans_mp,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 4),
)
stats_mp = merge(stats_avec_mp, stats_avec_mp2)



p = performance_profile(stats, df->Float64.(df.neval_grad))

CSV.write("benchmark sans mp.csv", stats_sans_mp)
CSV.write("benchmark avec mp.csv", stats_mp)
