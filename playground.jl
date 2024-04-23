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
nlp = ADNLPModel(x -> 2(1-x[1])^2 + 10(x[1]-x[2]^2)^2, Float16.([-1., -1.345]), backend=:generic)
nlp = BOX3(;n=100, type = Val(Float16), backend=:generic)
h = NormL1(1.0)
options = ROSolverOptions(verbose=1, maxIter = 10000)

####################

#TODO 18-04 : MP avec jolies structures. 
params = iR2RegParams([Float64], verb=false, activate_mp=false)

jso_res = RegularizedOptimization.R2(nlp, h, options)
my_res = MPR2(nlp, h, options, params) # launches vanilla R2-Reg (one might add verbose=1 for more verbosity)
my_res = MPR2(nlp, h, options) # iR2-Reg with additional verbosity
my_res = MPR2(nlp, h, options, activate_mp=true, Π = [Float16, Float32]) # for choosing fp formats

####################
# Benchmark : 
using SolverBenchmark

problems = (eval(Meta.parse(problem))() for problem ∈ OptimizationProblems.meta[!, :name])

h = NormL1(1.0)
options = ROSolverOptions(verbose=0)

solvers = Dict(
  :iR2_Reg => model -> MPR2(model, h, options, params),
  :R2_Reg => model -> RegularizedOptimization.R2(model, h, options),
)

stats_sans_mp = bmark_solvers(
  solvers, problems,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 4),
)

params2 = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true)
solvers2 = Dict(
  :iR2_Reg => model -> MPR2(model, h, options, params2),
  :R2_Reg => model -> RegularizedOptimization.R2(model, h, options),
)

stats_avec_mp = bmark_solvers(
  solvers, problems,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 4),
)

p = performance_profile(stats, df->Float64.(df.neval_grad))

using DataFrames
df = DataFrame(R2_Reg_neval_grad = stats[:R2_Reg].neval_grad, iR2_Reg_neval_grad = stats[:iR2_Reg].neval_grad)
df.Diff = df.R2_Reg_neval_grad - df.iR2_Reg_neval_grad
df