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

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1.55, 2.345])
nlp = woods(get_nvar=1000)
h = NormL1(Float16(1.0))
h2 = NormL1(1.0)
options = ROSolverOptions(verbose=1, maxIter = 1000)


####################

jso_res = RegularizedOptimization.R2(nlp, h, options)
my_res = MPR2(nlp, h2, options) # launches vanilla R2-Reg (one might add verbose=1 for more verbosity)
my_res = MPR2(nlp, h, options, activate_mp=true, Π=[Float16, Float32, Float64]) # launches iR2-Reg 
my_res = MPR2(nlp, h, options, activate_mp=true, Π=[Float16, Float32, Float64], verb=true) # iR2-Reg with additional verbosity
my_res = MPR2(nlp, h, options, activate_mp=true, Π = [Float16, Float32]) # for choosing fp formats

####################

K = [k for k=1:my_res.iter]
P1 = my_res.solver_specific[:p_hist]
P2 = abs.(my_res.solver_specific[:Fhist] .- my_res.solver_specific[:Fhist][end])
E = [callback_precision(P1[i])[2] for i=1:my_res.iter]

plot(K, P1, label="Niveau de précision selon les itérations", xlabel="itération", ylabel="niveau de précision")
plot(K, E, label="Erreur maximale admise selon les itérations", xlabel="itération", ylabel="erreur", yscale=:log10)
plot!(K, P2,  ylabel="", label="Distance a la solution", yscale=:log10)


# test sur 1 probleme de OptimizationProblems
prob = woods(get_nvar = 100)
options = ROSolverOptions(verbose=1)
my = MPR2(prob, h, options, verb=true)
jso = RegularizedOptimization.R2(prob, h, options)

options_dict = Dict{Symbol, Any}()
for field in fieldnames(typeof(options))
  # Ajouter la paire nom de champ (comme un symbole) et sa valeur dans le dictionnaire
  options_dict[field] = getfield(options, field)
end

# Benchmark : 
using SolverBenchmark

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

p = performance_profile(stats, df->Float64.(df.neval_grad))

using DataFrames
df = DataFrame(R2_Reg_neval_grad = stats[:R2_Reg].neval_grad, iR2_Reg_neval_grad = stats[:iR2_Reg].neval_grad)
df.Diff = df.R2_Reg_neval_grad - df.iR2_Reg_neval_grad
df