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
nlp = curly30(;backend=:generic)
h = NormL1(1.0)
options = ROSolverOptions(verbose=3, maxIter = 1000)

####################

params = iR2RegParams([Float16, Float32, Float64], verb=true, activate_mp=true)

jso_res = RegularizedOptimization.R2(nlp, h, options)
my_res = iR2(nlp, h, options, params) # launches vanilla R2-Reg (one might add verbose=1 for more verbosity)

####################
# Benchmark : 
using SolverBenchmark
using CS
using DataFrames
options = ROSolverOptions(verbose=0)
h = NormL1(1.0)

## sans MP ##
problems_sans_mp = (eval(Meta.parse(problem))() for problem ∈ OptimizationProblems.meta[!, :name])
params_sans_mp = iR2RegParams([Float64], verb=false, activate_mp=false)
solvers_sans_mp = Dict(
  :R2_Reg => model -> RegularizedOptimization.R2(model, h, options),
  :iR2_Reg => model -> iR2(model, h, options, params_sans_mp),
)

stats_sans_mp = bmark_solvers(
  solvers_sans_mp, problems_sans_mp,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < 4),
)


## avec MP ##
problems_mp = (eval(Meta.parse(problem))(;backend = :generic) for problem ∈ OptimizationProblems.meta[!, :name])
params_mp = iR2RegParams([Float16, Float32, Float64], activate_mp=true)


solvers_avec_mp = Dict(
  :iR2_Reg => model -> iR2(model, h, options, params_mp)
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



##########################
Π = [Float16, Float32, Float64]
params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true)
options = ROSolverOptions(verbose=0, maxIter = 1000)
df_str = ":name => String[], :status => Symbol[], :objective => Real[]"
col_str = [":neval_obj_",":neval_h_",":neval_grad_",":neval_prox_"]
for col in col_str
  for fp in Π
    df_str *= ","*col*"$fp => Int64[]"
  end
end
stats_ir2 = eval(Meta.parse("DataFrame($df_str)"))
meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems
filter!(row -> row[:name] != "cosine", names_pb_vars)
filter!(row -> row[:name] != "scosine", names_pb_vars)
filter!(row -> row[:name] != "rat42", names_pb_vars)
filter!(row -> row[:name] != "rat43", names_pb_vars)


for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
  @show nlp.meta.name
  params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true)
  
  stat_ir2 = iR2(nlp, h, options, params_mp)
  push!(stats_ir2,
      [nlp.meta.name,
      stat_ir2.status,
      [stat_ir2.objective]...,
      [stat_ir2.solver_specific[:special_counters][:f][i] for i = 1:length(Π)]...,
      [stat_ir2.solver_specific[:special_counters][:h][i] for i = 1:length(Π)]...,
      [stat_ir2.solver_specific[:special_counters][:∇f][i] for i = 1:length(Π)]...,
      [stat_ir2.solver_specific[:special_counters][:prox][i] for i = 1:length(Π)]...]
    )
end


df_str = ":name => String[], :status => Symbol[], :objective => Real[], :neval_obj => Int[], :neval_h => Int[], :neval_grad => Int[], :neval_prox => Int[]"
stats_r2 = eval(Meta.parse("DataFrame($df_str)"))

for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
  @show nlp.meta.name
  
  stat_r2 = RegularizedOptimization.R2(nlp, h, options)
  push!(stats_r2,
      [nlp.meta.name,
      stat_r2.status,
      [stat_r2.objective]...,
      [nlp.counters.neval_obj]...,
      [stat_r2.iter]...,
      [nlp.counters.neval_grad]...,
      [stat_r2.iter]...,]
    )
end

stats_mp = Dict(:iR2Reg => stats_ir2,
                :R2Reg => stats_r2
)

p = performance_profile(stats_mp, df -> Float64.(df.neval_grad_Float16 + 4*df.neval_grad_Float32 + 16*df.neval_grad_Float64))

# pour chaque problème, je calcule le nombre d'évaluations de la fonction objectif, de la fonction h, du gradient et de la proximale et je divise par le nombre d'itérations
cost_grad_ir2 = sum(stats_ir2[!,:neval_grad_Float16]) + 4*sum(stats_ir2[!,:neval_grad_Float32]) + 16*sum(stats_ir2[!,:neval_grad_Float64])
cost_grad_r2 = 16*sum(stats_r2[!,:neval_grad])


sum(stats_ir2[!,:neval_grad_Float64]) /( sum(stats_ir2[!,:neval_grad_Float16]) + sum(stats_ir2[!,:neval_grad_Float32]) + sum(stats_ir2[!,:neval_grad_Float64]))
sum(stats_ir2[!,:neval_grad_Float16]) + 4*sum(stats_ir2[!,:neval_grad_Float32]) + 16*sum(stats_ir2[!,:neval_grad_Float64])
sum(16*stats_r2[!,:neval_grad]) 

