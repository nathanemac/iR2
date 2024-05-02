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
# tests sur 1 problème
nlp = woods(;n=100, type = Val(Float64), backend=:generic)
nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1.2, -1.345], backend=:generic)
h = NormL1(1.0)
options = ROSolverOptions(verbose=0, maxIter = 100)
params = iR2RegParams([Float16, Float32, Float64], verb=true, activate_mp=true)
jso_res = RegularizedOptimization.R2(nlp, h, options)
my_res = iR2(nlp, h, options, params) # launches vanilla R2-Reg (one might add verbose=1 for more verbosity)


####################
# Benchmark : 
using SolverBenchmark
using DataFrames
using Statistics

##########################
Π = [Float16, Float32, Float64]
h = NormL1(1.0)

params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true)
options = ROSolverOptions(verbose=0, maxIter = 1000)
df_str = ":name => String[], :status => Symbol[], :objective => Real[], :iter => Int[], :elapsed_time => Float64[]"
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
      [stat_ir2.iter]...,
      [stat_ir2.elapsed_time]...,
      [stat_ir2.solver_specific[:special_counters][:f][i] for i = 1:length(Π)]...,
      [stat_ir2.solver_specific[:special_counters][:h][i] for i = 1:length(Π)]...,
      [stat_ir2.solver_specific[:special_counters][:∇f][i] for i = 1:length(Π)]...,
      [stat_ir2.solver_specific[:special_counters][:prox][i] for i = 1:length(Π)]...]
    )
end


df_str = ":name => String[], :status => Symbol[], :objective => Real[], :iter => Int[], :elapsed_time => Float64[], :neval_obj => Int[], :neval_h => Int[], :neval_grad => Int[], :neval_prox => Int[]"
stats_r2 = eval(Meta.parse("DataFrame($df_str)"))

for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
  @show nlp.meta.name
  
  stat_r2 = RegularizedOptimization.R2(nlp, h, options)
  push!(stats_r2,
      [nlp.meta.name,
      stat_r2.status,
      [stat_r2.objective]...,
      [stat_r2.iter]...,
      [stat_r2.elapsed_time]...,
      [nlp.counters.neval_obj]...,
      [stat_r2.iter]...,
      [nlp.counters.neval_grad]...,
      [stat_r2.iter]...,]
    )
end
stats_r2[!, :neval_grad_Float16] = zeros(nrow(stats_r2))
stats_r2[!, :neval_grad_Float32] = zeros(nrow(stats_r2))
stats_r2[!, :neval_grad_Float64] = stats_r2[!,:neval_grad]

stats_mp = Dict(:iR2Reg => stats_ir2,
                :R2Reg => stats_r2
)

# pour chaque problème, je calcule le cout d'évaluations du gradient avec pour cout d'evaluation de 1 gradient en Float16 = 1, en Float32 = 4 et en Float64 = 16
cost_grad_ir2 = sum(stats_ir2[!,:neval_grad_Float16]) + 4*sum(stats_ir2[!,:neval_grad_Float32]) + 16*sum(stats_ir2[!,:neval_grad_Float64])
cost_grad_r2 = 16*sum(stats_r2[!,:neval_grad])

# pourcentage d'évaluations du gradient en Float16, Float32 et Float64
percentage_eval_grad_f64=sum(stats_ir2[!,:neval_grad_Float64]) /( sum(stats_ir2[!,:neval_grad_Float16]) + sum(stats_ir2[!,:neval_grad_Float32]) + sum(stats_ir2[!,:neval_grad_Float64]))
percentage_eval_grad_f32=sum(stats_ir2[!,:neval_grad_Float32]) /( sum(stats_ir2[!,:neval_grad_Float16]) + sum(stats_ir2[!,:neval_grad_Float32]) + sum(stats_ir2[!,:neval_grad_Float64]))
percentage_eval_grad_f16=sum(stats_ir2[!,:neval_grad_Float16]) /( sum(stats_ir2[!,:neval_grad_Float16]) + sum(stats_ir2[!,:neval_grad_Float32]) + sum(stats_ir2[!,:neval_grad_Float64]))


# nombre de problemes pour lesquels les 2 algos ont first_order
n_FO_iR2Reg = nrow(filter(row -> row[:status] == :first_order, stats_ir2))
n_FO_R2Reg = nrow(filter(row -> row[:status] == :first_order, stats_r2))

# nombre de problemes pour lesquels les 2 algos ont max_iter
n_MI_iR2Reg = nrow(filter(row -> row[:status] == :max_iter, stats_ir2))
n_MI_R2Reg = nrow(filter(row -> row[:status] == :max_iter, stats_r2))

# nombre de problemes pour lesquels les 2 algos ont un autre status (:exception)
n_Autre_iR2Reg = nrow(filter(row -> row[:status] == :exception, stats_ir2))
n_Autre_R2Reg = nrow(filter(row -> row[:status] == :exception, stats_r2))


#### performance profile : 

# sur tous les problèmes :
solved(df) = df.status .== :first_order
costnames = ["gradient evaluation cost", "time"]
costs = [
  df -> .!solved(df) .* Inf .+ Float64.(df.neval_grad_Float16 + 4*df.neval_grad_Float32 + 16*df.neval_grad_Float64),
  df -> .!solved(df) .* Inf .+ df.elapsed_time
]
gr()
profile_solvers(stats_mp, costs, costnames)

# sur les problèmes résolus par iR2-Reg:
FO_iR2Reg = filter(row -> row[:status] == :first_order, stats_ir2)
extracted_R2Reg = filter(row -> row[:name] in FO_iR2Reg[!, :name], stats_r2)
stats_mp_2 = Dict(:iR2Reg => FO_iR2Reg,
                :R2Reg => extracted_R2Reg
)

gr()
profile_solvers(stats_mp_2, costs, costnames)







# convergence plot sur Rosenbrock 2D:
Fhist_jso = jso_res.solver_specific[:Fhist][10:end]
Fhist_my = my_res.solver_specific[:Fhist][10:end]
ymin, ymax = 0.99, 1.  # Définissez les limites selon vos données et vos besoins
plot(Fhist_jso, label="jso_res Fhist", title="Objective History", xlabel="Iterations", ylabel="Objective Value", ylim=(ymin, ymax), yaxis = :log10)
plot!(Fhist_my, label="my_res Fhist")

using BenchmarkProfiles
using BenchmarkTools
@benchmark iR2(nlp, h, options, params_mp)
@benchmark RegularizedOptimization.R2(nlp, h, options)
