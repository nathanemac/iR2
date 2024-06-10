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
nlp = AMPGO20(;n=100, type = Val(Float64), backend=:generic)
nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1.2, -1.345], backend=:generic)
h = NormL1(1.0)
options = ROSolverOptions(verbose=1, maxIter = 100, ϵa = 1e-4, ϵr = 1e-4)
params = iR2RegParams([Float16, Float32, Float64],  verb=false, activate_mp=true, κξ = 1., pf = 1, ps = 1, pg = 1, ph = 1)
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
params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ=1.)
options = ROSolverOptions(verbose=0, maxIter = 1000, ϵa = 1e-4, ϵr = 1e-4)
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
  params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ = 1.)
  
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
costnames = ["gradient evaluation cost"]
costs = [
  df -> .!solved(df) .* Inf .+ Float64.(df.neval_grad_Float16 + 4*df.neval_grad_Float32 + 16*df.neval_grad_Float64),
  #df -> .!solved(df) .* Inf .+ df.elapsed_time
]
gr()
profile_solvers(stats_mp, costs, costnames, size = (800, 400), margin = 10Plots.px,)

# sur les problèmes résolus par iR2-Reg:
FO_iR2Reg = filter(row -> row[:status] == :first_order, stats_ir2)
extracted_R2Reg = filter(row -> row[:name] in FO_iR2Reg[!, :name], stats_r2)
delete!(extracted_R2Reg, 63)
stats_mp_2 = Dict(:iR2Reg => FO_iR2Reg,
                :R2Reg => extracted_R2Reg
)

gr()
profile_solvers(stats_mp_2, costs, costnames, size = (800, 400), margin = 10Plots.px,)
title!("gradient evaluation cost on problems solved by iR2Reg")

# convergence plot sur Rosenbrock 2D:
Fhist_jso = jso_res.solver_specific[:Fhist][10:end]
Fhist_my = my_res.solver_specific[:Fhist][10:end]
ymin, ymax = 0.99, 1.  # Définissez les limites selon vos données et vos besoins
plot(Fhist_jso, label="jso_res Fhist", title="Objective History", xlabel="Iterations", ylabel="Objective Value", ylim=(ymin, ymax), yaxis = :log10)
plot!(Fhist_my, label="my_res Fhist")


# 
using BenchmarkProfiles
using BenchmarkTools
@benchmark iR2(nlp, h, options, params_mp)
@benchmark RegularizedOptimization.R2(nlp, h, options)

# évolution du % d'évaluation du gradient et du % de problemes résolus en first_order pour iR2 et R2 en fonction de maxIter
MaxIters = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]
stats_max_iters = Dict(
  :percentage_f16 => Float64[],
  :percentage_f32 => Float64[],
  :nb_fo_r2 => Int[],
  :nb_mi_r2 => Int[],
  :nb_fo_ir2 => Int[],
  :nb_mi_ir2 => Int[],
  :cost_grad_ir2 => Int[],
  :cost_grad_r2 => Int[]

)
for mI in MaxIters
  params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ = 1.)
  options = ROSolverOptions(verbose=0, maxIter = mI, ϵa = 1e-4, ϵr = 1e-4)
  df_str = ":name => String[], :status => Symbol[], :objective => Real[], :iter => Int[], :elapsed_time => Float64[]"
  col_str = [":neval_obj_",":neval_h_",":neval_grad_",":neval_prox_"]
  for col in col_str
    for fp in Π
      df_str *= ","*col*"$fp => Int64[]"
    end
  end
  stats_ir2 = eval(Meta.parse("DataFrame($df_str)"))

  for pb in eachrow(names_pb_vars)
    nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
    params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ = 1.)
  
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
      [stat_ir2.solver_specific[:special_counters][:prox][i] for i = 1:length(Π)]...,]
    )
  end

  df_str = ":name => String[], :status => Symbol[], :objective => Real[], :iter => Int[], :elapsed_time => Float64[], :neval_obj => Int[], :neval_h => Int[], :neval_grad => Int[], :neval_prox => Int[]"
  stats_r2 = eval(Meta.parse("DataFrame($df_str)"))

  for pb in eachrow(names_pb_vars)
    nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
  
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

  cost_grad_ir2 = sum(stats_ir2[!,:neval_grad_Float16]) + 4*sum(stats_ir2[!,:neval_grad_Float32]) + 16*sum(stats_ir2[!,:neval_grad_Float64])
  cost_grad_r2 = 16*sum(stats_r2[!,:neval_grad])

  percentage_eval_grad_f32=sum(stats_ir2[!,:neval_grad_Float32]) /( sum(stats_ir2[!,:neval_grad_Float16]) + sum(stats_ir2[!,:neval_grad_Float32]) + sum(stats_ir2[!,:neval_grad_Float64]))
  percentage_eval_grad_f16=sum(stats_ir2[!,:neval_grad_Float16]) /( sum(stats_ir2[!,:neval_grad_Float16]) + sum(stats_ir2[!,:neval_grad_Float32]) + sum(stats_ir2[!,:neval_grad_Float64]))


  nb_first_order_ir2 = nrow(filter(row -> row[:status] == :first_order, stats_ir2))
  nb_max_iter_ir2 = nrow(filter(row -> row[:status] == :max_iter, stats_ir2))

  nb_first_order_r2 = nrow(filter(row -> row[:status] == :first_order, stats_r2))
  nb_max_iter_r2 = nrow(filter(row -> row[:status] == :max_iter, stats_r2))

  push!(stats_max_iters[:percentage_f16], percentage_eval_grad_f16)
  push!(stats_max_iters[:percentage_f32], percentage_eval_grad_f32)
  push!(stats_max_iters[:nb_fo_r2], nb_first_order_r2)
  push!(stats_max_iters[:nb_mi_r2], nb_max_iter_r2)
  push!(stats_max_iters[:nb_fo_ir2], nb_first_order_ir2)
  push!(stats_max_iters[:nb_mi_ir2], nb_max_iter_ir2)
  push!(stats_max_iters[:cost_grad_ir2], cost_grad_ir2)
  push!(stats_max_iters[:cost_grad_r2], cost_grad_r2)

end

plot(MaxIters, 100 .* stats_max_iters[:percentage_f16], size = (800, 400), label="% eval grad f16", xlabel="value of maxIter", ylabel="% of problems solved", legend=:topright, title="% of evaluations of the gradient in FP formats for iR2",  margin = 10Plots.px)
plot!(MaxIters, 100 .* stats_max_iters[:percentage_f32], label="% eval grad f32")
plot!(MaxIters, 100 .* (1 .- stats_max_iters[:percentage_f32] .- stats_max_iters[:percentage_f16]), label="% eval grad f64")

plot(MaxIters, stats_max_iters[:nb_fo_r2] ./ 174, label="% first order r2", size = (800, 400), xlabel="value of maxIter", ylabel="% of problems solved", legend=:bottomright, title="% of problems solved for iR2 and R2",  margin = 10Plots.px)
plot!(MaxIters, stats_max_iters[:nb_fo_ir2] ./ 174, label="% first order ir2")

plot(MaxIters, stats_max_iters[:cost_grad_ir2], label="cost grad eval iR2", size = (800, 400), xlabel="value of maxIter", ylabel="gradient evaluation costs", legend=:bottomright, title="Gradient evaluation costs in relation to maxIter",  margin = 10Plots.px)
plot!(MaxIters, stats_max_iters[:cost_grad_r2], label="cost grad eval R2")

# évolution du % de problèmes résolus en fonction de la valeur de κξ pour iR2 pour maxIter fixé
Kξ = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]
stats_Kξ = Dict(
  :nb_fo_r2 => Int[],
  :nb_mi_r2 => Int[],
  :nb_fo_ir2 => Int[],
  :nb_mi_ir2 => Int[], 
  :cost_grad_ir2 => Int[],
  :cost_grad_r2 => Int[])
for κξ in Kξ
  params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ=κξ)
  options = ROSolverOptions(verbose=0, maxIter = 1000, ϵa = 1e-4, ϵr = 1e-4)
  df_str = ":name => String[], :status => Symbol[], :objective => Real[], :iter => Int[], :elapsed_time => Float64[]"
  col_str = [":neval_obj_",":neval_h_",":neval_grad_",":neval_prox_"]
  for col in col_str
    for fp in Π
      df_str *= ","*col*"$fp => Int64[]"
    end
  end
  stats_ir2 = eval(Meta.parse("DataFrame($df_str)"))

  for pb in eachrow(names_pb_vars)
    nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
    params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ=κξ)
  
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

  nb_first_order_ir2 = nrow(filter(row -> row[:status] == :first_order, stats_ir2))
  nb_max_iter_ir2 = nrow(filter(row -> row[:status] == :max_iter, stats_ir2))

  push!(stats_Kξ[:nb_fo_ir2], nb_first_order_ir2)
  push!(stats_Kξ[:nb_mi_ir2], nb_max_iter_ir2)

  cost_grad_ir2 = sum(stats_ir2[!,:neval_grad_Float16]) + 4*sum(stats_ir2[!,:neval_grad_Float32]) + 16*sum(stats_ir2[!,:neval_grad_Float64])
  cost_grad_r2 = 16*sum(stats_r2[!,:neval_grad])
  push!(stats_Kξ[:cost_grad_ir2], cost_grad_ir2)

end
for κξ in Kξ
  push!(stats_Kξ[:nb_fo_r2], 91)
  push!(stats_Kξ[:nb_mi_r2], 83)
end
for κξ in Kξ
  push!(stats_Kξ[:cost_grad_r2], 772000)
end

plot(Kξ, stats_Kξ[:nb_fo_r2] ./ 174, label="% first order r2", size = (800, 400), xlabel="value of κξ", ylabel="% of problems solved", legend=:bottomright, title="% solved",  margin = 10Plots.px, xaxis = :log10)
plot!(Kξ, stats_Kξ[:nb_fo_ir2] ./ 174, label="% first order ir2")
plot(Kξ, stats_Kξ[:cost_grad_ir2], label="cost grad eval iR2", size = (800, 400), xlabel="value of κξ", ylabel="gradient evaluation costs", legend=:bottomleft, title="Gradient evaluation costs in relation to κξ",  margin = 10Plots.px, xaxis = :log10)
plot!(Kξ, stats_Kξ[:cost_grad_r2], label="cost grad eval R2")


# trying to find out why iR2 is not able to solve some problems that R2 can solve
for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Val(Float64),backend = :generic)"))
  params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ=1.)
  options = ROSolverOptions(verbose=0, maxIter = 1000, ϵa = 1e-4, ϵr = 1e-4)
  my_res = iR2(nlp, h, options, params_mp)
  jso_res = RegularizedOptimization.R2(nlp, h, options)
  if my_res.status != :first_order && jso_res.status == :first_order
    println(" $(nlp.meta.name)")
  end
end
nlp=jennrichsampson(; type = Val(Float64), backend=:generic)
options = ROSolverOptions(verbose=1, maxIter = 5, ϵa = 1e-4, ϵr = 1e-4)
params_mp = iR2RegParams([Float16, Float32, Float64], verb=false, activate_mp=true, κξ=1.)
my_res = iR2(nlp, h, options, params_mp)
jso_res = RegularizedOptimization.R2(nlp, h, options)

# sqrt_ξ_νInv is NaN at first iteration. Keep digging.



############## TV PROBLEM ################

include("../TV.jl/tv.jl")
include("../TV.jl/utils.jl")

