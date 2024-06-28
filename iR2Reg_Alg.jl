import SolverCore.solve!

mutable struct iR2RegParams{T<:Real, H<:Real}
  pf::Int
  pg::Int
  ph::Int
  ps::Int
  Π::Vector{DataType}
  verb::Bool
  activate_mp::Bool
  flags::Vector{Bool}
  κf::T
  κh::T
  κ∇::T
  κs::T
  κξ::T
  σk::H # H is the "Highest" floating point format in Π.
  ν::H
end


function iR2RegParams(Π::Vector{DataType}; pf = 1, pg = 1, ph = 1, ps = 1, verb::Bool=false, activate_mp::Bool=true, flags::Vector{Bool}=[false, false, false], κf=1e-5, κh=2e-5, κ∇=4e-2, κs=1., κξ=1e-4, σk=Π[end](1.), ν=eps(Π[end])^(1/5))
  return iR2RegParams(pf, pg, ph, ps, Π, verb, activate_mp, flags, κf, κh, κ∇, κs, κξ, σk, ν)
end



mutable struct iR2Solver{R<:Real, S<:AbstractVector} <: AbstractOptimizationSolver #G <: Union{ShiftedProximableFunction, Nothing}
  xk::Vector{S}
  mν∇fk::Vector{S}
  gfk::Vector{S}
  fk::S
  hk::S
  sk::Vector{S}
  xkn::S
  has_bnds::Bool
  l_bound::S
  u_bound::S
  l_bound_m_x::S
  u_bound_m_x::S
  Fobj_hist::Vector{R}
  Hobj_hist::Vector{R}
  Complex_hist::Vector{Int}
  p_hist::Vector{Vector{Int}}
  special_counters::Dict{Symbol, Vector{Int}}
  ψ#::G
  params::iR2RegParams
end


function iR2Solver(
  reg_nlp::AbstractRegularizedNLPModel,
  params::iR2RegParams,
  options::ROSolverOptions,
) where {T, V}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar
  max_iter = options.maxIter
  Π = params.Π
  R = eltype(x0)
  xk = [Vector{eltype(T)}(undef, length(x0)) for T in Π]
  gfk = [Vector{eltype(T)}(undef, length(x0)) for T in Π]
  fk = [zero(eltype(T)) for T in Π]
  hk = [zero(eltype(T)) for T in Π]
  sk = [Vector{eltype(T)}(undef, length(x0)) for T in Π]
  mν∇fk = [Vector{eltype(T)}(undef, length(x0)) for T in Π]
  xkn = similar(x0)
  has_bnds = any(l_bound .!= R(-Inf)) || any(u_bound .!= R(Inf))
  if has_bnds
    l_bound_m_x = similar(x0)
    u_bound_m_x = similar(x0)
  else
    l_bound_m_x = similar(x0, 0)
    u_bound_m_x = similar(x0, 0)
  end
  Fobj_hist = zeros(R, max_iter+2)
  Hobj_hist = zeros(R, max_iter+2)
  Complex_hist = zeros(Int, max_iter+2)
  p_hist = [zeros(Int, 4) for _ in 1:max_iter]
  special_counters = Dict(:f => zeros(Int, length(Π)), :h => zeros(Int, length(Π)), :∇f => zeros(Int, length(Π)), :prox => zeros(Int, length(Π)))
  ψ = has_bnds ? shifted(reg_nlp.h, x0, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, x0)
  return iR2Solver(
    xk,
    mν∇fk,
    gfk,
    fk, 
    hk, 
    sk,
    xkn,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    Fobj_hist,
    Hobj_hist,
    Complex_hist,
    p_hist,
    special_counters,
    ψ, 
    params
  )
end

"""
    iR2_lazy(nlp, h, options)
    iR2_lazy(f, ∇f!, h, options, x0)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs is the Taylor linear approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and σₖ > 0 is the regularization parameter.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `x0::AbstractVector`: an initial guess (in the second calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first calling form: default = `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:length(x0)`).

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`.

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function iR2_lazy(
  nlp::AbstractNLPModel{R, V},
  h,
  options::ROSolverOptions{R},
  params::iR2RegParams;
  kwargs...) where{ R <: Real, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:nlp.meta.nvar) 
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return iR2_lazy(
    reg_nlp,
    params,
    options, 
    x = x0,
    atol = options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    σmin = options.σmin,
    η1 = options.η1,
    η2 = options.η2,
    ν = options.ν,
    γ = options.γ,
  )
  return stats
end

function iR2_lazy(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real}
  nlp = FirstOrderModel(f,∇f!,x0)
  reg_nlp = RegularizedNLPModel(nlp,h,selected) 
  stats = iR2_lazy(
  reg_nlp,
  params,
  options, 
  x=x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.max_iter,
  max_time = options.maxTime,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  ν = options.ν,
  γ = options.γ,
  )
  outdict = Dict(
     :Fhist => stats.solver_specific[:Fhist],
     :Hhist => stats.solver_specific[:Hhist],
     :Chist => stats.solver_specific[:SubsolverCounter],
     :NonSmooth => h,
     :status => stats.status,
     :fk => stats.solver_specific[:smooth_obj],
     :hk => stats.solver_specific[:nonsmooth_obj],
     :ξ => stats.solver_specific[:xi],
     :elapsed_time => stats.elapsed_time,
     :p_hist => stats.solver_specific[:p_hist],
     :special_counters => stats.solver_specific[:special_counters],
   )
  
return stats.solution,stats.iter,outdict
end

function iR2_lazy(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  params::iR2RegParams,
  x0::AbstractVector{R},
  l_bound::AbstractVector{R},
  u_bound::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real}
  nlp = FirstOrderModel(f,∇f!,x0,lcon = l_bound, ucon = u_bound)
  reg_nlp = RegularizedNLPModel(nlp,h,selected) 
  stats = iR2_lazy(
  reg_nlp,
  params, 
  options,
  x=x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.max_iter,
  max_time = options.maxTime,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  ν = options.ν,
  γ = options.γ,
)
  outdict = Dict(
     :Fhist => stats.solver_specific[:Fhist],
     :Hhist => stats.solver_specific[:Hhist],
     :Chist => stats.solver_specific[:SubsolverCounter],
     :NonSmooth => h,
     :status => stats.status,
     :fk => stats.solver_specific[:smooth_obj],
     :hk => stats.solver_specific[:nonsmooth_obj],
     :ξ => stats.solver_specific[:xi],
     :elapsed_time => stats.elapsed_time,
     :p_hist => stats.solver_specific[:p_hist],
     :special_counters => stats.solver_specific[:special_counters],
   )
  
return stats.solution,stats.iter,outdict
end

function iR2_lazy(reg_nlp::AbstractRegularizedNLPModel, params::iR2RegParams, options::ROSolverOptions; kwargs...)
  kwargs_dict = Dict(kwargs...)
  solver = iR2Solver(reg_nlp, params, options)
  stats = GenericExecutionStats(reg_nlp.model)
  cb = (nlp, solver, stats) -> begin
    solver.Fobj_hist[stats.iter+1] = stats.solver_specific[:smooth_obj]
    solver.Hobj_hist[stats.iter+1] = stats.solver_specific[:nonsmooth_obj]
    solver.Complex_hist[stats.iter+1] += 1
  end
  solve!(
    solver,
    reg_nlp,
    stats,
    options;
    callback = cb,
    kwargs...
  )
  set_solver_specific!(stats, :Fhist, solver.Fobj_hist[1:stats.iter+1])
  set_solver_specific!(stats, :Hhist, solver.Hobj_hist[1:stats.iter+1])
  set_solver_specific!(stats, :SubsolverCounter, solver.Complex_hist[1:stats.iter+1])
  return stats
end

function solve!(
  solver::iR2Solver,
  reg_nlp::AbstractRegularizedNLPModel{T, V}, 
  stats::GenericExecutionStats{T, V},
  options::ROSolverOptions{T};
  callback = (args...) -> nothing,
  max_eval::Int = -1,
  kwargs...,
  ) where {T, V}

  reset!(stats)

  p = solver.params

  ϵ = options.ϵa
  ϵr = options.ϵr
  neg_tol = options.neg_tol
  verbose = options.verbose
  max_iter = options.maxIter
  max_time = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  x0 = reg_nlp.model.meta.x0

  selected = reg_nlp.selected
  h = reg_nlp.h
  nlp = reg_nlp.model
  

  if p.activate_mp
    check_κ_valid(p.κs, p.κf, p.κ∇, p.κh, η1, η2)
  end  

  Π = p.Π
  P = length(Π)

  for i=1:P 
    solver.xk[i] .= x0
  end
  has_bnds = solver.has_bnds
  if has_bnds
    l_bound = solver.l_bound
    u_bound = solver.u_bound
    l_bound_m_x = solver.l_bound_m_x
    u_bound_m_x = solver.u_bound_m_x
  end
  Fobj_hist = solver.Fobj_hist
  Hobj_hist = solver.Hobj_hist
  Complex_hist = solver.Complex_hist
  p_hist = solver.p_hist

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(max_iter / 10)
  elseif verbose == 2
    ptf = round(max_iter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  improper = false
  hxk = @views h(solver.xk[p.ph][selected]) # ph = 1 au début
  solver.special_counters[:h][p.ph] += 1
  if hxk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(solver.xk[p.ph][selected], h, x0, one(eltype(x0)))
    hxk = @views h(solver.xk[p.ph][selected])
    if hxk == Inf
      status = :exception 
      @warn "prox computation must be erroneous. Early stopping iR2-Reg."
      return 1, status, solver.fk[end], solver.hk[end], Inf, [p.pf, p.pg, p.ph, p.ps]
    end
  end
  for i=1:P 
    solver.hk[i] = Π[i].(hxk)
  end
  improper = (solver.hk[end] == -Inf)

  if verbose > 0
    @info log_header(
      [:iter, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :arrow],
      [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Char],
      hdr_override = Dict{Symbol,String}(   # TODO: Add this as constant dict elsewhere
        :iter => "iter",
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ/ν)",
        :ρ => "ρ",
        :σ => "σ",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :arrow => " "
      ),
      colsep = 1,
    )
  end

  local ξ
  p.σk = max(1 / p.ν, options.σmin)
  
  p.ν = 1 / p.σk
  sqrt_ξ_νInv = 1.0

  fxk = obj(nlp, solver.xk[p.pf]) 
  solver.special_counters[:f][p.pf] += 1 # on incrémente le compteur de f en la précision pf.
  for i=1:P
    solver.fk[i] = Π[i](fxk) # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
  end 

  grad!(nlp, solver.xk[p.pg], solver.gfk[p.pg])
  solver.special_counters[:∇f][p.pg] += 1
  for i=1:P 
    solver.gfk[i] .= solver.gfk[p.pg]
    solver.mν∇fk[i] .= -Π[end].(p.ν) * solver.gfk[i] 
  end

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, T(solver.fk[end]) + T(solver.hk[end])) # TODO maybe change this to avoid casting
  set_solver_specific!(stats,:smooth_obj, T(solver.fk[end]))
  set_solver_specific!(stats,:nonsmooth_obj, T(solver.hk[end]))
  h = NormL1(Π[p.ps](1.0)) # need to redefine h at each iteration because when shifting: MethodError: no method matching shifted(::NormL1{Float64}, ::Vector{Float16}) so the norm and the shift vector must be same FP Format. 
  solver.ψ = shifted(h, solver.xk[p.ps]) # therefore ψ FP format is s FP format
  φk(d) = dot(solver.gfk[p.ps], d)   
  mk(d) = φk(d) + solver.ψ(d)
  prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps](p.ν))
  for i=1:P
    solver.sk[i] .= solver.sk[p.ps] 
  end

  mks = mk(solver.sk[p.ps]) # on evite les casts en mettant tout en la précision de s

  ξ = solver.hk[p.ps] - mks + max(1, abs(solver.hk[p.ps])) * 10 * eps(Π[p.ps]) # on evite les casts en mettant tout en la précision de s
  ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)") # TODO : est-ce que ξ est pas mieux en H ? 
  sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)
  ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative

  set_solver_specific!(stats, :xi, sqrt_ξ_νInv)

  solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ * sqrt(p.κξ))
  set_status!(
    stats,
    get_status(
      reg_nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      improper = improper,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  # Implémentation d'une fonction qui s'occupe de la boucle principale de l'algo :

  function inner_loop!(solver, stats, options, selected, h, p, Π, sqrt_ξ_νInv, verbose, max_iter, max_time, η1, η2, γ, start_time, T, P)

    while !done
      # Update xk, sigma_k
      solver.xkn .= solver.xk[p.ps] .+ solver.sk[p.ps]
      fkn = obj(nlp, solver.xkn)
      hkn = @views h(solver.xkn[selected])
      improper = (hkn == -Inf)

      Δobj = (solver.fk[end] + solver.hk[end]) - (fkn + hkn) + max(1, abs(solver.fk[end] + solver.hk[end])) * 10 * eps(Π[end])
      global ρk = Δobj / ξ  

      verbose > 0 && 
      stats.iter % verbose == 0 &&
      @info log_row(Any[stats.iter, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, ρk, p.σk, norm(solver.xk[end]), norm(solver.sk[end]), (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")], colsep = 1)

      if η1 ≤ ρk < Inf
        solver.xk[p.ps] .= solver.xkn
        if has_bnds #TODO
          @error "Not implemented yet"
          @. l_bound_m_x = l_bound - xk[end]
          @. u_bound_m_x = u_bound - xk[end]
          set_bounds!(solver.ψ, l_bound_m_x, u_bound_m_x)
        end
        solver.fk[p.pf] = fkn
        solver.hk[p.ph] = hkn
        grad!(nlp, solver.xk[p.pg], solver.gfk[p.pg])
        shift!(solver.ψ, solver.xk[p.ps])
        for i=1:P
          solver.xk[i] .= solver.xk[p.ps] # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
        end
        for i=1:P
          solver.fk[i] = solver.fk[p.pf] # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
        end
        for i=1:P
          solver.hk[i] = solver.hk[p.ph] # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
        end  
        for i=1:P 
          solver.gfk[i] .= solver.gfk[p.pg]
        end
      end

      if η2 ≤ ρk < Inf
        p.σk = max(p.σk / γ, options.σmin)
      end
      if ρk < η1 || ρk == Inf
        p.σk = p.σk * γ
      end
      p.ν = 1 / p.σk
      for i=1:P
        solver.mν∇fk[i] .= -p.ν * solver.gfk[i]
      end
      for i=1:P
        solver.sk[i] .= solver.sk[p.ps] 
      end


      set_objective!(stats, T(solver.fk[end] + solver.hk[end]))
      set_solver_specific!(stats,:smooth_obj, solver.fk[end])
      set_solver_specific!(stats,:nonsmooth_obj, solver.hk[end])
      set_iter!(stats, stats.iter + 1)
      set_time!(stats, time() - start_time)

      φk(d) = dot(solver.gfk[p.ps], d)   
      mk(d) = φk(d) + solver.ψ(d)
      prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps](p.ν))

      mks = mk(solver.sk[p.ps]) # on evite les casts en mettant tout en la précision de s
      ξ = solver.hk[p.ps] - mks + max(1, abs(solver.hk[p.ps])) * 10 * eps(Π[p.ps]) # on evite les casts en mettant tout en la précision de s

      sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)
      solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ * sqrt(p.κξ))

      set_solver_specific!(stats, :xi, sqrt_ξ_νInv)
      set_status!(
      stats,
      get_status(
        reg_nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        improper = improper,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter
      ),
      )

      callback(nlp, solver, stats)

      done = stats.status != :unknown
    end
  end

  inner_loop!(solver, stats, options, selected, h, p, Π, sqrt_ξ_νInv, verbose, max_iter, max_time, η1, η2, γ, start_time, T, P)

    verbose > 0 &&
    if stats.status == :first_order
      @info log_row(Any[stats.iter, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, ρk, p.σk, norm(solver.xk[end]), norm(solver.sk[end]), (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")], colsep = 1)
      @info "R2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
    end

    set_solution!(stats, solver.xk[end])
    return stats
end

#       # define model
#       if solver.has_bnds #TODO updatde this later 
#         @. solver.l_bound_m_x = solver.l_bound - solver.xk[1]
#         @. solver.u_bound_m_x = solver.u_bound - solver.xk[1]
#         solver.ψ = shifted(reg_nlp.h, solver.xk[1], l_bound_m_x, u_bound_m_x, selected)
#       else
#         reg_nlp.h = NormL1(Π[p.ps](1.0)) # need to redefine h at each iteration because when shifting: MethodError: no method matching shifted(::NormL1{Float64}, ::Vector{Float16}) so the norm and the shift vector must be same FP Format. 
#         solver.ψ = shifted(reg_nlp.h, solver.xk[p.ps]) # therefore ψ FP format is s FP format
#       end
#       φk(d) = dot(solver.gfk[p.pg], d) 
#       mk(d) = φk(d) + solver.ψ(d)

#       prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps].(p.ν)) 
#       solver.special_counters[:prox][p.ps] += 1

#       solver.Complex_hist[k] += 1
#       for i=1:P
#         solver.sk[i] .= solver.sk[p.ps] # on a mis a jour solver.sk[p.ps] dans prox!() donc c'est celui qu'on met à jour. 
#       end

#       if p.activate_mp 
#         test_condition_f(nlp, solver, p, Π, k)
#         test_condition_h(nlp, solver, p, Π, k)
#         test_condition_∇f(nlp, solver, p, Π, k)
#       end

#       # update precision levels: 
#       max_prec_k = max(p.pf, p.pg, p.ph, p.ps)
#       p.pf, p.pg, p.ph, p.ps = max_prec_k, max_prec_k, max_prec_k, max_prec_k

#       mks = mk(solver.sk[p.ps]) 
#       ξ = solver.hk[p.ph] - mks + max(1, abs(solver.hk[p.ph])) * 10 * eps() # en la precision de eps() #TODO : check which one is it

#       if p.activate_mp
#         ξ = test_assumption_6(nlp, solver, options, p, Π, k, ξ)
#       end
#       max_prec_k = max(p.pf, p.pg, p.ph, p.ps)
#       p.pf, p.pg, p.ph, p.ps = max_prec_k, max_prec_k, max_prec_k, max_prec_k

#       #-------------------------------------------------------------------------------------------
#       # -- à partir de là, toutes les conditions de convergence sont garanties à l'itération k. --
#       #-------------------------------------------------------------------------------------------

#       sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)

#       if ξ ≥ 0 && k == 1
#         ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative
#       end
#       if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ * sqrt(p.κξ))
#         if k > 1 # add this to avoid the case where the first iteration is optimal because of float16. 
#           optimal = true
#           continue
#         end
#       end

#       if (ξ < 0 && sqrt_ξ_νInv > 1e4*neg_tol) #TODO change this 
#         status = :exception 
#         @info @sprintf "%6d %8.1e %8.1e %8s %8s %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] "" "" p.σk norm(solver.xk[end]) norm(solver.xk[end]) "" Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
#         @warn "R2: prox-gradient step should produce a decrease but ξ = $(ξ). Early stopping iR2-Reg."
#         return k, status, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, [p.pf, p.pg, p.ph, p.ps]
#       end

#       solver.xkn .= solver.xk[end] .+ solver.sk[end] # choix de le mettre en la précision la + haute car utilisé pour calculer ρk
#       fkn = f(solver.xkn)
#       solver.special_counters[:f][end] += 1
#       hkn = @views reg_nlp.h(solver.xkn[selected])
#       solver.special_counters[:h][p.ph] += 1
#       hkn == -Inf && error("nonsmooth term is not proper")

#       Δobj = (solver.fk[end] + solver.hk[end]) - (fkn + hkn) + max(1, abs(solver.fk[end] + solver.hk[end])) * 10 * eps() #TODO change eps() to eps(Π[p]), but which one? 
#       ρk = Δobj / ξ # En la précision la + haute des 2

#       if (verbose > 0) && (k % ptf == 0)
#         #! format: off
#         σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")
#         @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] sqrt_ξ_νInv ρk p.σk norm(solver.xk[end]) norm(solver.xk[end]) σ_stat Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
#         #! format: on
#       end

#       if η2 ≤ ρk < Inf
#         p.σk = max(p.σk / γ, σmin)
#       end

#       if η1 ≤ ρk < Inf
#         for i=1:P
#           solver.xk[i] .= solver.xkn # xkn casté en chacune des précisions de Π puis assigné à xk[i]
#         end

#         if has_bnds #TODO maybe change this
#           @. solver.l_bound_m_x = solver.l_bound - solver.xk[1]
#           @. solver.u_bound_m_x = solver.u_bound - solver.xk[1]
#           set_bounds!(solver.ψ, l_bound_m_x, u_bound_m_x)
#         end
      
#         for i=1:P
#           solver.fk[i] = Π[i](fkn)
#           solver.hk[i] = Π[i](hkn)
#         end

#         ∇f!(solver.gfk[p.pg], solver.xk[p.pg])
#         solver.special_counters[:∇f][p.pg] += 1
#         for i=1:P 
#           solver.gfk[i] .= solver.gfk[p.pg]
#         end

#         shift!(solver.ψ, solver.xk[p.ph])
#       end

#       if ρk < η1 || ρk == Inf
#         p.σk = p.σk * γ
#       end

#       p.ν = 1 / p.σk # !!! Under/Overflow possible here.
#       tired = max_iter > 0 && k ≥ max_iter
#       if !tired
#         for i=1:P
#           solver.mν∇fk[i] .= -Π[end].(p.ν) * solver.gfk[i]
#         end
#       end
#     end
#     return k, optimal, tired, elapsed_time, sqrt_ξ_νInv
#   end # end of inner_loop! function

#   k, optimal, tired, elapsed_time, sqrt_ξ_νInv = inner_loop!(solver, p, Π, k, verbose, max_iter, maxTime, σmin, η1, η2, γ)

#   if verbose > 0
#     if k == 1
#       @info @sprintf "%6d %8.1e %8.1e" k solver.fk[p.pf] solver.hk[p.ph]
#     elseif optimal
#       #! format: off
#       @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] sqrt(ξ/p.ν) "" p.σk norm(solver.xk[end]) norm(solver.xk[end]) "" Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
#       @info "R2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
#     end
#   end

#   status = if optimal
#     :first_order
#   elseif elapsed_time > maxTime
#     :max_time
#   elseif tired
#     :max_iter
#   else
#     :exception
#   end
#   return k, status, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, [p.pf, p.pg, p.ph, p.ps]
# end



function get_status(
  reg_nlp;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  improper = false,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
)
  if optimal
    :first_order
  elseif improper
    :improper
  elseif iter > max_iter
    :max_iter
  elseif elapsed_time > max_time
    :max_time
  elseif neval_obj(reg_nlp.model) > max_eval && max_eval != -1
    :max_eval
  else
    :unknown
  end
end