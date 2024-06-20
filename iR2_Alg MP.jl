"""
mutable struct iR2Solver{R<:Real, S<:AbstractVector} <: AbstractOptimizationSolver

    xk::Vector{S}
      Contains the current iterate in the floating point formats specified in Π (see iR2RegParams). 
      For example, if Π = [Float16, Float32, Float64], then xk = [xk16, xk32, xk64] where xk16, xk32, xk64 are the iterates in Float16, Float32 and Float64 respectively.

    mν∇fk::Vector{S}
      Contains the negative gradient of f in the proximal step in the floating point formats specified in Π

    gfk::Vector{S}
      Contains the gradient of f in the floating point formats specified in Π

    fk::S
      Contains the value of f in the current floating point formats of the smooth term

    hk::S
      Contains the value of h in the floating point format of the nonsmooth term

    sk::Vector{S}
      Contains the step in the proximal step in the floating point formats specified in Π

    xkn::S
      Contains the next iterate in the floating point format of ??? (#TODO: check this)

    has_bnds::Bool
      Indicates whether the problem has bounds

    l_bound::S
      Lower bounds of the problem

    u_bound::S
      Upper bounds of the problem

    l_bound_m_x::S
      Contains the update of the lower bounds shifted to the current iterate xk

    u_bound_m_x::S
      Contains the update of the upper bounds shifted to the current iterate xk

    Fobj_hist::Vector{R}
      Contains the history of the values of the smooth term

    Hobj_hist::Vector{R}
      Contains the history of the values of the nonsmooth term

    Complex_hist::Vector{Int}
      Contains the history of number of inner iterations (iterations for solving the prox subproblem). 

    p_hist::Vector{Vector{Int}}
      Contains the history of the precision levels used at each iteration

    special_counters::Dict{Symbol, Vector{Int}}
      Contains the history of the number of times the following evaluations were performed at each precision level specified in Π:
        - f
        - h
        - ∇f
        - prox

    h
      Function that computes the value of the nonsmooth term

    ψ
      Model of the the nonsmooth term at xk + s

"""
mutable struct iR2Solver{R<:Real, S<:AbstractVector} <: AbstractOptimizationSolver
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
  h
  ψ
end


"""
mutable struct iR2RegParams

    pf::Int
      current precision level of the smooth term

    pg::Int
      current precision level of the gradient of the smooth term

    ph::Int
      current precision level of the nonsmooth term

    ps::Int
      current precision level of the step

    Π::Vector{DataType}
      Contains the floating point formats used in the algorithm. For example, if Π = [Float16, Float32, Float64], then the algorithm will use Float16, Float32 and Float64.

    verb::Bool
      whether to display general information about the algorithm

    activate_mp::Bool
      whether to activate the mixed precision mode. If true, the algorithm will test the conditions of the mixed precision mode.

    flags::Vector{Bool}
      Contains some flags of the conditions of the mixed precision mode. If a flag is true, it means that the maximum precision level has been reached for the corresponding variable among : 
        - f
        - h
        - ∇f
        - s

    κf::H
      accuracy parameter for the smooth term (see paper for details). The accuracy condition on the smooth term is: 
        abs(fk * (1 - 1/(1 + eps)) ≤ κf * σk * ‖sk‖²
      H is the "Highest" floating point format in Π. #TODO : check this. 

    κh::H
      accuracy parameter for the nonsmooth term (see paper for details). The accuracy condition on the nonsmooth term is: 
        abs(hk * (1 - 1/(1 + eps)) ≤ κh * σk * ‖sk‖²

    κ∇::H
      accuracy parameter for the gradient of the smooth term (see paper for details). The accuracy condition on the gradient of the smooth term is: 
        abs(dot(gfk, sk) * (1 - 1/(1 + eps)) ≤ κ∇ * σk * ‖sk‖

    κs::H
      accuracy parameter for the step (see paper for details). The condition on the step is: 
        ξ ≥ 1/2 * κs* σk * ‖sk‖²

    κξ::H
      accuracy parameter for ξ (see paper for details). Involved in the stopping criterion of the algorithm : 
        sqrt(ξk * σk) ≤ ϵ * sqrt(κξ)
        

    σk::H
      regularization parameter. The algorithm uses the regularization parameter σk = 1/ν

    ν::H
      precision parameter. The algorithm uses the precision parameter ν = 1/σk

"""
mutable struct iR2RegParams{H<:Real}
  pf::Int
  pg::Int
  ph::Int
  ps::Int
  Π::Vector{DataType}
  verb::Bool
  activate_mp::Bool
  flags::Vector{Bool}
  κf::H
  κh::H
  κ∇::H
  κs::H
  κξ::H
  σk::H
  ν::H
end

function iR2RegParams(Π::Vector{DataType}; pf = 1, pg = 1, ph = 1, ps = 1, verb::Bool=false, activate_mp::Bool=true, flags::Vector{Bool}=[false, false, false], κf=1e-5, κh=2e-5, κ∇=4e-2, κs=1., κξ=1e-4, σk=1., ν=eps(Π[end])^(1/5))
  return iR2RegParams(pf, pg, ph, ps, Π, verb, activate_mp, flags, κf, κh, κ∇, κs, κξ, σk, ν)
end

function iR2Solver(
  x0::S,
  options::ROSolverOptions,
  params::iR2RegParams,
  l_bound::S,
  u_bound::S;
) where {S <: AbstractVector}
  maxIter = options.maxIter
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
  Fobj_hist = zeros(R, maxIter)
  Hobj_hist = zeros(R, maxIter)
  Complex_hist = zeros(Int, maxIter)
  p_hist = [zeros(Int, 4) for _ in 1:maxIter]
  special_counters = Dict(:f => zeros(Int, length(Π)), :h => zeros(Int, length(Π)), :∇f => zeros(Int, length(Π)), :prox => zeros(Int, length(Π)))
  h = nothing
  ψ = nothing
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
    h, 
    ψ
  )
end



"""
    iR2(nlp, h, options)
    iR2(f, ∇f!, h, options, x0)

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

function iR2(
  nlp::AbstractNLPModel, 
  args...; 
  kwargs...
)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  x, k, outdict = iR2(
  t -> obj(nlp, t),
  (g, t) -> grad!(nlp, t, g),
  args...,
  x0,
  nlp.meta.lvar,
  nlp.meta.uvar;
  kwargs_dict...
  )
  
  ξ = outdict[:ξ]
  stats = GenericExecutionStats(nlp)
  stats.solution = eltype(x).(stats.solution)
  stats.objective = eltype(x).(stats.objective)
  stats.primal_feas = eltype(x).(stats.primal_feas)
  stats.dual_feas = eltype(x).(stats.dual_feas)
  set_status!(stats, outdict[:status])
  set_solution!(stats, x)
  set_objective!(stats, typeof(nlp.meta.x0[1]).(outdict[:fk]) + typeof(nlp.meta.x0[1]).(outdict[:hk])) #TODO : change this
  set_residuals!(stats, typeof(nlp.meta.x0[1]).(zero(eltype(x))), typeof(nlp.meta.x0[1]).(ξ))
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  set_solver_specific!(stats, :p_hist, outdict[:p_hist])
  set_solver_specific!(stats, :special_counters, outdict[:special_counters])
  
  return stats
end

# method without bounds
function iR2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  params::iR2RegParams,
  x0::AbstractVector{S};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...
) where {F <: Function, G <: Function, H, R <: Real, S <: Real}
  start_time = time()
  elapsed_time = 0.0
  solver = iR2Solver(x0, options, params, similar(x0, 0), similar(x0, 0))
  k, status, fk, hk, ξ = iR2!(solver, f, ∇f!, h, options, x0; selected=selected, params=params,)
  elapsed_time = time() - start_time
  
  outdict = Dict(
    :Fhist => solver.Fobj_hist[1:k],
    :Hhist => solver.Hobj_hist[1:k],
    :Chist => solver.Complex_hist[1:k],
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
    :p_hist => solver.p_hist[1:k],
    :special_counters => solver.special_counters
  )
  
  return solver.xk[end], k, outdict
end

function iR2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  params::iR2RegParams, #TODO : commentaire DO : "autodocumentation" ? 
  x0::AbstractVector{S},
  l_bound::AbstractVector{S},
  u_bound::AbstractVector{S};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...
) where {F <: Function, G <: Function, H, R <: Real, S <: Real}
  start_time = time()
  elapsed_time = 0.0
  solver = iR2Solver(x0, options, params, l_bound, u_bound,)
  k, status, fk, hk, ξ = iR2!(solver, f, ∇f!, h, options, params, x0; selected=selected)
  elapsed_time = time() - start_time
  outdict = Dict(
    :Fhist => solver.Fobj_hist[1:k],
    :Hhist => solver.Hobj_hist[1:k],
    :Chist => solver.Complex_hist[1:k],
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
    :p_hist => solver.p_hist[1:k],
    :special_counters => solver.special_counters
  )

  return solver.xk[end], k, outdict
end

function iR2!(
  solver::iR2Solver{R, S},
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions,
  p::iR2RegParams,
  x0::S;
  selected::AbstractVector{<:Integer} = 1:length(x0),
) where {F <: Function, G <: Function, H, R <: Real, S}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵr = options.ϵr
  neg_tol = options.neg_tol
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  σmin = options.σmin
  η1 = options.η1
  η2 = options.η2
  p.ν = options.ν
  γ = options.γ

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
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  solver.h = h
  # initialize parameters
  hxk = @views solver.h(solver.xk[p.ph][selected])
  solver.special_counters[:h][p.ph] += 1
  if hxk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(solver.xk[p.ph][selected], solver.h, x0, one(eltype(x0)))
    hxk = @views solver.h(solver.xk[p.ph][selected])
    if hxk == Inf
      status = :exception 
      @warn "prox computation must be erroneous. Early stopping iR2-Reg."
      return 1, status, solver.fk[end], solver.hk[end], Inf, [p.pf, p.pg, p.ph, p.ps]
    end
  end
  hxk == -Inf && error("nonsmooth term is not proper")

  for i=1:P 
    solver.hk[i] = Π[i].(hxk)
  end

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s %6s %6s %6s %6s" "iter" "f(x)" "h(x)" "√(ξ/ν)" "ρ" "σ" "‖x‖" "‖s‖" "" "πf" "πg" "πh" "πs"
    #! format: off
  end

  local ξ
  k = 0
  p.σk = max(1 / p.ν, σmin)
  p.ν = 1 / p.σk
  sqrt_ξ_νInv = one(R)

  fxk = f(solver.xk[p.pf]) # ici, on calcule f(xk) en la précision pf (la + basse initialement).
  solver.special_counters[:f][p.pf] += 1 # on incrémente le compteur de f en la précision pf.
  for i=1:P
    solver.fk[i] = Π[i](fxk) # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
  end # Q: est-ce que on recalcule quelquechose en faisant Float16(fxk) sachant que fxk est déjà en float16 ?

  ∇f!(solver.gfk[p.pg], solver.xk[p.pg])
  solver.special_counters[:∇f][p.pg] += 1
  for i=1:P 
    solver.gfk[i] .= solver.gfk[p.pg]
    solver.mν∇fk[i] .= -Π[end].(p.ν) * solver.gfk[i]
  end

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time

    Fobj_hist[k] = solver.fk[p.pf] 
    Hobj_hist[k] = solver.hk[p.ph] 
    p_hist[k] = [p.pf, p.pg, p.ph, p.ps]

    # define model
    if has_bnds #TODO updatde this later 
      @. l_bound_m_x = l_bound - solver.xk[1]
      @. u_bound_m_x = u_bound - solver.xk[1]
      solver.ψ = shifted(solver.h, solver.xk[1], l_bound_m_x, u_bound_m_x, selected)
    else
      solver.h = NormL1(Π[p.ps](1.0)) # need to redefine h at each iteration because when shifting: MethodError: no method matching shifted(::NormL1{Float64}, ::Vector{Float16}) so the norm and the shift vector must be same FP Format. 
      solver.ψ = shifted(solver.h, solver.xk[p.ps]) # therefore ψ FP format is s FP format
    end
    φk(d) = dot(solver.gfk[end], d) # FP format : highest available to avoid over/underflow
    mk(d) = φk(d) + solver.ψ(d)

    prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps].(p.ν)) # on calcule le prox en FP format de s car tous les arguments de prox!() doivent être de même FP format.
    solver.special_counters[:prox][p.ps] += 1

    Complex_hist[k] += 1
    for i=1:P
      solver.sk[i] .= solver.sk[p.ps] # on a mis a jour solver.sk[p.ps] dans prox!() donc c'est celui qu'on met à jour. 
    end

    if p.activate_mp 
      test_condition_f(nlp, solver, p, Π, k)
      test_condition_h(nlp, solver, p, Π, k)
      test_condition_∇f(nlp, solver, p, Π, k)
    end
    mks = mk(solver.sk[p.ps]) # casté en la précision la + haute entre celle de mk et ps
    ξ = solver.hk[p.ph] - mks + max(1, abs(solver.hk[p.ph])) * 10 * eps() # casté en la précision la plus haute entre ph et ps. 
    if p.activate_mp
      ξ = test_assumption_6(nlp, solver, options, p, Π, k, ξ)
    end

    #-------------------------------------------------------------------------------------------
    # -- à partir de là, toutes les conditions de convergence sont garanties à l'itération k. --
    #-------------------------------------------------------------------------------------------

    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)

    if ξ ≥ 0 && k == 1
      ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative

    end
    if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ * sqrt(p.κξ))
      if k > 1 # add this to avoid the case where the first iteration is optimal because of float16. 
        optimal = true
        continue
      end
    end

    if (ξ < 0 && sqrt_ξ_νInv > 1e4*neg_tol) #TODO change this 
      status = :exception 
      @info @sprintf "%6d %8.1e %8.1e %8s %8s %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] "" "" p.σk norm(solver.xk[end]) norm(solver.xk[end]) "" Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
      @warn "R2: prox-gradient step should produce a decrease but ξ = $(ξ). Early stopping iR2-Reg."
      return k, status, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, [p.pf, p.pg, p.ph, p.ps]
    end

    solver.xkn .= solver.xk[end] .+ solver.sk[end] # choix de le mettre en la précision la + haute
    fkn = f(solver.xkn)
    solver.special_counters[:f][end] += 1
    hkn = @views solver.h(solver.xkn[selected]) # fkn et hkn en la précision de xkn = celle de ps 
    solver.special_counters[:h][p.ph] += 1
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = (solver.fk[end] + solver.hk[end]) - (fkn + hkn) + max(1, abs(solver.fk[end] + solver.hk[end])) * 10 * eps() #TODO change eps() to eps(Π[p]), but which one? 
    ρk = Δobj / ξ # En la précision la + haute des 2

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] sqrt_ξ_νInv ρk p.σk norm(solver.xk[end]) norm(solver.xk[end]) σ_stat Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
      #! format: on
    end

    if η2 ≤ ρk < Inf
      p.σk = max(p.σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      for i=1:P
        solver.xk[i] .= solver.xkn
      end

      if has_bnds #TODO maybe change this
        @. l_bound_m_x = l_bound - xk[1]
        @. u_bound_m_x = u_bound - xk[1]
        set_bounds!(solver.ψ, l_bound_m_x, u_bound_m_x)
      end
      
      for i=1:P
        solver.fk[i] = Π[i](fkn)
        solver.hk[i] = Π[i](hkn)
      end

      ∇f!(solver.gfk[p.pg], solver.xk[p.pg])
      solver.special_counters[:∇f][p.pg] += 1
      for i=1:P 
        solver.gfk[i] .= solver.gfk[p.pg]
      end

      shift!(solver.ψ, solver.xk[p.ph])
    end

    if ρk < η1 || ρk == Inf
      p.σk = p.σk * γ
    end

    p.ν = 1 / p.σk # !!! Under/Overflow possible here.
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      for i=1:P
        solver.mν∇fk[i] .= -Π[end].(p.ν) * solver.gfk[i]
      end
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k solver.fk[p.pf] solver.hk[p.ph]
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] sqrt(ξ/p.ν) "" p.σk norm(solver.xk[end]) norm(solver.xk[end]) "" Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
      @info "R2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
    end
  end

  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end
  return k, status, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, [p.pf, p.pg, p.ph, p.ps]
end