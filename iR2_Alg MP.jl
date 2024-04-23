
mutable struct MPR2Solver{R, S} <: AbstractOptimizationSolver
  xk # conteneur des x en les 3 précisions #TODO : tout typer
  ∇fk
  mν∇fk
  gfk
  fk
  hk
  sk
  xkn::S
  s::S
  has_bnds::Bool
  l_bound::S
  u_bound::S
  l_bound_m_x::S
  u_bound_m_x::S
  Fobj_hist::Vector{R}
  Hobj_hist::Vector{R}
  Complex_hist::Vector{Int}
  p_hist::Vector{Vector{Int64}}
  h
  ψ
end

mutable struct iR2RegParams
  pf::Int
  pg::Int
  ph::Int
  ps::Int
  Π::Vector{DataType}
  verb::Bool
  activate_mp::Bool
  flags::Vector{Bool}
  κf
  κh
  κ∇
  κs
  σk
  ν
end

function iR2RegParams(Π::Vector{DataType}; verb::Bool=false, activate_mp::Bool=true, flags::Vector{Bool}=[false, false, false], κf=1e-5, κh=2e-5, κ∇=4e-2, κs=1., σk=1., ν=0.000740095979741405)
  return iR2RegParams(1, 1, 1, 1, Π, verb, activate_mp, flags, κf, κh, κ∇, κs, σk, ν)
end

function MPR2Solver(
  x0::S,
  options::ROSolverOptions,
  params::iR2RegParams,
  l_bound::S,
  u_bound::S;
) where {R <: Real, S <: AbstractVector{R}}
  maxIter = options.maxIter
  Π = params.Π
  xk = [Vector{R}(undef, length(x0)) for R in Π]
  ∇fk = similar(x0)
  gfk = [Vector{R}(undef, length(x0)) for R in Π]
  fk = [zero(R) for R in Π]
  hk = [zero(R) for R in Π]
  sk = [Vector{R}(undef, length(x0)) for R in Π]
  mν∇fk = [Vector{R}(undef, length(x0)) for R in Π]
  xkn = similar(x0)
  s = zero(x0)
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
  h = nothing
  ψ = nothing
  return MPR2Solver(
    xk,
    ∇fk,
    mν∇fk,
    gfk,
    fk, 
    hk, 
    sk,
    xkn,
    s,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    Fobj_hist,
    Hobj_hist,
    Complex_hist,
    p_hist,
    h, 
    ψ
  )
end


"""
    MPR2(nlp, h, options)
    MPR2(f, ∇f!, h, options, x0)

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

function MPR2(
  nlp::AbstractNLPModel, 
  args...; 
  kwargs...
)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
    x, k, outdict = MPR2(
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
  
  return stats
end

# method without bounds
function MPR2(
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
  solver = MPR2Solver(x0, options, params, similar(x0, 0), similar(x0, 0))
  k, status, fk, hk, ξ = MPR2!(solver, f, ∇f!, h, options, x0; selected=selected, params=params,)
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
  )
  
  return solver.xk[end], k, outdict
end

function MPR2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  params::iR2RegParams,
  x0::AbstractVector{S},
  l_bound::AbstractVector{S},
  u_bound::AbstractVector{S};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...
) where {F <: Function, G <: Function, H, R <: Real, S <: Real}
  start_time = time()
  elapsed_time = 0.0
  solver = MPR2Solver(x0, options, params, l_bound, u_bound,)
  k, status, fk, hk, ξ = MPR2!(solver, f, ∇f!, h, options, params, x0; selected=selected)
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
  )

  return solver.xk[end], k, outdict
end
  


function MPR2!(
  solver::MPR2Solver{R, S},
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
    test_κ(p.κs, p.κf, p.κ∇, p.κh, η1, η2)
  end

  params.pf, params.pg, params.ph, params.ps = 1, 1, 1, 1

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
  if hxk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(solver.xk[p.ph][selected], solver.h, x0, one(eltype(x0)))
    hxk = @views solver.h(solver.xk[p.ph][selected])
    hxk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
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
  p.ν = 1 / p.σk # !!! Under/Overflow possible here.
  sqrt_ξ_νInv = one(R)

  fxk = f(solver.xk[p.pf])
  for i=1:P
    solver.fk[i] = Π[i].(fxk)
  end

  ∇f!(solver.gfk[p.pg], solver.xk[p.pg])
  for i=1:P 
    solver.gfk[i] .= solver.gfk[p.pg]
  end
  # peut-être moyen de fusionner les deux boucles.
  for i=1:P
    solver.mν∇fk[i] .= -Π[i].(p.ν) * solver.gfk[i]
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
    if has_bnds #TODO maybe change this 
      @. l_bound_m_x = l_bound - solver.xk[1]
      @. u_bound_m_x = u_bound - solver.xk[1]
      ψ = shifted(solver.h, solver.xk[1], l_bound_m_x, u_bound_m_x, selected)
    else
      solver.h = NormL1(Π[p.ps](1.0)) # need to redefine h at each iteration because when shifting: MethodError: no method matching shifted(::NormL1{Float64}, ::Vector{Float16}) so the norm and the shift vector must be same FP Format. 
      ψ = shifted(solver.h, solver.xk[p.ps])
    end
    φk(d) = dot(solver.gfk[end], d) # FP format : highest available to avoid over/underflow
    mk(d) = φk(d) + ψ(d) 

    prox!(solver.sk[p.ps], ψ, solver.mν∇fk[p.ps], Π[p.ps].(p.ν)) # # on calcule le prox en la précision de ps. 

    Complex_hist[k] += 1
    for i=1:P
      solver.sk[i] .= solver.sk[p.ps]
    end

    if p.activate_mp 
      test_condition_f(nlp, solver, p, Π, k)
      test_condition_h(nlp, solver, p, Π, k)
      test_condition_∇f(nlp, solver, p, Π, k)
    end

    mks = mk(solver.sk[p.ps]) # casté en la précision la + haute entre celle de mk et ps
    ξ = solver.hk[p.ph] - mks + max(1, abs(solver.hk[p.ph])) * 10 * eps() # casté en la précision la plus haute entre ph et ps. 
    if p.activate_mp
      ξ = test_assumption_6(nlp, solver, p, Π, k, ξ)
    end

    #-------------------------------------------------------------------------------------------
    # -- à partir de là, toutes les conditions de convergence sont garanties à l'itération k. --
    #-------------------------------------------------------------------------------------------

    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)

    if ξ ≥ 0 && k == 1
      ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative
    end
    
    if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ)
      if k > 1 # add this to avoid the case where the first iteration is optimal because of float16. 
        optimal = true
        continue
      end
    end

    if (ξ < 0 && sqrt_ξ_νInv > 1e4*neg_tol)
      status = :exception 
      @info @sprintf "%6d %8.1e %8.1e %8s %8s %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k solver.fk[p.pf] solver.hk[p.ph] "" "" p.σk norm(solver.xk[end]) norm(solver.xk[end]) "" Π[p.pf] Π[p.pg] Π[p.ph] Π[p.ps]
      @warn "R2: prox-gradient step should produce a decrease but ξ = $(ξ). Early stopping iR2-Reg."
      return k, status, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, [p.pf, p.pg, p.ph, p.ps]
    end

    solver.xkn .= solver.xk[end] .+ solver.sk[end] # choix de le mettre en la précision la + haute, mais on pourrait le mettre en la précision de ps
    fkn = f(solver.xkn)
    hkn = @views solver.h(solver.xkn[selected]) # fkn et hkn en la précision de xkn = celle de ps 
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
      for i=1:P # on met à jour le conteneur
        solver.xk[i] .= solver.xkn # !! vérifier que chaque élément a un FP format différent
      end

      if has_bnds #TODO maybe change this
        @. l_bound_m_x = l_bound - xk[1]
        @. u_bound_m_x = u_bound - xk[1]
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end
      
      for i=1:P
        solver.fk[i] = Π[i].(fkn)
      end

      for i=1:P
        solver.hk[i] = Π[i].(hkn)
      end

      ∇f!(solver.gfk[p.pg], solver.xk[p.pg])
      for i=1:P 
        solver.gfk[i] .= solver.gfk[p.pg]
      end

      shift!(ψ, solver.xk[p.ph])
    end

    if ρk < η1 || ρk == Inf
      p.σk = p.σk * γ
    end

    p.ν = 1 / p.σk # !!! Under/Overflow possible here.
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      for i=1:P
        solver.mν∇fk[i] .= -Π[i].(p.ν) * solver.gfk[i]
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
