
mutable struct MPR2Solver{R, S} <: AbstractOptimizationSolver
  x :: S # variable du point courant
  xk # conteneur des x en les 3 précisions #TODO : tout typer
  ∇fk::S
  mν∇fk::S
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
end

function MPR2Solver(
  x0::S,
  options::ROSolverOptions,
  l_bound::S,
  u_bound::S,
  Π::Vector{DataType};# Ajout de Π comme argument du constructeur
) where {R <: Real, S <: AbstractVector{R}}
  maxIter = options.maxIter
  x = similar(x0)
  xk =  [Vector{R}(undef, length(x0)) for R in Π]
  ∇fk = similar(x0)
  gfk = [Vector{R}(undef, length(x0)) for R in Π]
  fk = [zero(R) for R in Π]
  hk = [zero(R) for R in Π]
  sk = [Vector{R}(undef, length(x0)) for R in Π]
  mν∇fk = similar(x0)
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
  return MPR2Solver(
    x,
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
  Π::Vector{DataType} = [Float64], 
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
    Π=Π,
    kwargs_dict...
  )
  
  ξ = outdict[:ξ]
  stats = GenericExecutionStats(nlp)
  set_status!(stats, outdict[:status])
  set_solution!(stats, x)
  set_objective!(stats, outdict[:fk] + outdict[:hk])
  set_residuals!(stats, zero(eltype(x)), ξ)
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  set_solver_specific!(stats, :p_hist, outdict[:p_hist])
  
  return stats
end

function MPR2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{S};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  verb::Bool = false,
  Π::Vector{DataType} = [Float64], 
  activate_mp::Bool = true,
  kwargs...
) where {F <: Function, G <: Function, H, R <: Real, S <: Real}
  start_time = time()
  elapsed_time = 0.0
  solver = MPR2Solver(x0, options, similar(x0, 0), similar(x0, 0), Π)
  k, status, fk, hk, ξ = MPR2!(solver, f, ∇f!, h, options, x0; selected=selected, verb=verb, activate_mp=activate_mp, Π=Π)
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
  
  return solver.x, k, outdict
end

function MPR2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{S},
  l_bound::AbstractVector{S},
  u_bound::AbstractVector{S};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  verb::Bool = false, 
  Π::Vector{DataType} = [Float16, Float32, Float64],
  activate_mp::Bool = false,
  kwargs...
) where {F <: Function, G <: Function, H, R <: Real, S <: Real}
  start_time = time()
  elapsed_time = 0.0
  solver = MPR2Solver(x0, options, l_bound, u_bound, Π)  # Ajout de Π comme argument
  k, status, fk, hk, ξ = MPR2!(solver, f, ∇f!, h, options, x0; selected=selected, verb=verb, activate_mp=activate_mp, Π=Π)
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

  return solver.x, k, outdict
end
  


function MPR2!(
  solver::MPR2Solver{R, S},
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions,
  x0::S;
  selected::AbstractVector{<:Integer} = 1:length(x0),
  verb::Bool = false,
  activate_mp::Bool = false,
  Π::Vector{DataType} = [Float64]
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
  ν = options.ν
  γ = options.γ



  κf = 1e-5 # respectent les conditions du papier
  κ∇ = 4e-2
  κh = 2e-5
  κs = 1.
  # κξ = 1. inutile

  if activate_mp
    test_κ(κs, κf, κ∇, κh, η1, η2) # pour respecter les conditions de positivité (cf analyse de convergence)
  end


  # initializing levels of precision for our inexact functions. 
  pf, pg, ph, ps, px = 1, 1, 1, 1, 1
  P = length(Π)

  # retrieve workspace
  x = solver.x
  x .= x0
  xk = solver.xk
  ∇fk = solver.∇fk
  gfk = solver.gfk 
  fk = solver.fk 
  hk = solver.hk 
  sk = solver.sk
  mν∇fk = solver.mν∇fk
  xkn = solver.xkn
  s = solver.s
  xk[px] .= x0 # on met à jour le conteneur
  println("x0 = ", x0)
  println("x = ", x)
  println("xk = ", xk)

  #TODO 15-04 : modifier xk car j'appelle xk partout ici, mais maintenant c'est un conteneur 

  # à l'initialisation, tous les vecteurs sont en Float16. 
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

  # initialize parameters
  hxk = @views h(x[selected])
  if hxk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(x, h, x0, one(eltype(x0)))
    hxk = @views h(x[selected])
    hxk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  hxk == -Inf && error("nonsmooth term is not proper")

  hk[ph] = hxk # on met à jour le conteneur 

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s %6s %6s %6s %6s" "iter" "f(x)" "h(x)" "√(ξ/ν)" "ρ" "σ" "‖x‖" "‖s‖" "" "πf" "πg" "πh" "πs"
    #! format: off
  end

  local ξ::R
  k = 0
  σk = max(1 / ν, σmin) # !! toujours en Float64
  ν = 1 / σk
  sqrt_ξ_νInv = one(R)

  fxk = f(x) # en Float16 à l'initialisation
  fk[pf] = fxk # on met à jour le conteneur

  ∇f!(∇fk, x) # En float 16 initialement car en la précison de xk
  gfk[pg] .= ∇fk # on met à jour le conteneur

  @. mν∇fk = -ν * gfk[pg] # En Float16 initialement

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  flags=[false, false, false] # pour contrôler l'affichage dans les fonctions de test de précision.

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = Π[end].(fk[pf]) # cast en Float64 pour l'affichage
    Hobj_hist[k] = Π[end].(hk[ph]) # cast en Float64 pour l'affichage
    p_hist[k] = [pf, pg, ph, ps]

    # define model
    if has_bnds
      @. l_bound_m_x = l_bound - x
      @. u_bound_m_x = u_bound - x
      ψ = shifted(h, x, l_bound_m_x, u_bound_m_x, selected)
    else
      ψ = shifted(h, x) # en Float16 à l'initialisation, puis en celle de x
    end
    φk(d) = dot(gfk[pg], d) # En Float16 initialement, puis dans celle du gradient. 
    mk(d)::R = φk(d) + ψ(d)::R # !! à la précision de ψ. On met le calcul de ψ dans le while pour que sa précision reste celle de x pour toutes les itérations. 

    (activate_mp == true) && (ν = Π[pg](ν)) # on met à jour la précision de ν en celle du gradient pour que le calcul du prox passe. 
    prox!(s, ψ, mν∇fk, ν)

    Complex_hist[k] += 1

    sk[ps] .= s # on met à jour le conteneur

    # if activate_mp # changer cela pour faire de la vraie MP = recalculer les variables en précision supérieure plutôt que de les caster.
    #   pf, ps, flags = test_condition_f(fk, sk, σk, κf, pf, ps, Π, k, verb, flags)
    #   ph, ps, flags = test_condition_h(hk, sk, σk, κh, ph, ps, Π, k, verb, flags)
    #   pg, ps, flags = test_condition_∇f(gfk, sk, σk, κ∇, pg, ps, Π, k, verb, flags)
    # end

    mks = mk(sk[ps]) 
    ξ = hk[ph] - mks + max(1, abs(hk[ph])) * 10 * eps() # casté en la précision la plus haute entre ph et ps. 

    if activate_mp
      ps, ph, ξ, flags = test_assumption_6(ξ, κs, σk, sk, Π, ps, ph, mk, hk, k, verb, flags)
    end
    #---------------------------------------------------------------
    # à partir de là, toutes les conditions de convergence sont garanties à l'itération k.
    #---------------------------------------------------------------

    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν) # en Float64 car ξ en Float64

    if ξ ≥ 0 && k == 1
      ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative
    end
    
    if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ)
      optimal = true
      continue
    end

    if (ξ < 0 && sqrt_ξ_νInv > neg_tol) 
      error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")
    end

    xkn .= x .+ sk[ps] # casté en la précision la plus haute entre x et ps. Est-ce la même ? --> sûrement après avoir implémenté correctement les tests de précision.
    fkn = f(xkn)
    hkn = @views h(xkn[selected]) # fkn et hkn en la précision de xkn
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = (fk[pf] + hk[ph]) - (fkn + hkn) + max(1, abs(fk[pf] + hk[ph])) * 10 * eps() #TODO change eps() to eps(Π[p]), but which one? 
    ρk = Δobj / ξ # En la précision la + haute des 2

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k fk[pf] hk[ph] sqrt_ξ_νInv ρk σk norm(x) norm(sk[ps]) σ_stat Π[pf] Π[pg] Π[ph] Π[ps]
      #! format: on
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      x .= xkn
      if has_bnds
        @. l_bound_m_x = l_bound - x
        @. u_bound_m_x = u_bound - x
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end

      fk[pf] = fkn
      hk[ph] = hkn

      ∇f!(∇fk, x) #TODO changer cela pour calculer le gradient en la précision courante. Pour l'instant, en la précision de x = tout le temps FLoat64. 
      gfk[pg] .= ∇fk # on met à jour le conteneur

      shift!(ψ, x)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      @. mν∇fk = -ν * gfk[pg]
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk[pf] hk[ph]
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e %1s %6s %6s %6s %6s" k fk[pf] hk[ph] sqrt(ξ/ν) "" σk norm(x) norm(s) "" Π[pf] Π[pg] Π[ph] Π[ps]
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
  return k, status, Float64(fk[pf]), Float64(hk[ph]), sqrt_ξ_νInv, [pf, pg, ph, ps]
end
