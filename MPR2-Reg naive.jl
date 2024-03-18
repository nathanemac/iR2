using LinearAlgebra
using Distributions
using NLPModels
using ADNLPModels
using ShiftedProximalOperators
using RegularizedOptimization
using ProximalOperators
using Printf
using SolverCore

function noise(xk; ϵ=1e-3)
  return zero(xk) .+ rand(Uniform(-ϵ, ϵ))
end

options = ROSolverOptions(maxIter=500)

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1., 2.])
nlp = ADNLPModel(x -> x[1]^2 + x[2]^2, [-10., 2.9])
h = NormL1(1.0)


function naive_R2(nlp::ADNLPModel,
                  h; 
                  η1 = 0.02,
                  η2 = 0.95,
                  γ = 3.,
                  ν = 1e-3,
                  crit = 1e-3,
                  κf = .5,
                  κ∇ = .5,
                  κh = .5,
                  maxIter=100,
                  σmin = eps(),
                  verbose = -1, 
                  print_every = 2
                  )

  f = x -> obj(nlp, x)
  ∇f! = (g, x) -> grad!(nlp, x, g)



  x0 = nlp.meta.x0
  xk = copy(x0)
  hk = h(xk)

  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk)

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√(ξσ)" "ρ" "σ" "‖x‖" "‖s‖" ""
    #! format: off
  end

  local ξ

  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk
  
  fk = f(xk)
  ϵ = noise(xk)

  ∇fk = similar(xk)
  ∇f!(∇fk, xk)

  h∇fk = similar(xk)
  ∇f!(h∇fk, xk + ϵ)
  mν∇fk = -ν * ∇fk

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    # define model (on calcule s exact donc on met les termes non-bruités )
    φk(d) = dot(∇fk, d)
    mk(d) = φk(d) + ψ(d)

    prox!(s, ψ, mν∇fk, ν)
    mks = mk(s)
    ξ = hk - mks

    if (ξ ≥ 0 && sqrt(ξ*σk) ≤ crit)
      optimal = true
      continue
    end
    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")

    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn)

    ΔFk = (fk + hk) - (fkn + hkn)
    ρk = ΔFk / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % print_every == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ*σk) ρk σk norm(xk) norm(s) σ_stat
      #! format: on
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      fk = fkn
      hk = hkn
      ∇f!(∇fk, xk)
      shift!(ψ, xk)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      @. mν∇fk = -ν * ∇fk
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e" k fk hk sqrt(ξ*σk) "" σk norm(xk) norm(s)
      #! format: on
      @info "R2: terminating with √(ξσ) = $(sqrt(ξ*σk))"
    end
  end

  return xk, k, optimal, fk, hk, ξ
end

# old
function naive_inexact_R2(nlp::ADNLPModel,
  h; 
  η1 = 0.02,
  η2 = 0.95,
  γ = 3.,
  ν = 1e-3,
  crit = 1e-3,
  κf = .5,
  κ∇ = .5,
  κh = .5,
  maxIter=100,
  σmin = eps(),
  verbose = -1, 
  print_every = 2, 
  activate_bounds=false
  )

  f = x -> obj(nlp, x)
  ∇f! = (g, x) -> grad!(nlp, x, g)

  x0 = nlp.meta.x0
  xk = copy(x0)
  hk = h(xk)

  ϵ = noise(xk)

  xkn = similar(xk)
  s = zero(xk)
  if activate_bounds==true
    ψ = shifted(h, xk)
  else
    ψ = shifted(h, xk + ϵ)
  end

  if verbose > 0
  #! format: off
  @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√(ξσ)" "ρ" "σ" "‖x‖" "‖s‖" ""
  #! format: off
  end

  local ξ

  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk

  fk = f(xk)
  hfk = f(xk + ϵ)
  hhk = h(xk + ϵ)

  ∇fk = similar(xk)
  ∇f!(∇fk, xk)

  h∇fk = similar(xk)
  ∇f!(h∇fk, xk + ϵ)
  if activate_bounds == true
    mν∇fk = -ν * h∇fk
  else
    mν∇fk = -ν * ∇fk
  end
  

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    prox!(s, ψ, mν∇fk, ν)
    # define model 
    if activate_bounds==true
      φk = dot(h∇fk, s)
    else
      φk = dot(∇fk, s)
    end

    mks = φk + ψ(s)
    if activate_bounds==true
      ξ = hhk - mks
    else
      ξ = hk - mks
    end

    # conditions de convergence
    # pour f
    if (abs(hfk - fk) > κf*norm(s)^2) && (activate_bounds == true)
      if verbose > 0
        @info "Bound on f not respected at iteration $k : reducing noise"
      end
      while abs(hfk - fk) > κf*norm(s)^2
        ϵ ./= 10
        hfk = f(xk + ϵ)
      end
      @info "   ϵ ~ 1e$(floor(Int, log10(abs(ϵ[1]))))" # pour obtenir la puissance
    end

    # pour ∇f
    if norm(∇fk - h∇fk) > κ∇*norm(s) && activate_bounds == true
      if verbose > 0
        @info "Bound on ∇f not respected at iteration $k : reducing noise"
      end      
      while norm(∇fk - h∇fk) > κ∇*norm(s)
        ϵ ./= 10
        ∇f!(h∇fk, xk + ϵ) # pourquoi h∇fk n'est pas mis à jour avec les itérations??
        println(h∇fk)
        println(∇fk)
      end
      @info "   ϵ ~ 1e$(floor(Int, log10(abs(ϵ[1]))))"

      mν∇fk = -ν * h∇fk
      ψ = shifted(h, xk + ϵ)

      prox!(s, ψ, mν∇fk, ν)
      mks = dot(h∇fk, s) + ψ(s)
      ξ = hhk - mks
    end

    # pour h 
    if abs(hhk - hk) > κh*ξ && activate_bounds == true
      if verbose > 0
        @info "Bound on h not respected at iteration $k : reducing noise"
      end        
      while abs(hhk - hk) > κh*ξ
        ϵ ./= 10
        hhk = h(xk + ϵ)
      end
      @info "   ϵ ~ 1e$(floor(Int, log10(abs(ϵ[1]))))"
      ψ = shifted(h, xk + ϵ)
      prox!(s, ψ, mν∇fk, ν)
      mks = mk(s)
      ξ = hhk - mks
    end

    if (ξ ≥ 0 && sqrt(ξ*σk) ≤ crit)
      optimal = true
      continue
    end
    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")

    xkn .= xk .+ s 
    fkn = f(xkn)
    hkn = h(xkn)

    hfkn = f(xkn + ϵ)
    hhkn = h(xkn + ϵ)


    if activate_bounds == true
      ΔFk = (hfk + hhk) - (hfkn + hhkn)
    else
      ΔFk = (fk + hk) - (fkn + hkn)
    end
    ρk = ΔFk / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % print_every == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ*σk) ρk σk norm(xk) norm(s) σ_stat
      #! format: on
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      if activate_bounds == true
        fk = hfkn
        hk = hhkn
        ∇f!(h∇fk, xk)
      else
        fk = fkn
        hk = hkn
        ∇f!(∇fk, xk)
      end
      shift!(ψ, xk)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      if activate_bounds == true
        @. mν∇fk = -ν * h∇fk
      else
        @. mν∇fk = -ν * ∇fk
      end
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e" k fk hk sqrt(ξ*σk) "" σk norm(xk) norm(s)
      #! format: on
      @info "R2: terminating with √(ξσ) = $(sqrt(ξ*σk))"
    end
  end

  return xk, k, optimal, fk, hk, ξ
end

# callback pour le changement de précision
function callback_precision(p::Int)
  E = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-20, 0.]
  l = length(E)
  if p > l
    return l, E[l]
  end
  return p, E[p]
end

# new
function naive_inexact_R2(nlp::ADNLPModel,
  h; 
  η1 = 0.02,
  η2 = 0.95,
  γ = 3.,
  ν = 1e-3,
  crit = 1e-3,
  κf = .5,
  κ∇ = .5,
  κh = .5,
  maxIter=100,
  σmin = eps(),
  verbose = -1, 
  print_every = 2, 
  activate_bounds=false, 
  info_bounds = false
  )

  f = x -> obj(nlp, x)
  ∇f! = (g, x) -> grad!(nlp, x, g)

  x0 = nlp.meta.x0
  xk = copy(x0)
  hk = h(xk)

  p = 1
  p, ϵ = callback_precision(p) # bruit initial

  xkn = similar(xk)
  s = zero(xk)

  ψ = shifted(h, xk)

  if verbose > 0
  #! format: off
  @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s %6s " "iter" "f(x)" "h(x)" "√(ξσ)" "ρ" "σ" "‖x‖" "‖s‖" "" "p" 
  #! format: off
  end

  local ξ

  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)

  h∇fk = ∇fk * (1+ϵ*(rand()-1/2))
  hfk = fk * (1+ϵ*(rand()-1/2))
  hhk = hk * (1+ϵ*(rand()-1/2))

  # Désormais, on a les 3 quantités inexactes.

  if activate_bounds == true
    mν∇fk = -ν * h∇fk
  else
    mν∇fk = -ν * ∇fk
  end
  

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    prox!(s, ψ, mν∇fk, ν)

    # define model 
    if activate_bounds==true
      φk = dot(h∇fk, s)
      mks = φk + ψ(s) * (1+ϵ*(rand()-1/2))
    else
      φk = dot(∇fk, s)
      mks = φk + ψ(s) 
    end
    
    if activate_bounds==true
      ξ = hhk - mks
    else
      ξ = hk - mks
    end

    # conditions de convergence
    # pour f
    if (abs(hfk - fk) > κf*norm(s)^2) && activate_bounds == true
      if info_bounds == true
        @info "Bound on f not respected at iteration $k : reducing noise"
      end
      while abs(hfk - fk) > κf*norm(s)^2
        p+=1
        p, ϵ = callback_precision(p)

        hfk = fk * (1+ϵ*(rand()-1/2))
      end
      if verbose > 0 && ϵ > 0 && info_bounds == true
        @info "   ϵ ~ 1e$(floor(Int, log10(abs(ϵ))))" # pour obtenir la puissance
        @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s %6s " "iter" "f(x)" "h(x)" "√(ξσ)" "ρ" "σ" "‖x‖" "‖s‖" "" "p" 
      end
    end

    # pour ∇f
    if norm(∇fk - h∇fk) > κ∇*norm(s) && activate_bounds == true
      if verbose > 0 info_bounds == true
        @info "Bound on ∇f not respected at iteration $k : reducing noise"
      end      
      while norm(∇fk - h∇fk) > κ∇*norm(s)
        p+=1
        p, ϵ = callback_precision(p)
        h∇fk = ∇fk * (1+ ϵ*(rand()-1/2))
      end
      if verbose > 0 && ϵ > 0 && info_bounds == true
        @info "   ϵ ~ 1e$(floor(Int, log10(abs(ϵ))))" # pour obtenir la puissance
        @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s %6s " "iter" "f(x)" "h(x)" "√(ξσ)" "ρ" "σ" "‖x‖" "‖s‖" "" "p" 
      end

      mν∇fk = -ν * h∇fk

      prox!(s, ψ, mν∇fk, ν)
      mks = dot(h∇fk, s) + ψ(s) * (1+ ϵ*(rand()-1/2))
      ξ = hhk - mks
    end

    # pour h 
    if abs(hhk - hk) > κh*ξ && activate_bounds == true
      if verbose > 0 && info_bounds == true
        @info "Bound on h not respected at iteration $k : reducing noise"
      end        
      while abs(hhk - hk) > κh*ξ
        p+=1
        p, ϵ = callback_precision(p)

        hhk = hk * (1+ ϵ*(rand()-1/2))
        mks = dot(h∇fk, s) + ψ(s) * (1+ ϵ*(rand()-1/2))
        ξ = hhk - mks
      end
      if verbose > 0 && ϵ > 0 && info_bounds == true
        @info "   ϵ ~ 1e$(floor(Int, log10(abs(ϵ))))" # pour obtenir la puissance
        @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s %6s " "iter" "f(x)" "h(x)" "√(ξσ)" "ρ" "σ" "‖x‖" "‖s‖" "" "p" 
      end

      prox!(s, ψ, mν∇fk, ν)
      mks = dot(h∇fk, s) + ψ(s) * (1+ ϵ*(rand()-1/2))
      ξ = hhk - mks
    end

    if (ξ ≥ 0 && sqrt(ξ*σk) ≤ crit)
      optimal = true
      continue
    end
    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")

    xkn .= xk .+ s 
    fkn = f(xkn)
    hkn = h(xkn)

    hfkn = fkn * (1+ ϵ*(rand()-1/2))
    hhkn = hkn * (1+ ϵ*(rand()-1/2))

    if activate_bounds == true
      ΔFk = (hfk + hhk) - (hfkn + hhkn)
    else
      ΔFk = (fk + hk) - (fkn + hkn)
    end
    ρk = ΔFk / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % print_every == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s %6d" k fk hk sqrt(ξ*σk) ρk σk norm(xk) norm(s) σ_stat p
      #! format: on
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      ∇f!(∇fk, xk)
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
    end

    hfk = fk * (1+ ϵ*(rand()-1/2))
    hhk = hk * (1+ ϵ*(rand()-1/2))
    h∇fk = ∇fk * (1+ ϵ*(rand()-1/2))

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      if activate_bounds == true
        @. mν∇fk = -ν * h∇fk
      else
        @. mν∇fk = -ν * ∇fk
      end
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e" k fk hk sqrt(ξ*σk) "" σk norm(xk) norm(s) 
      #! format: on
      @info "R2: terminating with √(ξσ) = $(sqrt(ξ*σk))"
    end
  end

  return xk, k, optimal, fk, hk, ξ, ϵ
end