# # Test conditions initiales 
function test_κ(κs, κf, κ∇, κh, η1, η2)
   if 1/2*κs*(1- η2) - (2κf + κ∇) ≤ 0
     @error "Initial parameters κs, κf, κg, η2 don't respect convergence conditions."

   elseif 1/2*κs*η1 - (2κf + κh) ≤ 0
     @error "Initial parameters κs, κf, κh, η1 don't respect convergence conditions."
   end
 end

######################################################
#################### Real MP #########################
######################################################

function test_condition_f(nlp, solver, p, Π, flags)
  while abs(solver.fk[p.pf])*(1- 1/(1 + eps(Π[p.pf]))) > p.κf*p.σk*norm(solver.sk[p.ps])^2
    if ((Π[p.pf] == Π[end]) && (Π[p.ps] == Π[end]))
      if (flags[1] == false)
        @warn "maximum precision already reached on f and s at iteration $k."
        flags[1] = true
      end
      break # on passe sous le tapis pour les fois d'après que la condition passe pas.
    end
    p.verb == true && @info "condition on f not reached at iteration $k with precision $(Π[p.pf]) on f and $(Π[p.ps]) on s. Increasing precision : "
    if Π[p.pf] == Π[end]
      p.verb == true && @info " └──> maximum precision already reached on f. Trying to increase precision on s."
      ... = recompute_prox!(nlp, ...)
    else
      p.pf+=1
      solver.fk[p.pf] = obj(nlp, solver.xk[p.pf])
      for i=1:len(Π)
        solver.fk[i] .= solver.fk[p.pf]
      end
    verb == true && @info " └──> current precision on f is now $(Π[p.pf]) and s is $(Π[p.ps])"
  end
  return flags
end

function test_condition_h(nlp, h, hk, sk, xk, σk, κh, ph, ps, s, pg, gfk, ∇fk, mν∇fk, ν, Π, k, verb, flags) # p : current level of precision
  while abs(hk[ph])*(1- 1/(1 + eps(Π[ph]))) > κh*σk*norm(sk[ps])^2
    if (Π[ph] == Π[end]) && (Π[ps] == Π[end])
      if (flags[2] == false)
        @warn "maximum precision already reached on h and s for condition on h at iteration $k."
        flags[2] = true
      end
      break
    end
    verb ==true && @info "condition on h not reached at iteration $k with precision $(Π[ph]) on h and $(Π[ps]) on s. Increasing precision : "
    if Π[ph] == Π[end] 
      @info " └──> maximum precision already reached on h. Trying to increase precision on s."
      h, ps, s, pg, gfk, ∇fk, mν∇fk, ν, sk = recompute_prox!(nlp, pg, k, Π, xk, gfk, ∇fk, ps, s, mν∇fk, ν, sk)
    else
      ph+=1
      hk[ph] = h(xk[ph])
    end
    verb ==true && @info " └──> current precision on s is now $(Π[ps]) and h is $(Π[ph])"
  end
  return h, ps, s, pg, gfk, ∇fk, mν∇fk, ν, sk, ph, hk, flags
end

function test_condition_∇f(nlp, h, gfk, ∇fk, sk, xk, σk, κ∇, pg, ps, s, mν∇fk, ν, Π, k, verb, flags)
  while norm(gfk[pg])*(1- 1/(1 + eps(Π[pg]))) > κ∇*σk*norm(sk[ps])
    if (Π[pg] == Π[end]) && (Π[ps] == Π[end])
      if (flags[3] == false)
        @warn "maximum precision already reached on ∇f and s for condition on ∇f at iteration $k."
        flags[3] = true
      end
      break
    end
    verb ==true && @info "condition on ∇f not reached at iteration $k with precision $(Π[pg]) on ∇f and $(Π[ps]) on s. Increasing precision : "
    if Π[pg] == Π[end] 
      verb ==true && @info " └──> maximum precision already reached on ∇f. Trying to increase precision on s."
      h, ps, s, pg, gfk, mν∇fk, ν, sk = recompute_prox!(nlp, pg, k, Π, xk, gfk, ∇fk, ps, s, mν∇fk, ν, sk)
    else
      pg, gfk, mν∇fk, ν, ∇fk = recompute_grad!(nlp, pg, k, Π, xk, gfk, mν∇fk, ν, ∇fk)
    end
    verb ==true && @info " └──> current precision on s is now $(Π[ps]) and ∇f is $(Π[pg])"
  end
  return h, ps, s, pg, gfk, ∇fk, mν∇fk, ν, sk, flags
end


# check assumption 6
function test_assumption_6(ξ, κs, σk, sk, xk, Π, ps, s, pg, gfk, ∇fk, mν∇fk, ν, ph, mk, mks, hk, k, verb, flags)
  while ξ < 1/2*κs*σk*norm(sk[ps])^2
    if (Π[ps] == Π[end]) && (Π[ph] == Π[end])
      if (flags[2] == false)
        @warn "maximum precision already reached on f and s for Assumption 6 at iteration $k."
        flags[2] = true
      end
      break
    end

    verb == true && @info "condition on Assumption 6 not reached at iteration $k with precision $(Π[ps]) on s and $(Π[ph]) on h. Increasing precision : "
    if Π[ph] == Π[end]
      verb ==true && @info " └──> maximum precision already reached on h to satisfy Assumption 6. Trying to increase precision on s."
      h, ps, s, pg, gfk, ∇fk, mν∇fk, ν, sk = recompute_prox!(nlp, pg, k, Π, xk, gfk, ∇fk, ps, s, mν∇fk, ν, sk)

      mks = mk(sk[ps])
      ξ = hk[ph] - mks + max(1, abs(hk[ph])) * 10 * eps()

    else
      ph+=1
      hk[ph] = h(xk[ph])
      ξ = hk[ph] - mks + max(1, abs(hk[ph])) * 10 * eps()
    end
    verb == true && @info " └──> current precision on h is now s is $(Π[ps]) and h is $(Π[ph])"
  end
  return h, ps, ph, ξ, flags, s, pg, gfk, ∇fk, mν∇fk, ν, sk, mks
end



function recompute_grad!(nlp, solver, p, Π, k) # p : current level of precision
  if Π[p.pg] == Π[end]
    @warn "maximum precision already reached on ∇f when recomputing gradient at iteration $k."
    return 
  end
  p.pg+=1
  @info "recomputing gradient at iteration $k with precision $(Π[p.pg])"
  grad!(nlp, solver.gfk[p.pg], solver.xk[p.pg])
  for i=1:len(Π)
    solver.gfk[i] .= solver.gfk[p.pg]
  end

  p.ν = Π[p.pg](ν) # toujours en la précision du gradient
  @. solver.mν∇fk = -p.ν * solver.gfk[p.pg]
  return
end

function recompute_prox!(nlp, solver, p, k, Π) 
  # first, recompute_grad because we need the updated version of solver.mν∇fk to compute the proximal operator 
  recompute_grad!(nlp, solver, p, Π, k)

  # then, recompute proximal operator
  if Π[p.ps] == Π[end]
    @warn "maximum precision already reached on s when recomputing prox at iteration $k."
    return 
  end

  p.ps+=1

  h = NormL1(Π[p.ps](1.0))
  ψ = shifted(h, solver.xk[p.ps])

  solver.mν∇fk = Π[p.ps].(solver.mν∇fk) # pourquoi le cast? 
  p.ν = Π[p.ps].(p.ν)

  prox!(solver.sk[p.ps], ψ, solver.mν∇fk, p.ν)
  for i=1:len(Π)
    solver.sk[i] .= solver.sk[p.ps]
  end

  return 
end