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

function test_condition_f(nlp, solver, verb, Π, flags) # p : current level of precision
  while abs(solver.fk[pf])*(1- 1/(1 + eps(Π[pf]))) > κf*σk*norm(solver.sk[ps])^2
    if ((Π[pf] == Π[end]) && (Π[ps] == Π[end]))
      if (flags[1] == false)
        @warn "maximum precision already reached on f and s at iteration $k."
        flags[1] = true
      end
      break
    end
    verb ==true && @info "condition on f not reached at iteration $k with precision $(Π[pf]) on f and $(Π[ps]) on s. Increasing precision : "
    if Π[pf] == Π[end]
      verb == true && @info " └──> maximum precision already reached on f. Trying to increase precision on s."
      ps, pg = recompute_prox!(nlp, ...)
    else
      pf+=1
      solver.fk[pf] = obj(nlp, solver.xk[pf])
      for i=1:len(Π)
        solver.fk[i] .= solver.fk[pf]
      end
    verb ==true && @info " └──> current precision on f is now $(Π[pf]) and s is $(Π[ps])"
  end
  return pf, ps, flags
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



function recompute_grad!(nlp, solver, Π, pg) # p : current level of precision
  if Π[pg] == Π[end]
    @warn "maximum precision already reached on ∇f when recomputing gradient at iteration $k."
    return pg 
  end
  pg+=1
  @info "recomputing gradient at iteration $k with precision $(Π[pg])"
  grad!(nlp, solver.gfk[pg], xk[pg])
  for i=1:len(Π)
    solver.gfk[i] .= solver.gfk[pg]
  end

  ν = Π[pg](ν) # toujours en la précision du gradient
  @. solver.mν∇fk = -ν * gfk[pg]
  return pg
end

function recompute_prox!(nlp, pg, k, Π, xk, gfk, ∇fk, ps, s, mν∇fk, ν, sk) 
  # first, recompute_grad because we need the updated version to compute the proximal operator
  pg, gfk, mν∇fk, ν, ∇fk = recompute_grad!(nlp, pg, k, Π, xk, gfk, mν∇fk, ν, ∇fk)

  # then, recompute proximal operator
  if Π[ps] == Π[end]
    @warn "maximum precision already reached on s when recomputing prox at iteration $k."
    return h, ps, s, pg, gfk, ∇fk, mν∇fk, ν, sk
  end
  ps+=1

  h = NormL1(Π[ps](1.0))
  ψ = shifted(h, xk[ps])

  mν∇fk = Π[ps].(mν∇fk)
  ν = Π[ps].(ν)
  s = Π[ps].(s)
  prox!(s, ψ, mν∇fk, ν) 

  sk[ps] .= s
  return h, ps, s, pg, gfk, ∇fk, mν∇fk, ν, sk
end