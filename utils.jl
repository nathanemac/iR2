# Test conditions initiales 
function test_κ(κs, κf, κ∇, κh, η1, η2)
  if 1/2*κs*(1- η2) - (2κf + κ∇) ≤ 0
    @error "Initial parameters κs, κf, κg, η2 don't respect convergence conditions."

  elseif 1/2*κs*η1 - (2κf + κh) ≤ 0
    @error "Initial parameters κs, κf, κh, η1 don't respect convergence conditions."
  end
end

function test_condition_f(fk, sk, σk, κf, pf, ps, Π, k, verb, flags) # p : current level of precision
  while abs(fk[pf])*(1- 1/(1 + eps(Π[pf]))) > κf*σk*norm(sk[ps])^2
    if ((Π[pf] == Π[end]) && (Π[ps] == Π[end]))
      if (flags[1] == false)
        @warn "maximum precision already reached on f and s at iteration $k."
        flags[1] = true
      end
      break
    end
    verb ==true && @info "condition on f not reached at iteration $k with precision $(Π[pf]) on f and $(Π[ps]) on s. Increasing precision : "
    if Π[pf] == Π[end] 
      verb ==true && @info " └──> maximum precision already reached on f. Trying to increase precision on s."
      ps+=1
    else
      pf+=1
    end
    verb ==true && @info " └──> current precision on f is now $(Π[pf]) and s is now $(Π[ps])"
  end
  return pf, ps, flags
end

function test_condition_h(hk, sk, σk, κh, ph, ps, Π, k, verb, flags) # p : current level of precision
  while abs(hk[ph])*(1- 1/(1 + eps(Π[ph]))) > κh*σk*norm(sk[ps])^2
    if (Π[ph] == Π[end]) && (Π[ps] == Π[end])
      if (flags[2] == false)
        @warn "maximum precision already reached on h and s at iteration $k."
        flags[2] = true
      end
      break
    end
    verb ==true && @info "condition on h not reached at iteration $k with precision $(Π[ph]) on h and $(Π[ps]) on s. Increasing precision : "
    if Π[ph] == Π[end] 
      @info " └──> maximum precision already reached on h. Trying to increase precision on s."
      ps+=1
    else
      ph+=1
    end
    verb ==true && @info " └──> current precision on h is now $(Π[ph]) and s is now $(Π[ps])"
  end
  return ph, ps, flags
end

function test_condition_∇f(gfk, sk, σk, κ∇, pg, ps, Π, k, verb, flags) # p : current level of precision
  while norm(gfk[pg])*(1- 1/(1 + eps(Π[pg]))) > κ∇*σk*norm(sk[ps])
    if (Π[pg] == Π[end]) && (Π[ps] == Π[end])
      if (flags[3] == false)
        @warn "maximum precision already reached on ∇f and s at iteration $k."
        flags[3] = true
      end
      break
    end
    verb ==true && @info "condition on ∇f not reached at iteration $k with precision $(Π[pg]) on ∇f and $(Π[ps]) on s. Increasing precision : "
    if Π[ps] == Π[end] 
      verb ==true && @info " └──> maximum precision already reached on s. Trying to increase precision on ∇f."
      pg+=1
    else
      ps+=1
    end
    verb ==true && @info " └──> current precision on ∇f is now $(Π[pg]) and s is now $(Π[ps])"
  end
  return pg, ps, flags
end

# check assumption 6
function test_assumption_6(ξ, κs, σk, sk, Π, ps, ph, mk, hk, k, verb, flags)
  while ξ < 1/2*κs*σk*norm(sk[ps])^2
    if (Π[ps] == Π[end]) && (Π[ph] == Π[end])
      if (flags[2] == false)
        @warn "maximum precision already reached on f and s at iteration $k."
        flags[2] = true
      end
      break
    end
    verb == true && @info "condition on Assumption 6 not reached at iteration $k with precision $(Π[ps]) on s and $(Π[ph]) on h. Increasing precision : "
    if Π[ph] == Π[end]
      verb ==true && @info " └──> maximum precision already reached on h to satisfy Assumption 6. Trying to increase precision on s."
      ps+=1
      mks = Π[ps](mk(sk[ps]))
      ξ = hk[ph] - mks + max(1, abs(hk[ph])) * 10 * eps() #TODO change eps() to eps(Π[p]), but which one?
    else
      ph+=1
      ξ = hk[ph] - mks + max(1, abs(hk[ph])) * 10 * eps()
    end
    verb == true && @info " └──> current precision on h is now $(Π[ph]) and s is now $(Π[ps])"
  end
  return ps, ph, ξ, flags
end

######################################################
#################### Real MP #########################
######################################################

function test_condition_f(fk, sk, σk, κf, pf, ps, Π, k, verb, flags) # p : current level of precision
  while abs(fk[pf])*(1- 1/(1 + eps(Π[pf]))) > κf*σk*norm(sk[ps])^2
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
      ps+=1
      #TODO comment recalculer s avec la nouvelle précision?
    else
      pf+=1
      #TODO comment recalculer f(xk) avec la nouvelle précision?

    end
    verb ==true && @info " └──> current precision on f is now $(Π[pf]) and s is now $(Π[ps])"
  end
  return pf, ps, flags
end