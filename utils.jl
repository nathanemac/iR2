# # Test conditions initiales 
function check_κ_valid(κs, κf, κ∇, κh, η1, η2)
   if 1/2*κs*(1- η2) - (2κf + κ∇) ≤ 0
     @error "Initial parameters κs, κf, κg, η2 don't respect convergence conditions."

   elseif 1/2*κs*η1 - 2(κf + κh) ≤ 0
     @error "Initial parameters κs, κf, κh, η1 don't respect convergence conditions."
   end
 end

######################################################
#################### Real MP #########################
######################################################

function test_condition_f(nlp, solver, p, Π, k)
  while abs(solver.fk[p.pf])*(1- 1/(1 + eps(Π[p.pf]))) > p.κf*p.σk*norm(solver.sk[p.ps])^2
    if ((Π[p.pf] == Π[end]) && (Π[p.ps] == Π[end]))
      if (p.flags[1] == false)
        @warn "maximum precision already reached on f and s at iteration $k."
        p.flags[1] = true
      end
      break # on passe sous le tapis pour les fois d'après que la condition passe pas.
    end
    p.verb == true && @info "condition on f not reached at iteration $k with precision $(Π[p.pf]) on f and $(Π[p.ps]) on s. Increasing precision : "
    if Π[p.pf] == Π[end]
      p.verb == true && @info " └──> maximum precision already reached on f. Trying to increase precision on s."
      recompute_prox!(nlp, solver, p, k, Π)
    else
      p.pf+=1
      solver.fk[p.pf] = obj(nlp, solver.xk[p.pf])
      solver.special_counters[:f][p.pf] += 1
      for i=1:length(Π)
        solver.fk[i] = solver.fk[p.pf]
      end
    p.verb == true && @info " └──> current precision on f is now $(Π[p.pf]) and s is $(Π[p.ps])"
    end
  end
  return 
end

function test_condition_h(nlp, solver, p, Π, k) # p : current level of precision
  while abs(solver.hk[p.ph])*(1- 1/(1 + eps(Π[p.ph]))) > p.κh*p.σk*norm(solver.sk[p.ps])^2
    if (Π[p.ph] == Π[end]) && (Π[p.ps] == Π[end])
      if (p.flags[2] == false)
        @warn "maximum precision already reached on h and s for condition on h at iteration $k."
        p.flags[2] = true
      end
      break
    end
    p.verb == true && @info "condition on h not reached at iteration $k with precision $(Π[p.ph]) on h and $(Π[p.ps]) on s. Increasing precision : "
    if Π[p.ph] == Π[end] 
      @info " └──> maximum precision already reached on h. Trying to increase precision on s."
      recompute_prox!(nlp, solver, p, k, Π)
    else
      p.ph+=1
      solver.hk[p.ph] = solver.h(solver.xk[p.ph])
      solver.special_counters[:h][p.ph] += 1
      for i=1:length(Π)
        solver.hk[i] = solver.hk[p.ph]
      end
    end
    p.verb == true && @info " └──> current precision on s is now $(Π[p.ps]) and h is $(Π[p.ph])"
  end
  return
end


function test_condition_∇f(nlp, solver, p, Π, k)
  while abs(dot(solver.gfk[p.pg], solver.sk[p.ps]))*(1- 1/(1 + eps(Π[p.pg]))) > p.κ∇*p.σk*norm(solver.sk[p.ps])
    if (Π[p.pg] == Π[end]) && (Π[p.ps] == Π[end])
      if (p.flags[3] == false)
        @warn "maximum precision already reached on ∇f and s for condition on ∇f at iteration $k."
        p.flags[3] = true
      end
      break
    end
    p.verb == true && @info "condition on ∇f not reached at iteration $k with precision $(Π[p.pg]) on ∇f and $(Π[p.ps]) on s. Increasing precision : "
    if Π[p.pg] == Π[end] 
      p.verb == true && @info " └──> maximum precision already reached on ∇f. Trying to increase precision on s."
      recompute_prox!(nlp, solver, p, k, Π)
    else
      recompute_grad!(nlp, solver, p, k, Π)
    end
    p.verb == true && @info " └──> current precision on s is now $(Π[p.ps]) and ∇f is $(Π[p.pg])"
  end
  return
end


# check assumption 6
function test_assumption_6(nlp, solver, options, p, Π, k, ξ)
  while ξ < 1/2*p.κs*p.σk*norm(solver.sk[p.ps])^2 
    if (Π[p.ps] == Π[end]) && (Π[p.ph] == Π[end])
      if (p.flags[2] == false)
        @warn "maximum precision already reached on f and s for Assumption 6 at iteration $k."
        p.flags[2] = true
      end
      break
    end

    p.verb == true && @info "condition on Assumption 6 not reached at iteration $k with precision $(Π[p.ps]) on s and $(Π[p.ph]) on h. Increasing precision : "
    if Π[p.ph] == Π[end]
      p.verb == true && @info " └──> maximum precision already reached on h to satisfy Assumption 6. Trying to increase precision on s."

      recompute_prox!(nlp, solver, p, k, Π)

      φk(d) = dot(solver.gfk[end], d)
      mks = φk(solver.sk[p.ps]) + solver.ψ(solver.sk[p.ps])
      ξ = solver.hk[p.ph] - mks + max(1, abs(solver.hk[p.ph])) * 10 * eps()

      sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)
      while ξ < 0 && sqrt_ξ_νInv > options.neg_tol && p.ps < length(Π)
        @info " └──> R2: prox-gradient step should produce a decrease but ξ = $(ξ). Increasing precision on s."
        recompute_prox!(nlp, solver, p, k, Π)
        φk(d) = dot(solver.gfk[p.pg], d)
        mk(d) = φk(d) + solver.ψ(d) # FP format : highest between φk and ψ
  
        mks = mk(solver.sk[p.ps])
        solver.special_counters[:h][p.ps] += 1
        ξ = solver.hk[p.ph] - mks + max(1, abs(solver.hk[p.ph])) * 10 * eps()
        sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)
      end

    else
      p.ph+=1
      solver.hk[p.ph] = solver.h(solver.xk[p.ph])
      solver.special_counters[:h][p.ph] += 1
      for i=1:length(Π)
        solver.hk[i] = solver.hk[p.ph]
      end
      mks = dot(solver.gfk[end], solver.sk[p.ps]) + solver.ψ(solver.sk[p.ps])
      ξ = solver.hk[p.ph] - mks + max(1, abs(solver.hk[p.ph])) * 10 * eps()

      sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)

      
      while ξ < 0 && sqrt_ξ_νInv > neg_tol && p.ph < length(Π)
        @info " └──> R2: prox-gradient step should produce a decrease but ξ = $(ξ). Increasing precision on h."
        p.ph+=1
        solver.hk[p.ph] = solver.h(solver.xk[p.ph])
        solver.special_counters[:h][p.ph] += 1
        for i=1:length(Π)
          solver.hk[i] = solver.hk[p.ph]
        end
        ξ = hk[ph] - mks + max(1, abs(hk[ph])) * 10 * eps()
        sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)
      end
    end
    p.verb == true && @info " └──> current precision on s is $(Π[p.ps]) and h is $(Π[p.ph])"
  end
  sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / p.ν) : sqrt(-ξ / p.ν)
  return ξ
end

function recompute_grad!(nlp, solver, p, k, Π) 
  if Π[p.pg] == Π[end]
    @warn "maximum precision already reached on ∇f when recomputing gradient at iteration $k."
    return 
  end
  p.pg+=1
  p.verb==true && @info "recomputing gradient at iteration $k with precision $(Π[p.pg])"
  grad!(nlp, solver.xk[p.pg], solver.gfk[p.pg])
  solver.special_counters[:∇f][p.pg] += 1
  for i=1:length(Π)
    solver.gfk[i] .= solver.gfk[p.pg]
  end

  for i=1:length(Π)
    solver.mν∇fk[i] .= -Π[end].(p.ν) * solver.gfk[i]
  end  
  return
end

function recompute_prox!(nlp, solver, p, k, Π) 
  # first, recompute_grad because we need the updated version of solver.mν∇fk to compute the proximal operator 
  recompute_grad!(nlp, solver, p, k, Π)

  # then, recompute proximal operator
  if Π[p.ps] == Π[end]
    @warn "maximum precision already reached on s when recomputing prox at iteration $k."
    solver.h = NormL1(Π[p.ps](1.0))
    solver.ψ = shifted(solver.h, solver.xk[p.ps])
    return 
  end

  p.ps+=1

  solver.h = NormL1(Π[p.ps](1.0))
  hxk = solver.h(solver.xk[p.ps]) #TODO add selected
  for i=1:length(Π)
    solver.hk[i] = Π[i].(hxk)
  end
  solver.ψ = shifted(solver.h, solver.xk[p.ps])

  prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps].(p.ν)) # on recalcule le prox en la précision de ps. 
  solver.special_counters[:prox][p.ps] += 1
  for i=1:length(Π)
    solver.sk[i] .= solver.sk[p.ps]
  end
  return
end