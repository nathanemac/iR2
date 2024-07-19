# ce code est le code principal de l'algorithme iR2-Reg. 

import SolverCore.solve!

"""
    mutable struct iR2RegParams{T<:Real, H<:Real}

Structure to hold parameters for the iR2Reg algorithm, including mixed-precision settings and various conditions.

# Fields
- `Π::Vector{DataType}`: A vector containing floating point datatypes. By default, Π is `[Float16, Float32, Float64]`.
- `pf::Int`: Current level of precision for `f`.
- `pg::Int`: Current level of precision for `∇f`.
- `ph::Int`: Current level of precision for `h`.
- `ps::Int`: Current level of precision for `s`.
- `verb::Bool`: Whether to activate verbosity in mixed-precision mode.
- `activate_mp::Bool`: Whether to activate mixed-precision mode.
- `flags::Vector{Bool}`: Flags to check if maximum precision for `f`, `h`, and `s` has been reached. Once a flag is set to true, precision will no longer be increased, and no warnings will be printed.
- `κf::T`: Condition on `f` (refer to the paper for more details).
- `κh::T`: Condition on `h` (refer to the paper for more details).
- `κ∇::T`: Condition on `∇f` (refer to the paper for more details).
- `κs::T`: Condition on `s` (refer to the paper for more details).
- `κξ::T`: Condition on `ξ` (refer to the paper for more details).
- `H`: The highest floating point format in `Π`. Used to cast some inexpensive values to avoid rounding errors.
- `σk::H`: Current value of `σk` (refer to the algorithm in the paper).
- `ν::H`: Current value of `ν = 1/σk` (refer to the algorithm in the paper).
"""
mutable struct iR2RegParams{T<:Real, H<:Real}
  Π::Vector{DataType}
  pf::Int
  pg::Int
  ph::Int
  ps::Int
  verb::Bool
  activate_mp::Bool
  flags::Vector{Bool}
  κf::T
  κh::T
  κ∇::T
  κs::T
  κξ::T
  H # H is the "Highest" floating point format in Π.
  σk::H 
  ν::H
end


"""
    iR2RegParams(Π::Vector{DataType}; pf=1, pg=1, ph=1, ps=1, verb::Bool=false, activate_mp::Bool=true, flags::Vector{Bool}=[false, false, false], κf=1e-5, κh=2e-5, κ∇=4e-2, κs=1., κξ=1., H=Float128, σk=H(1.), ν=eps(H)^(1/5))

Create an instance of the iR2RegParams struct with specified or default parameter values.

# Arguments
- `Π::Vector{DataType}`: A vector containing floating point datatypes. By default, Π is [Float16, Float32, Float64].
- `pf::Int=1`: Current level of precision on f.
- `pg::Int=1`: Current level of precision on ∇f.
- `ph::Int=1`: Current level of precision on h.
- `ps::Int=1`: Current level of precision on s.
- `verb::Bool=false`: Whether to activate verbosity in mixed-precision mode.
- `activate_mp::Bool=true`: Whether to activate mixed-precision mode.
- `flags::Vector{Bool}=[false, false, false]`: Flags to check if maximum precision on f, h, and s has been reached. Once a flag is set to true, precision will no longer be increased, and no warnings will be printed.
- `κf=1e-5`: Condition on f (refer to the paper for more details).
- `κh=2e-5`: Condition on h (refer to the paper for more details).
- `κ∇=4e-2`: Condition on ∇f (refer to the paper for more details).
- `κs=1.`: Condition on s (refer to the paper for more details).
- `κξ=1.`: Condition on ξ (refer to the paper for more details).
- `H=Float128`: The highest floating point format in Π. Used to cast some inexpensive values to avoid rounding errors.
- `σk=H(1.)`: Current value of σk (refer to the algorithm in the paper).
- `ν=eps(H)^(1/5)`: Current value of ν = 1/σk (refer to the algorithm in the paper).

# Returns
- An instance of `iR2RegParams` with the specified or default parameter values.

# Errors
- Throws an error if `Π` is an empty vector.
- Displays a warning if the highest precision format in the algorithm and the highest precision format in `Π` are the same (Float128), as this may lead to unexpected behavior.

# Example
params = iR2RegParams([Float16, Float32, Float64], activate_mp=true, verb=true) # for mixed-precision with additional verbosity
params = iR2RegParams([Float32]; activate_mp=false,  verb=false) # for single precision without mixed-precision
"""
function iR2RegParams(Π::Vector{DataType}; pf = 1, pg = 1, ph = 1, ps = 1, verb::Bool=false, activate_mp::Bool=true, flags::Vector{Bool}=[false, false, false], κf=1e-5, κh=2e-5, κ∇=4e-2, κs=1., κξ=1., H=Float128, σk=H(1.), ν=eps(H)^(1/5))
  if length(Π) == 0
    error("Π must be a non-empty vector of floating point types")
  end
  if Π[end] == H && verb==true
    @warn "Highest precision format in the algorithm (H) and highest precision format in Π are the same ($H). \n This may lead to unexpected behavior. \n Please consider changing the highest precision format in Π to a lower precision format, or changing the highest precision format in the algorithm to a higher precision format."
  end
  return iR2RegParams(Π, pf, pg, ph, ps, verb, activate_mp, flags, κf, κh, κ∇, κs, κξ, H, σk, ν)
end

"""
    mutable struct iR2Solver{R<:Real, S<:AbstractVector} <: AbstractOptimizationSolver

Structure to hold the state and parameters of the iR2Solver algorithm.

# Fields
- `xk::Vector{S}`: Current iterate values. xk[i] contains the current iterate in the precision Π[i]. Type ?iR2RegParams for more details on Π.   
- `mν∇fk::Vector{S}`: Current gradient values scaled by -ν.
- `gfk::Vector{S}`: Current gradient values.
- `fk::S`: Smooth term values at the current iterate. fk[i] contains f(xk) in the precision Π[i]. Type ?iR2RegParams for more details on Π.   
- `hk::S`: Non-smooth term values at the current iterate.
- `sk::Vector{S}`: Current values of the proximal step. sk[i] contains s in the precision Π[i]. Type ?iR2RegParams for more details on Π.
- `xkn::S`: Potential next iterate values.
- `has_bnds::Bool`: Indicates if the problem is bound constrained.
- `l_bound::S`: Lower bound constraints.
- `u_bound::S`: Upper bound constraints.
- `l_bound_m_x::S`: Lower bounds shifted by the current iterate.
- `u_bound_m_x::S`: Upper bounds shifted by the current iterate.
- `Fobj_hist::Vector{R}`: History of smooth function values.
- `Hobj_hist::Vector{R}`: History of non-smooth term values.
- `Complex_hist::Vector{Int}`: History of iterations for solving the subproblem at each step.
- `p_hist::Vector{Vector{Int}}`: History of precision levels for f, ∇f, h, and s at each iteration.
- `special_counters::Dict{Symbol, Vector{Int}}`: Special counters for function evaluations.
- `h`: Non-smooth term (regularization function).
- `ψ`: Shifted proximable function.
- `ξ`: Criticality measure.
- `params::iR2RegParams`: Parameters for the iR2Reg algorithm.

# Example
solver = iR2Solver(reg_nlp, params, options)
"""
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
  h
  ψ#::G
  ξ
  params::iR2RegParams
end

"""
    iR2Solver(reg_nlp::AbstractRegularizedNLPModel, params::iR2RegParams, options::ROSolverOptions) where {T, V}

Initialize an instance of the `iR2Solver` struct with specified parameters and options.

# Arguments
- `reg_nlp::AbstractRegularizedNLPModel`: The regularized nonlinear programming model.
- `params::iR2RegParams`: Parameters for the iR2Reg algorithm.
- `options::ROSolverOptions`: Solver options.

# Returns
- An instance of `iR2Solver` initialized with the given parameters and options.

# Example
solver = iR2Solver(reg_nlp, params, options)
"""
function iR2Solver(
  reg_nlp::AbstractRegularizedNLPModel,
  params::iR2RegParams,
  options::ROSolverOptions,
) where {T, V}
  x0 = reg_nlp.model.meta.x0
  nvar = length(x0)
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar
  max_iter = options.maxIter
  Π = params.Π
  R = eltype(x0)
  xk = [Vector{eltype(T)}(undef, nvar) for T in Π]
  gfk = [Vector{eltype(T)}(undef, nvar) for T in Π]
  fk = [zero(eltype(T)) for T in Π]
  hk = [zero(eltype(T)) for T in Π]
  sk = [Vector{eltype(T)}(undef, nvar) for T in Π]
  mν∇fk = [Vector{eltype(T)}(undef, nvar) for T in Π]
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
  #h = reg_nlp.h
  h = NormL1(Π[1](1.0)) #TODO : change this to accept other regularizers.
  ψ = has_bnds ? shifted(reg_nlp.h, x0, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, x0)
  ξ = one(params.H)
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
    ψ,
    ξ,
    params
  )
end

"""
    iR2Reg(nlp, h, options, parameters)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs is the Taylor linear approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and σₖ > 0 is the regularization parameter.

The iR2Reg algorithm is a mixed-precision variant of the R2-Reg algorithm by Aravkin A., Baraldi R. and Orban D. (https://arxiv.org/abs/2103.15993).
If the flag activate_mp is set to true, the algorithm will work in mixed-precision mode. It will start in the lowest precision of Π and increase the precision of the variables as needed. Refer to the paper and documentation for more details.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `parameters::iR2RegParams`: a structure containing mixed-precision parameters

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (default = `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:length(x0)`).

The objective and gradient of `nlp` will be accessed.

### Returns

* `stats`: An instance of `GenericExecutionStats` containing usual values 
* `solver_specific` : a dictionary containing the following fields :
    - :smooth_obj : the value of the smooth part of the objective function at last iteration
    - :SubsolverCounter : the number of iterations spent to solve the subproblem
    - :Fhist : the history of the smooth part of the objective function
    - :eval_counters : a dictionary containing the number of evaluations of f, ∇f, h, and proximal operator at each iteration
    - :π_Hist : the history of the precision levels for f, ∇f, h, and s at each iteration           
    - :xi : the value of ξ at last iteration               
    - :Hhist : the history of the nonsmooth part of the objective function            
    - :nonsmooth_obj : the value of the nonsmooth part of the objective function at last iteration

### Example

* Input:
  - nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1.2, -1.345], backend=:generic)
  - h = NormL1(1.0)
  - options = ROSolverOptions(verbose=5, maxIter = 100, ϵa = 1e-4, ϵr = 1e-4)
  - params = iR2RegParams([Float16, Float32, Float64], activate_mp=true, verb=true)
  - my_res = iR2Reg(nlp, h, options, params)

* Output:
[ Info:   iter     f(x)     h(x)   √(ξ/ν)        ρ        σ      ‖x‖      ‖s‖   \n
[ Info: condition on f not reached at iteration 0 with precision Float16 on f and Float16 on s. Increasing precision : \n
[ Info:  └──> current precision on f is now Float32 and s is Float16 \n
[ Info: condition on f not reached at iteration 0 with precision Float32 on f and Float16 on s. Increasing precision : \n
[ Info:  └──> current precision on f is now Float64 and s is Float16 \n
[ Info: condition on h not reached at iteration 0 with precision Float16 on h and Float16 on s. Increasing precision : \n
[ Info:  └──> current precision on s is now Float16 and h is Float32 \n
[ Info:      0  9.1e+02  2.5e+00  1.7e+03  1.0e+00  5.5e+06  1.8e+00  3.1e-04               ↘ \n
[ Info:      5  9.1e+02  2.5e+00  1.7e+03  9.9e-01  6.1e+05  1.8e+00  2.8e-03               ↘ \n
[ Info:     10  8.3e+02  2.5e+00  1.6e+03  3.5e-01  2.3e+04  1.8e+00  2.4e-02               = \n
[ Info:     15  2.9e+02  1.8e+00  6.6e+02  9.6e-01  2.5e+03  1.3e+00  3.5e-01               ↘ \n
[ Info:     20  1.3e+00  1.9e-01  1.3e+01 -9.7e+00  2.8e+02  1.3e-01  1.9e-01               ↗ \n
[ Info: condition on h not reached at iteration 24 with precision Float32 on h and Float16 on s. Increasing precision : \n
[ Info:  └──> current precision on s is now Float16 and h is Float64 \n
[ Info:     25  9.7e-01  1.3e-01  7.9e-01  3.1e-01  2.8e+02  1.1e-01  9.0e-04               = \n
[ Info:     30  9.8e-01  1.0e-01  8.1e-01  1.0e+00  3.1e+01  9.1e-02  2.5e-02               ↘ \n
[ Info:     35  1.0e+00  0.0e+00  1.0e+00 -1.8e+00  3.1e+01  0.0e+00  2.7e-02               ↗ \n
[ Info:     40  9.9e-01  4.7e-03  1.7e+03  6.8e-01  2.8e+02  4.7e-03  1.0e-03               = \n
[ Info: R2: terminating with √(ξ/ν) = 1.72725123540098540832733280958833139e+03 \n
"Execution stats: first-order stationary" \n
"""
function iR2Reg(
  nlp::AbstractNLPModel{R, V},
  h,
  options::ROSolverOptions{R},
  params::iR2RegParams;
  kwargs...) where {R <: Real, V}
  
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:nlp.meta.nvar)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  
  # Clone the initial parameters
  initial_params = clone_params(params)
  
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  
  stats = iR2Reg(
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
      ν = params.ν,
      γ = options.γ,
  )
  
  # Check if parameters have changed, and if not, reset them
  if params == initial_params
      params = clone_params(initial_params)
      # TODO : check if we need to reset the parameters in the solver
  end
  
  return stats
end

function iR2Reg(reg_nlp::AbstractRegularizedNLPModel, params::iR2RegParams, options::ROSolverOptions; kwargs...)
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
  set_solver_specific!(stats, :π_Hist, solver.p_hist)
  set_solver_specific!(stats, :eval_counters, solver.special_counters)
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
  h = solver.h
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
    solver.special_counters[:prox][p.ph] += 1
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
      hdr_override = Dict{Symbol,String}( 
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

  p.σk = max(1 / p.ν, options.σmin)
  
  p.ν = 1 / p.σk
  sqrt_ξ_νInv = Π[end](1.0)

  fxk = obj(nlp, solver.xk[p.pf]) 
  while (any(isnan, solver.fk[p.pf]) || any(isinf, solver.fk[p.pf])) && activate_mp
    if p.pf == P
      @error "Reached max precision on f at initial point. Early stopping iR2-Reg."
    end
    @warn "Initial objective overflows/underflows. Increasing precision on f."
    p.pf+=1
    fxk = obj(nlp, solver.xk[p.pf])
    solver.special_counters[:f][p.pf] += 1
  end
  solver.special_counters[:f][p.pf] += 1 # on incrémente le compteur de f en la précision pf.
  for i=1:P
    solver.fk[i] = Π[i](fxk) # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
  end 

  grad!(nlp, solver.xk[p.pg], solver.gfk[p.pg])
  while (any(isnan, solver.gfk[p.pg]) || any(isinf, solver.gfk[p.pg])) && activate_mp
    if p.pg == P
      @error "Reached max precision on ∇f at initial point. Early stopping iR2-Reg."
    end
    @warn "Initial gradient overflows/underflows. Increasing precision on g."
    p.pg+=1
    grad!(nlp, solver.xk[p.pg], solver.gfk[p.pg])
    solver.special_counters[:∇f][p.pg] += 1
  end

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
  solver.ψ = shifted(solver.h, solver.xk[p.ps]) # therefore ψ FP format is s FP format
  φk(d) = dot(solver.gfk[p.ps], d)   
  mk(d) = φk(d) + solver.ψ(d)

  prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps](p.ν))
  while (any(isnan, solver.sk[p.ps]) || any(isinf, solver.sk[p.ps])) && activate_mp
    if p.ps == P
      @error "Reached max precision on s at initial point. Early stopping iR2-Reg."
    end
    @warn "Initial proximal step overflows/underflows. Increasing precision on s."
    recompute_prox!(nlp, solver, p, 0, Π)
  end
  solver.special_counters[:prox][p.ps] += 1
  for i=1:P
    solver.sk[i] .= solver.sk[p.ps]
  end

  mks = mk(solver.sk[p.ps]) # on evite les casts en mettant tout en la précision de s

  solver.ξ = p.H(solver.hk[p.ps]) - p.H(mks) + p.H(max(1, abs(p.H(solver.hk[p.ps]))) * 10 * eps(p.H)) # on evite les casts en mettant tout en la précision de s. Ensuite, on cast tout en H pour éviter les erreurs d'arrondis.

  stats.iter > 1 && (solver.ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(solver.ξ)")) # on check après la première itération car parfois Float16 crée beaucoup d'erreurs d'arrondis
  sqrt_ξ_νInv = solver.ξ ≥ 0 ? sqrt(solver.ξ / p.ν) : sqrt(-solver.ξ / p.ν)
  ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative

  # first check of accuracy conditions here:
  if p.activate_mp
    test_condition_f!(nlp, solver, p, Π, stats.iter)
    test_condition_h!(nlp, solver, p, Π, stats.iter)
    test_condition_∇f!(nlp, solver, p, Π, stats.iter)
    test_assumption_6!(nlp, solver, options, p, Π, stats.iter)
  end
  set_solver_specific!(stats, :xi, sqrt_ξ_νInv)

  solved = (solver.ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (solver.ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ * sqrt(p.κξ))
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
      solver.xkn .= solver.xk[end] .+ solver.sk[end] 
      fkn = obj(nlp, solver.xkn)
      solver.special_counters[:f][p.pf] += 1
      hkn = @views h(solver.xkn[selected])
      solver.special_counters[:h][end] += 1
      improper = (hkn == -Inf)

      Δobj = (p.H(solver.fk[end]) + p.H(solver.hk[end])) - (p.H(fkn) + p.H(hkn)) + max(1, abs(p.H(solver.fk[end]) + p.H(solver.hk[end]))) * 10 * eps(p.H) # casté en haute précision pour éviter les erreurs d'arrondis
      global ρk = Δobj / solver.ξ  # ρk est en la precision de Δobj donc H

      verbose > 0 && 
      stats.iter % verbose == 0 &&
      @info log_row(Any[stats.iter, solver.fk[end], solver.hk[end], sqrt_ξ_νInv, ρk, p.σk, norm(solver.xk[end]), norm(solver.sk[end]), (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")], colsep = 1)

      if η1 ≤ ρk < Inf # success
        solver.xk[p.ps] .= solver.xkn
        if has_bnds #TODO
          @error "Not implemented yet"
          @. l_bound_m_x = l_bound - xk[end]
          @. u_bound_m_x = u_bound - xk[end]
          set_bounds!(solver.ψ, l_bound_m_x, u_bound_m_x)
        end
        grad!(nlp, solver.xk[p.pg], solver.gfk[p.pg])
        solver.special_counters[:∇f][p.pg] += 1
        shift!(solver.ψ, solver.xk[p.ps])
        for i=1:P
          solver.xk[i] .= solver.xk[p.ps] # on met à jour fk en les précisions de Π. Exemple : fxk est en float16 à l'itération 0, on caste fxk en float32 et float64 pour les autres précisions et on les ajoute à fk
          solver.fk[i] = Π[i](fkn)
          solver.hk[i] = Π[i](hkn)
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
        solver.sk[i] .= solver.sk[p.ps] 
      end


      set_objective!(stats, T(solver.fk[end] + solver.hk[end]))
      set_solver_specific!(stats,:smooth_obj, solver.fk[end])
      set_solver_specific!(stats,:nonsmooth_obj, solver.hk[end])
      set_solver_specific!(stats,:π_Hist, p_hist)
      set_iter!(stats, stats.iter + 1)
      set_time!(stats, time() - start_time)

      # new step starts here
      φk(d) = dot(solver.gfk[p.ps], d)   
      mk(d) = φk(d) + solver.ψ(d)
      prox!(solver.sk[p.ps], solver.ψ, solver.mν∇fk[p.ps], Π[p.ps](p.ν))
      solver.special_counters[:prox][p.ps] += 1
      mks = mk(solver.sk[p.ps]) # on evite les casts en mettant tout en la précision de s
      solver.ξ = p.H(solver.hk[p.ps]) - p.H(mks) + p.H(max(1, abs(p.H(solver.hk[p.ps]))) * 10 * eps(p.H)) # on evite les casts en mettant tout en la précision de s. Ensuite, on cast tout en H pour éviter les erreurs d'arrondis.

      if p.activate_mp
        test_condition_f!(nlp, solver, p, Π, stats.iter)
        test_condition_h!(nlp, solver, p, Π, stats.iter)
        test_condition_∇f!(nlp, solver, p, Π, stats.iter)
        test_assumption_6!(nlp, solver, options, p, Π, stats.iter)
      end

      sqrt_ξ_νInv = solver.ξ ≥ 0 ? sqrt(solver.ξ / p.ν) : sqrt(-solver.ξ / p.ν)
      solved = (solver.ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (solver.ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ * sqrt(p.κξ))
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


#TODOs: 
# 1. Implement bound constraints
# 2. Implement other regularizers
# 3. Implement parameters reinitialization