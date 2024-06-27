module iR2Solver

export iR2_lazy, iR2Solver, solve!

import SolverCore.solve!

mutable struct iR2Solver{R <: Real, G <: Union{ShiftedProximableFunction, Nothing}, S <: AbstractVector{R}} <: AbstractOptimizationSolver
    xk::S
    ∇fk::S
    mν∇fk::S
    ψ::G
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
end

function iR2Solver(
    x0::S,
    options::ROSolverOptions,
    l_bound::S,
    u_bound::S;
    ψ = nothing
) where {R <: Real, S <: AbstractVector{R}}
    maxIter = options.maxIter
    xk = similar(x0)
    ∇fk = similar(x0)
    mν∇fk = similar(x0)
    xkn = similar(x0)
    s = zero(x0)
    has_bnds = any(l_bound .!= R(-Inf)) || any(u_bound .!= R(Inf))
    if has_bnds
        l_bound_m_x = similar(xk)
        u_bound_m_x = similar(xk)
    else
        l_bound_m_x = similar(xk, 0)
        u_bound_m_x = similar(xk, 0)
    end
    Fobj_hist = zeros(R, maxIter + 2)
    Hobj_hist = zeros(R, maxIter + 2)
    Complex_hist = zeros(Int, maxIter + 2)
    return iR2Solver(
        xk,
        ∇fk,
        mν∇fk,
        ψ,
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
    )
end

function iR2Solver(
    reg_nlp::AbstractRegularizedNLPModel{T, V};
    max_iter::Int = 10000
) where {T, V}
    x0 = reg_nlp.model.meta.x0
    l_bound = reg_nlp.model.meta.lvar
    u_bound = reg_nlp.model.meta.uvar

    xk = similar(x0)
    ∇fk = similar(x0)
    mν∇fk = similar(x0)
    xkn = similar(x0)
    s = zero(x0)
    has_bnds = any(l_bound .!= T(-Inf)) || any(u_bound .!= T(Inf))
    if has_bnds
        l_bound_m_x = similar(xk)
        u_bound_m_x = similar(xk)
        @. l_bound_m_x = l_bound - x0
        @. u_bound_m_x = u_bound - x0
    else
        l_bound_m_x = similar(xk, 0)
        u_bound_m_x = similar(xk, 0)
    end
    Fobj_hist = zeros(T, max_iter + 2)
    Hobj_hist = zeros(T, max_iter + 2)
    Complex_hist = zeros(Int, max_iter + 2)
    
    ψ = has_bnds ? shifted(reg_nlp.h, x0, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, x0)
    return iR2Solver(
        xk,
        ∇fk,
        mν∇fk,
        ψ,
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
    )
end

function iR2_lazy(
    nlp::AbstractNLPModel, 
    args...; 
    kwargs...
)
    kwargs_dict = Dict(kwargs...)
    x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
    x, k, outdict = iR2_lazy(
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
    set_objective!(stats, typeof(nlp.meta.x0[1]).(outdict[:fk]) + typeof(nlp.meta.x0[1]).(outdict[:hk]))
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

function iR2_lazy(
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
    k, status, fk, hk, ξ = iR2_lazy!(solver, f, ∇f!, h, options, x0; selected=selected, params=params)
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

function iR2_lazy(
    f::F,
    ∇f!::G,
    h::H,
    options::ROSolverOptions{R},
    p::iR2RegParams,
    x0::AbstractVector{S},
    l_bound::AbstractVector{S},
    u_bound::AbstractVector{S};
    selected::AbstractVector{<:Integer} = 1:length(x0),
    kwargs...
) where {F <: Function, G <: Function, H, R <: Real, S <: Real}
    start_time = time()
    elapsed_time = 0.0
    solver = iR2Solver(x0, options, params, l_bound, u_bound,)
    k, status, fk, hk, ξ = iR2_lazy!(solver, f, ∇f!, h, options, params, x0; selected=selected)
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

function solve!(
    solver::iR2Solver{T}, 
    reg_nlp::AbstractRegularizedNLPModel{T, V}, 
    stats::GenericExecutionStats{T, V};
    callback = (args...) -> nothing,
    x::V = reg_nlp.model.meta.x0,
    atol::T = √eps(T),
    rtol::T = √eps(T),
    neg_tol::T = eps(T)^(1 / 4),
    verbose::Int = 0,
    max_iter::Int = 10000,
    max_time::Float64 = 30.0,
    max_eval::Int = -1,
    σmin::T = eps(T),
    η1::T = √√eps(T),
    η2::T = T(0.9),
    ν::T = eps(T)^(1 / 5),
    γ::T = T(3),
) where {T, V}
    # Initialize variables and workspace
    reset!(stats)
    selected = reg_nlp.selected
    h = reg_nlp.h
    nlp = reg_nlp.model
    if !all(solver.xk .== x)
        shift!(solver.ψ, x)
    end
    xk = solver.xk .= x
    ∇fk = solver.∇fk
    mν∇fk = solver.mν∇fk
    ψ = solver.ψ
    xkn = solver.xkn
    s = solver.s
    has_bnds = solver.has_bnds
    if has_bnds
        l_bound = solver.l_bound
        u_bound = solver.u_bound
        l_bound_m_x = solver.l_bound_m_x
        u_bound_m_x = solver.u_bound_m_x
    end

    # initialize parameters
    improper = false
    hk = @views h(xk[selected])
    if hk == Inf
        verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
        prox!(xk, h, xk, one(eltype(x0)))
        hk = @views h(xk[selected])
        hk < Inf || error("prox computation must be erroneous")
        verbose > 0 && @debug "R2: found point where h has value" hk
    end
    improper = (hk == -Inf)

    if verbose > 0
        @info log_header(
            [:iter, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :arrow],
            [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Char],
            hdr_override = Dict{Symbol, String}(
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

    local ξ::T
    σk = max(1 / ν, σmin)
    ν = 1 / σk
    sqrt_ξ_νInv = one(T)

    fk = obj(nlp, xk)
    grad!(nlp, xk, ∇fk)
    @. mν∇fk = -ν * ∇fk

    set_iter!(stats, 0)
    start_time = time()
    set_time!(stats, 0.0)
    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)

    φk(d) = dot(∇fk, d)
    mk(d)::T = φk(d) + ψ(d)::T

    prox!(s, ψ, mν∇fk, ν)
    mks = mk(s)

    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
    atol += rtol * sqrt_ξ_νInv

    solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ atol)

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

    while !done
        xkn .= xk .+ s
        fkn = obj(nlp, xkn)
        hkn = @views h(xkn[selected])
        improper = (hkn == -Inf)

        Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
        global ρk = Δobj / ξ

        verbose > 0 && 
        stats.iter % verbose == 0 &&
            @info log_row(Any[stats.iter, fk, hk, sqrt_ξ_νInv, ρk, σk, norm(xk), norm(s), (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")], colsep = 1)

        if η1 ≤ ρk < Inf
            xk .= xkn
            if has_bnds
                @. l_bound_m_x = l_bound - xk
                @. u_bound_m_x = u_bound - xk
                set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
            end
            fk = fkn
            hk = hkn
            grad!(nlp, xk, ∇fk)
            shift!(ψ, xk)
        end

        if η2 ≤ ρk < Inf
            σk = max(σk / γ, σmin)
        end
        if ρk < η1 || ρk == Inf
            σk = σk * γ
        end

        ν = 1 / σk
        @. mν∇fk = -ν * ∇fk
      
        set_objective!(stats, fk + hk)
        set_solver_specific!(stats, :smooth_obj, fk)
        set_solver_specific!(stats, :nonsmooth_obj, hk)
        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)

        prox!(s, ψ, mν∇fk, ν)
        mks = mk(s)

        ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
        sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
        solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ atol)

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

    verbose > 0 &&
        stats.status == :first_order &&
            @info log_row(Any[stats.iter, fk, hk, sqrt_ξ_νInv, ρk, σk, norm(xk), norm(s), (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")], colsep = 1)
            @info "R2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"

    set_solution!(stats, xk)
    return stats
end

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

end
