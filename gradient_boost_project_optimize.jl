using Jutul, JutulDarcy, GeoEnergyIO
using Statistics: mean
using ForwardDiff: value

# -----------------------------------------------------------
# Глобальные флаги
# -----------------------------------------------------------
USE_FASTMATH     = false  # ускорение за счёт @fastmath (потеря точности на уровне УЛП)
PLOT             = true   # включить графики (GLMakie)
PROFILE_SEPARATE = true   # мерить вперед и назад раздельно (доп. прогон forward)

if PLOT
    using GLMakie
end

# -----------------------------------------------------------
# 1) Инициализация модели SPE1 и «истинного» эталона
# -----------------------------------------------------------

# const data_pth  = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
const data_pth  = joinpath(GeoEnergyIO.test_input_file_path("EGG"), "EGG.DATA")

const data      = parse_data_file(data_pth)
const data_work = deepcopy(data)

const x_truth   = only(unique(data["GRID"]["PORO"]))
const prm_truth = Dict("poro" => x_truth)

fmt3(x) = isnan(x) ? "NaN" : string(round(x, digits=3))

@inline _p_fwd(s) = s[:Pressure]                 # давление прямого прогона
@inline _p_adj(s) = s[:Reservoir][:Pressure]     # давление adjoint

# -----------------------------------------------------------
# 2) Построение кейса с заданной пористостью
# -----------------------------------------------------------

@inline function update_poro!(d::Dict, poro::Float64)
    fill!(d["GRID"]["PORO"], poro)
    return d
end

@inline function build_case_with_poro(poro::Float64)
    update_poro!(data_work, poro)
    return setup_case_from_data_file(data_work)
end

# --- Истинный прогон для формирования «таргета» (давления по времени) ---
const case_true = build_case_with_poro(x_truth)

# измеряем время и достаем состояния из результата
t_fwd_true = 0.0
res_true = nothing
t_fwd_true = @elapsed begin
    res_true = simulate_reservoir(case_true)
end
@info "Прямой прогон (истина): time_fwd_true = $(round(t_fwd_true, digits=3)) s"
# в актуальном API берем вектор состояний из поля результата
states_true = res_true.states

# -----------------------------------------------------------
# 3) Временная сетка и масштабы нормировки лосса
# -----------------------------------------------------------

const case_base  = setup_case_from_data_file(data)
const step_times = cumsum(case_base.dt)
const total_time     = sum(case_base.dt)
const inv_total_time = 1.0 / total_time
const inv_scale2     = 1.0 / ( (si_unit(:bar) * 100.0)^2 )

# -----------------------------------------------------------
# 4) Вспомогательные функции
# -----------------------------------------------------------

@inline function nearest_report_index(t::Float64)
    i = searchsortedfirst(step_times, t)
    if i <= 1
        return 1
    elseif i > length(step_times)
        return length(step_times)
    else
        left = step_times[i-1]; right = step_times[i]
        return (t - left <= right - t) ? (i - 1) : i
    end
end

@inline function pdiff(p, p0)
    s = 0.0
    if USE_FASTMATH
        @inbounds @simd for i in eachindex(p)
            @fastmath s += (p[i] - p0[i])^2
        end
    else
        @inbounds @simd for i in eachindex(p)
            s += (p[i] - p0[i])^2
        end
    end
    return s
end

# -----------------------------------------------------------
# 5) Функция-объектив для adjoint и аккумулятор лосса
# -----------------------------------------------------------

const _loss_acc = Ref(0.0)

@inline function mismatch_objective_report(m, s, dt, step_info, forces)
    t = value(step_info[:time]) + dt
    k = nearest_report_index(t)
    v = pdiff(_p_adj(s), _p_fwd(states_true[k]))
    contrib = dt * v * inv_scale2 * inv_total_time
    _loss_acc[] += value(contrib)
    return contrib
end

# -----------------------------------------------------------
# 6) Извлечение скалярного градиента ∂L/∂poro из словаря adjoint-а
# -----------------------------------------------------------

@inline function _get_poro_grad_scalar(grad_dict)::Float64
    mk = haskey(grad_dict, :model) ? :model :
         haskey(grad_dict, "model") ? "model" : error("no :model in grad")
    gm = grad_dict[mk]
    pk = haskey(gm, :porosity) ? :porosity :
         haskey(gm, "porosity") ? "porosity" :
         haskey(gm, :PORO) ? :PORO :
         haskey(gm, "PORO") ? "PORO" : error("no porosity/PORO in grad[:model]")
    return mean(gm[pk])::Float64
end

# -----------------------------------------------------------
# 7) Оценка лосса, градиента и времени прогона для заданного poro
# -----------------------------------------------------------

"""
evaluate_loss_and_grad(poro) :: (L, g, times)

times :: NamedTuple
    - t_fwd_est    — оценка времени чистого forward (s)
    - t_grad_total — общее время вызова parameters_gradient_reservoir (s)
    - t_adj_est    — оценка времени adjoint = t_grad_total - t_fwd_est (s, неотрицательная)
"""
function evaluate_loss_and_grad(poro::Float64)
    case_c    = build_case_with_poro(poro)
    dprm_case = setup_reservoir_dict_optimization(case_c)
    free_optimization_parameters!(dprm_case)

    # по желанию — чистый forward для оценки времени
    t_fwd_est = PROFILE_SEPARATE ? @elapsed(simulate_reservoir(case_c)) : NaN

    _loss_acc[] = 0.0
    gdict = nothing
    t_grad_total = @elapsed begin
        gdict = parameters_gradient_reservoir(dprm_case, mismatch_objective_report)
    end

    L = _loss_acc[]
    g = _get_poro_grad_scalar(gdict)

    t_adj_est = PROFILE_SEPARATE ? max(t_grad_total - t_fwd_est, 0.0) : NaN
    return L, g, (t_fwd_est=t_fwd_est, t_grad_total=t_grad_total, t_adj_est=t_adj_est)
end


# -----------------------------------------------------------
# 8) Простейший градиентный цикл по одному параметру poro
# -----------------------------------------------------------

"""
train_boosting_scalar!(...) :: NamedTuple

Расширено: возвращает также траектории времен:
- fwd_times[k], adj_times[k], grad_total_times[k]
"""
function train_boosting_scalar!(poro0::Float64;
                                η::Float64=1e-4,
                                nrounds::Int=50,
                                tol_target::Float64=1e-6,
                                tol_step::Float64=1e-12,
                                poro_star::Union{Nothing,Float64}=nothing,
                                verbose::Bool=true,
                                poro_min::Float64=1e-6,
                                poro_max::Float64=1.0,
                                use_bb::Bool=true,
                                η_min::Float64=1e-6,
                                η_max::Float64=1e-2)

    losses = Vector{Float64}(undef, nrounds)
    poros  = Vector{Float64}(undef, nrounds)
    # новые буферы времени
    fwd_times       = Vector{Float64}(undef, nrounds)
    adj_times       = Vector{Float64}(undef, nrounds)
    grad_total_times= Vector{Float64}(undef, nrounds)

    K = 0
    poro      = clamp(poro0, poro_min, poro_max)
    poro_prev = NaN
    gprev     = NaN

    for k in 1:nrounds
        # теперь получаем и метрики времени
        Lk, gk, tms = evaluate_loss_and_grad(poro)

        # адаптивный шаг BB
        if use_bb && k > 1
            s = poro - poro_prev
            y = gk   - gprev
            denom = y*y
            if isfinite(denom) && denom > eps() && isfinite(s)
                η = clamp(abs(s)/abs(y), η_min, η_max)
            end
        end

        step     = -η * gk
        poro_new = clamp(poro + step, poro_min, poro_max)

        K += 1
        losses[K] = Lk
        poros[K]  = poro_new

        # сохраняем время
        fwd_times[K]        = tms.t_fwd_est
        grad_total_times[K] = tms.t_grad_total
        adj_times[K]        = tms.t_adj_est

        if verbose
            @info (
                "iter=$(k)  loss=$(round(Lk, sigdigits=7))  " *
                "poro=$(round(poro_new, sigdigits=8))  " *
                "step=$(round(step, sigdigits=6))  " *
                "grad=$(round(gk, sigdigits=6))  " *
                "η=$(round(η, sigdigits=6))" *
                "  |  t_fwd≈$(fmt3(tms.t_fwd_est)) s" *
                "  t_adj≈$(fmt3(tms.t_adj_est)) s" *
                "  t_grad=$(round(tms.t_grad_total, digits=3)) s"
            )
        end


        # стоп-критерии
        if abs(step) < tol_step
            verbose && @info "Остановка: |Δ| < $tol_step"
            poro = poro_new; break
        end
        if poro_star !== nothing && abs(poro_new - poro_star) < tol_target
            verbose && @info "Остановка: |poro - porо*| < $tol_target"
            poro = poro_new; break
        end

        poro_prev = poro
        gprev     = gk
        poro      = poro_new
    end

    return (
        losses = losses[1:K],
        poros  = poros[1:K],
        poro_final = poro,
        # новые поля
        fwd_times        = fwd_times[1:K],
        adj_times        = adj_times[1:K],
        grad_total_times = grad_total_times[1:K],
        t_fwd_true       = _t_fwd_true
    )
end

# -----------------------------------------------------------
# 9) Запуск эксперимента и графики
# -----------------------------------------------------------

poro0   = clamp(x_truth * 0.80, 1e-6, 1.0)
η       = 1e-4
nrounds = 10

result = train_boosting_scalar!(poro0; η=η, nrounds=nrounds,
                                tol_target=1e-8, poro_star=x_truth,
                                poro_min=1e-6, poro_max=1.0,
                                verbose=true, use_bb=true)

@info "Итог: poro_final=$(result.poro_final)  truth=$(x_truth)  abs.err=$(abs(result.poro_final - x_truth))"
# аккуратно считаем средние по только корректным значениям
fwd_mean  = isempty(result.fwd_times)  ? NaN : mean(filter(isfinite, result.fwd_times))
adj_mean  = isempty(result.adj_times)  ? NaN : mean(filter(isfinite, result.adj_times))
grad_mean = isempty(result.grad_total_times) ? NaN : mean(result.grad_total_times)

@info "Итог по времени: true_forward=$(round(t_fwd_true, digits=3)) s, " *
      "avg_forward≈$(isfinite(fwd_mean) ? round(fwd_mean, digits=3) : NaN) s, " *
      "avg_adjoint≈$(isfinite(adj_mean) ? round(adj_mean, digits=3) : NaN) s, " *
      "avg_grad_total=$(round(grad_mean, digits=3)) s"


if PLOT
    fig1 = Figure(size=(900, 320))
    ax1  = Axis(fig1[1,1], title="Loss vs iteration", xlabel="iteration", ylabel="loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    # display(fig1)

    fig2 = Figure(size=(900, 320))
    ax2  = Axis(fig2[1,1], title="Porosity vs iteration", xlabel="iteration", ylabel="poro")
    lines!(ax2, 1:length(result.poros), result.poros)
    display(fig2)

    fig3 = Figure(size=(900, 360))
    ax3  = Axis(fig3[1,1], title="Timing per iteration", xlabel="iteration", ylabel="seconds")
    lines!(ax3, 1:length(result.grad_total_times), result.grad_total_times, label="grad_total")
    if PROFILE_SEPARATE
        lines!(ax3, 1:length(result.fwd_times), result.fwd_times, label="forward (estimate)")
        lines!(ax3, 1:length(result.adj_times), result.adj_times, label="adjoint (estimate)")
    end
    axislegend(ax3; position=:rt)
    # display(fig3)
end
