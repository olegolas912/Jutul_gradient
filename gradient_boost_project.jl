using Jutul, JutulDarcy, GeoEnergyIO
using Statistics: mean
PLOT = true
if PLOT
    using GLMakie
end

# -----------------------------------------------------------
# 1) База: SPE1, сборщик кейса по словарю prm["poro"]
# -----------------------------------------------------------
data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data = parse_data_file(data_pth)

function F(prm::Dict{String,Float64})
    data_c = deepcopy(data)
    data_c["GRID"]["PORO"] = fill(prm["poro"], size(data_c["GRID"]["PORO"]))
    return setup_case_from_data_file(data_c)
end

# Истина и её прогон
x_truth   = only(unique(data["GRID"]["PORO"]))
prm_truth = Dict("poro" => x_truth)
case_true = F(prm_truth)
ws_true, states_true = simulate_reservoir(case_true)

# Базовый кейс для временной сетки (можно брать case_true)
case_base = setup_case_from_data_file(data)
dt_list    = case_base.dt
total_time = sum(dt_list)
step_times = cumsum(dt_list) # для привязки по времени в adjoint-obj

@inline _p_fwd(s) = s[:Pressure]
@inline _p_adj(s) = s[:Reservoir][:Pressure]


# -----------------------------------------------------------
# 2) Утилиты: доступ к давлению и лосс по репорт-шагам
# -----------------------------------------------------------
@inline function _pressure_from_state(s)
    # Пробуем несколько вариантов, чтобы работать и в forward, и в adjoint.
    try
        return s[:Pressure]
    catch
        try
            return s[:Reservoir][:Pressure]
        catch
            try
                return s[:Reservoir, :Pressure]
            catch e
                error("Не удалось извлечь давление из состояния типа $(typeof(s)). " *
                      "Пробовали [:Pressure], [:Reservoir][:Pressure], [:Reservoir, :Pressure]. " *
                      "Ошибка: $(e)")
            end
        end
    end
end

@inline function pdiff(p, p0)
    v = 0.0
    @inbounds @simd for i in eachindex(p)
        v += (p[i] - p0[i])^2
    end
    return v
end

"""
loss_for_prm(prm) -> Float64

L = Σ_k (dt_k/total_time) * ||P_k - P*_k||^2 / (100 bar)^2
(по репорт-шагам)
"""
function loss_for_prm(prm::Dict{String,Float64})
    case_c = F(prm)
    ws_c, states_c = simulate_reservoir(case_c)
    L = 0.0
    @inbounds for k in eachindex(states_c)
        p  = _pressure_from_state(states_c[k])
        p0 = _pressure_from_state(states_true[k])
        v  = pdiff(p, p0)
        L += (case_c.dt[k] / total_time) * (v/(si_unit(:bar)*100)^2)
    end
    return L
end

# -----------------------------------------------------------
# 3) Объектив для adjoint (на мини-шагах), согласованный с лоссом
# -----------------------------------------------------------
function mismatch_objective_report(m, s, dt, step_info, forces)
    # Привязываем текущий ministep ко "взвешенному" ближайшему репорт-индексу
    diag_once_adjoint(s)
    t = step_info[:time] + dt
    step = findmin(x -> abs(x - t), step_times)[2]
    p  = _p_adj(s)
    p0 = _p_fwd(states_true[step])
    v  = pdiff(p, p0)
    return (dt/total_time) * (v/(si_unit(:bar)*100)^2)
end

# -----------------------------------------------------------
# 4) Извлечение скалярного градиента ∂L/∂poro из словаря adjoint-а
#    (в разных версиях ключи могут быть Symbol/String — учитываем оба)
# -----------------------------------------------------------
@inline function _get_poro_grad_scalar(grad_dict)::Float64
    # В большинстве сборок: grad_dict[:model][:porosity] => Vector per cell
    model_key = haskey(grad_dict, :model) ? :model :
                haskey(grad_dict, "model") ? "model" : nothing
    model_key === nothing && error("grad нет ключа :model / \"model\"")

    gm = grad_dict[model_key]
    poro_key = haskey(gm, :porosity) ? :porosity :
               haskey(gm, "porosity") ? "porosity" :
               haskey(gm, :PORO) ? :PORO :
               haskey(gm, "PORO") ? "PORO" : nothing
    poro_key === nothing && error("grad[:model] нет :porosity / \"porosity\" / PORO")

    garray = gm[poro_key]
    # Берём среднее по ячейкам — масштаб стабильнее, чем сумма.
    return mean(garray)::Float64
end

"""
evaluate_loss_and_grad(prm) -> (L, g)

Строит кейс, активирует параметры, вызывает adjoint и возвращает:
  L (через loss_for_prm) и g = ∂L/∂poro (скаляр).
"""
function evaluate_loss_and_grad(prm::Dict{String,Float64})
    case_c = F(prm)
    dprm_case = setup_reservoir_dict_optimization(case_c)
    free_optimization_parameters!(dprm_case)  # активируем модельные параметры

    gdict = parameters_gradient_reservoir(dprm_case, mismatch_objective_report)
    gporo = _get_poro_grad_scalar(gdict)
    L = loss_for_prm(prm)
    return L, gporo
end

# -----------------------------------------------------------
# 5) ПРОСТЕЙШИЙ цикл обучения: poro ← poro − η * g
# -----------------------------------------------------------
"""
train_boosting_scalar!(poro0; η, nrounds, tol_target, tol_step, poro_star, ...)

На каждой итерации:
  (L, g) = evaluate_loss_and_grad(prm)
  poro  += (−η * g), с проекцией в [poro_min, poro_max]
"""
function train_boosting_scalar!(poro0::Float64;
                                η::Float64=1e-4,
                                nrounds::Int=50,
                                tol_target::Float64=1e-6,
                                tol_step::Float64=1e-12,
                                poro_star::Union{Nothing,Float64}=nothing,
                                verbose::Bool=true,
                                poro_min::Float64=1e-6,
                                poro_max::Float64=1.0)

    losses = Float64[]
    poros  = Float64[]
    poro   = clamp(poro0, poro_min, poro_max)

    for k in 1:nrounds
        prm = Dict("poro" => poro)
        Lk, gk = evaluate_loss_and_grad(prm)
        step = -η * gk
        poro_new = clamp(poro + step, poro_min, poro_max)

        push!(losses, Lk)
        push!(poros, poro_new)

        if verbose
            @info "iter=$k  loss=$(round(Lk, sigdigits=6))  poro=$(round(poro_new, sigdigits=8))  step=$(round(step, sigdigits=6))  grad=$(round(gk, sigdigits=6))"
        end

        # Критерии остановки
        if abs(step) < tol_step
            verbose && @info "Остановка: малый шаг |Δ| < $tol_step"
            poro = poro_new
            break
        end
        if poro_star !== nothing && abs(poro_new - poro_star) < tol_target
            verbose && @info "Остановка: |poro - porо*| < $tol_target"
            poro = poro_new
            break
        end

        poro = poro_new
    end

    return (losses=losses, poros=poros, poro_final=poro)
end

# -----------------------------------------------------------
# 6) Запуск
# -----------------------------------------------------------
poro0   = clamp(x_truth * 0.80, 1e-6, 1.0)  # старт (сдвигаем от истины)
η       = 1e-3                               # без line-search; подбирай: 1e-5…1e-3
nrounds = 50

result = train_boosting_scalar!(poro0; η=η, nrounds=nrounds,
                                tol_target=1e-8, poro_star=x_truth,
                                poro_min=1e-6, poro_max=1.0,
                                verbose=true)

@info "Итог: poro_final=$(result.poro_final)  truth=$(x_truth)  abs.err=$(abs(result.poro_final - x_truth))"

# -----------------------------------------------------------
# 7) Графики (по желанию)
# -----------------------------------------------------------
if PLOT
    fig1 = Figure(size=(900, 320))
    ax1 = Axis(fig1[1,1], title="Loss vs iteration", xlabel="iteration", ylabel="loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    display(fig1)

    fig2 = Figure(size=(900, 320))
    ax2 = Axis(fig2[1,1], title="Porosity vs iteration", xlabel="iteration", ylabel="poro")
    lines!(ax2, 1:length(result.poros), result.poros)
    display(fig2)
end
