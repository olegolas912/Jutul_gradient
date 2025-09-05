using Jutul, JutulDarcy, GeoEnergyIO
using Statistics: mean
using ForwardDiff: value

# -----------------------------------------------------------
# Глобальные флаги
# -----------------------------------------------------------
USE_FASTMATH = false  # ускорение за счёт @fastmath (возможна потеря точности на уровне УЛП)
PLOT = true           # включить графики (GLMakie)

if PLOT
    using GLMakie
end

# -----------------------------------------------------------
# 1) Инициализация модели SPE1 и «истинного» эталона
# -----------------------------------------------------------

# Полный путь к входному файлу примера SPE1.
const data_pth  = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")

# Исходный словарь данных (как прочитал GeoEnergyIO.parse_data_file).
const data      = parse_data_file(data_pth)

# Рабочая копия словаря данных, которую безопасно мутировать без deepcopy в цикле.
const data_work = deepcopy(data)

# Истинная (эталонная) пористость из сетки SPE1.
const x_truth   = only(unique(data["GRID"]["PORO"]))

# Для наглядности: «истинные» параметры (сейчас один скаляр poro).
const prm_truth = Dict("poro" => x_truth)

# --- Служебные «геттеры» давления для прямого и adjoint-состояний ---
# s[:Pressure] — давление на прямом прогоне (forward).
@inline _p_fwd(s) = s[:Pressure]
# s[:Reservoir][:Pressure] — давление в структуре состояния для adjoint.
@inline _p_adj(s) = s[:Reservoir][:Pressure]

# -----------------------------------------------------------
# 2) Построение кейса с заданной пористостью
# -----------------------------------------------------------

"""
update_poro!(d, poro) :: Dict

Назначает всем ячейкам сетки пористость `poro` в словаре данных `d`.
Мутирует входной словарь (без аллокаций) и возвращает его же.
"""
@inline function update_poro!(d::Dict, poro::Float64)
    fill!(d["GRID"]["PORO"], poro)
    return d
end

"""
build_case_with_poro(poro) :: Case

Заполняет `data_work` заданной пористостью и строит из него кейс Jutul.
Используется в цикле обучения для быстрого перестроения модели под новое `poro`.
"""
@inline function build_case_with_poro(poro::Float64)
    update_poro!(data_work, poro)
    return setup_case_from_data_file(data_work)
end

# --- Истинный прогон для формирования «таргета» (давления по времени) ---
const case_true = build_case_with_poro(x_truth)

# states_true :: Vector{State} — последовательность состояний на отчётных шагах
# (используем как целевые давления, к которым подгоняемся).
const (_, states_true) = simulate_reservoir(case_true)


# -----------------------------------------------------------
# 3) Временная сетка и масштабы нормировки лосса
# -----------------------------------------------------------

# Базовый «немодифицируемый» кейс — используем только для извлечения временной сетки.
const case_base  = setup_case_from_data_file(data)

# step_times[k] — накопленное время к концу k-го шага (секунды в SI).
const step_times = cumsum(case_base.dt)

# total_time — полная длительность моделирования; inv_total_time — её обратная.
const total_time     = sum(case_base.dt)
const inv_total_time = 1.0 / total_time

# inv_scale2 — масштаб нормировки (давления в (100 бар)^2 → безразмерность).
const inv_scale2 = 1.0 / ( (si_unit(:bar) * 100.0)^2 )

# -----------------------------------------------------------
# 4) Вспомогательные функции
# -----------------------------------------------------------

"""
nearest_report_index(t) :: Int

Возвращает индекс ближайшего отчётного шага к моменту времени `t` (в тех же единицах, что и `step_times`).
Используем, чтобы сопоставлять «необязательно совпадающий» момент внутри мини-шага с ближайшим сохранённым состоянием.
"""
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

"""
pdiff(p, p0) :: Float64

Квадратичная невязка давления Σ (p - p0)^2 без нормировки (быстро и без аллокаций).
При `USE_FASTMATH = true` включает @fastmath и SIMD.
"""
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

# _loss_acc[] :: Float64 — аккумулируем скалярный loss по всем мини-шагам (только primal-часть).
const _loss_acc = Ref(0.0)

"""
mismatch_objective_report(m, s, dt, step_info, forces) :: Dual или Float64

Adjoint-объектив, вызываемый на каждом мини-шаге интегратора.
- Вычисляет время t = time + dt, находит ближайший отчётный шаг k.
- Получает давления: p = _p_adj(s) (текущие), p0 = _p_fwd(states_true[k]) (таргет).
- Считает вклад contrib = (dt / total_time) * Σ(p - p0)^2 / (100 бар)^2.
Возвращает contrib как Dual (сохраняем производную), а в `_loss_acc` кладёт его primal-значение.
"""
@inline function mismatch_objective_report(m, s, dt, step_info, forces)
    # t :: Float64 — момент конца мини-шага; k :: Int — ближайший отчётный шаг
    t = value(step_info[:time]) + dt
    k = nearest_report_index(t)

    # v :: Float64 — квадратичная невязка; contrib :: Dual/Float64 — вклад в loss
    v = pdiff(_p_adj(s), _p_fwd(states_true[k]))
    contrib = dt * v * inv_scale2 * inv_total_time

    # В аккумулятор добавляем только численное значение (без Dual-части).
    _loss_acc[] += value(contrib)
    return contrib
end

# -----------------------------------------------------------
# 6) Извлечение скалярного градиента ∂L/∂poro из словаря adjoint-а
# -----------------------------------------------------------

"""
_get_poro_grad_scalar(grad_dict) :: Float64

Находит в словаре `grad_dict` градиент по пористости (вектор по ячейкам),
берёт среднее по ячейкам (устойчивее, чем сумма) и возвращает скаляр.
Учитывает разные возможные ключи (:model / "model", :porosity / "porosity" / :PORO / "PORO").
"""
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
# 7) Оценка лосса и градиента для заданного poro
# -----------------------------------------------------------

"""
evaluate_loss_and_grad(poro) :: (L::Float64, g::Float64)

Строит кейс с пористостью `poro`, размораживает параметры, запускает adjoint с
`mismatch_objective_report`, возвращает:
- L — суммарный loss (скаляр),
- g — скалярный градиент ∂L/∂poro (усреднённый по ячейкам).
"""
function evaluate_loss_and_grad(poro::Float64)
    case_c    = build_case_with_poro(poro)                  # кейс с текущей пористостью
    dprm_case = setup_reservoir_dict_optimization(case_c)   # обёртка параметров
    free_optimization_parameters!(dprm_case)                # делаем параметры свободными

    _loss_acc[] = 0.0                                       # сброс аккумулятора
    gdict = parameters_gradient_reservoir(dprm_case, mismatch_objective_report)

    # Возвращаем накопленный скалярный loss и усреднённый градиент по пористости.
    return _loss_acc[], _get_poro_grad_scalar(gdict)
end

# -----------------------------------------------------------
# 8) Простейший градиентный цикл по одному параметру poro
# -----------------------------------------------------------

"""
train_boosting_scalar!(poro0; kwargs...) :: NamedTuple

Мини-оптимизация одного скаляра `poro` по схеме:
  poro_{k+1} = clamp(poro_k - η_k * g_k, [poro_min, poro_max])

Параметры:
- η            — шаг (может подстраиваться правилом Барзилая–Борувайна при use_bb=true)
- nrounds      — максимум итераций
- tol_target   — точность по попаданию в целевое значение poro_star (если задано)
- tol_step     — критерий остановки по длине шага |Δ|
- poro_star    — опциональный «истинный» таргет пористости

Возвращает:
- (losses, poros, poro_final)
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

    losses = Vector{Float64}(undef, nrounds)
    poros  = Vector{Float64}(undef, nrounds)
    K = 0

    poro = clamp(poro0, poro_min, poro_max)

    for k in 1:nrounds
        Lk, gk = evaluate_loss_and_grad(poro)

        # фиксированный шаг
        step     = -η * gk
        poro_new = clamp(poro + step, poro_min, poro_max)

        K += 1
        losses[K] = Lk
        poros[K]  = poro_new

        if verbose
            @info "iter=$(k)  loss=$(round(Lk, sigdigits=7))  poro=$(round(poro_new, sigdigits=8))  step=$(round(step, sigdigits=6))  grad=$(round(gk, sigdigits=6))  η=$(round(η, sigdigits=6))"
        end

        if abs(step) < tol_step
            verbose && @info "Остановка: |Δ| < $tol_step"
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

    return (losses=losses[1:K], poros=poros[1:K], poro_final=poro)
end

# -----------------------------------------------------------
# 9) Запуск эксперимента и графики
# -----------------------------------------------------------

# poro0 — стартовое значение (берём 80% от «истины» и ограничиваем)
poro0   = clamp(x_truth * 0.80, 1e-6, 1.0)
η       = 1e-4          # начальный шаг
nrounds = 10            # число итераций

# result.losses — траектория лосса; result.poros — траектория poro; result.poro_final — итог
result = train_boosting_scalar!(poro0; η=1e-4, nrounds=nrounds,
                                tol_target=1e-8, poro_star=x_truth,
                                poro_min=1e-6, poro_max=1.0,
                                verbose=true)


@info "Итог: poro_final=$(result.poro_final)  truth=$(x_truth)  abs.err=$(abs(result.poro_final - x_truth))"

if PLOT
    fig1 = Figure(size=(900, 320))
    ax1  = Axis(fig1[1,1], title="Loss vs iteration", xlabel="iteration", ylabel="loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    display(fig1)

    fig2 = Figure(size=(900, 320))
    ax2  = Axis(fig2[1,1], title="Porosity vs iteration", xlabel="iteration", ylabel="poro")
    lines!(ax2, 1:length(result.poros), result.poros)
    display(fig2)
end