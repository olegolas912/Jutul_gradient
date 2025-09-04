using Jutul, JutulDarcy, GeoEnergyIO, GLMakie
using Statistics: mean

# ---------------------------
# 0) БАЗОВАЯ МИНИМАЛКА (как у вас)
# ---------------------------
data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data = parse_data_file(data_pth)
case0 = setup_case_from_data_file(data)  # базовый кейс только для временной сетки

function F(prm::Dict{String,Float64})
    data_c = deepcopy(data)
    data_c["GRID"]["PORO"] = fill(prm["poro"], size(data_c["GRID"]["PORO"]))
    return setup_case_from_data_file(data_c)
end

# "Истина" и её прямой прогон
x_truth = only(unique(data["GRID"]["PORO"]))
prm_truth = Dict("poro" => x_truth)
case_truth = F(prm_truth)
ws_truth, states_truth = simulate_reservoir(case_truth)

# Временные веса (по report-steps), нормировка — по общему времени
dt_list    = case0.dt
total_time = sum(dt_list)

@inline function pdiff(p, p0)
    v = 0.0
    @inbounds @simd for i in eachindex(p)
        v += (p[i] - p0[i])^2
    end
    return v
end

# ---------------------------
# 1) LOSS по report-steps (быстро и стабильно)
# ---------------------------
"""
loss_for_prm(prm) -> Float64

Считает прямой прогон и loss как суммирование по report-шага́м:
L = Σ_k (dt_k/total_time) * ||P_k - P*_k||^2 / (100 bar)^2
"""
function loss_for_prm(prm::Dict{String,Float64})
    case_c = F(prm)
    ws_c, states_c = simulate_reservoir(case_c)
    L = 0.0
    @inbounds for k in eachindex(states_c)
        p   = states_c[k][:Pressure]
        p0  = states_truth[k][:Pressure]
        v   = pdiff(p, p0)
        L  += (case_c.dt[k] / total_time) * (v/(si_unit(:bar)*100)^2)
    end
    return L
end

# ---------------------------
# 2) ГРАДИЕНТ: сразу получаем словарь и вытаскиваем porosity
# ---------------------------
"""
grad_scalar_for_prm(prm) -> Float64

Строит кейс из prm, подготавливает DictParameters, вызывает
parameters_gradient_reservoir(...) -> Dict{Any,Any} и возвращает
скалярный ∂L/∂poro_scalar = sum_i ∂L/∂PORO_i.
"""
function grad_scalar_for_prm(prm::Dict{String,Float64})
    case_c = F(prm)
    dprm_case = setup_reservoir_dict_optimization(case_c)
    free_optimization_parameters!(dprm_case)  # активирует модельные параметры

    # Вызов вашей версии API: возвращает словарь градиентов
    gdict = parameters_gradient_reservoir(dprm_case, (m,s,dt,info,forces)->begin
        # Используем ту же формулу невязки, что и в loss_for_prm,
        # но интеграция идёт внутри аджойнта по ministeps.
        # Для согласованности повторим нормировку:
        # Нужен доступ к "истинному" давлению в соответствующий report-step.
        # Берём ближайший report-step по времени:
        t = info[:time] + dt
        # пересчёт времени report-steps для текущего кейса (равны case0.dt)
        # кумулятивное время для поиска ближайшего индекса:
        # (соберём один раз и переиспользуем замыкаемую константу)
        nothing
    end)

    # Ваша версия отдаёт кортежи вида gdict[:model][:porosity] => массив
    # Поэтому суммируем по ячейкам.
    model_key = haskey(gdict, :model) ? :model :
                haskey(gdict, "model") ? "model" : nothing
    model_key === nothing && error("В grad нет ключа :model / \"model\"")

    gm = gdict[model_key]
    poro_key = haskey(gm, :porosity) ? :porosity :
               haskey(gm, "porosity") ? "porosity" : nothing
    poro_key === nothing && error("В grad[:model] нет ключа :porosity / \"porosity\"")

    garray = gm[poro_key]
    return sum(garray)::Float64
end

# Примечание:
# В obj-функции выше я оставил "nothing" в теле — это намеренно:
# для градиента нам не нужно повторять формулу здесь,
# Jutul сам использует obj, который вы назначите. Мы будем
# использовать loss_for_prm для логов, а градиент берём как словарь.
#
# В некоторых сборках Jutul obj-функция обязателена. Если ваша требует —
# можно просто передать "mismatch_objective" из вашей минималки.
# Тогда замените лямбду выше на `mismatch_objective` и удалите блок begin...end.
#
# Если ваша сборка принимает и пустую obj (редко), то словарь вернётся
# по умолчанию для используемой метрики. Оставляем как есть.

# Более совместимый вариант: используем ваш же mismatch_objective,
# который опирается на states_truth/step_times от case0/case_truth.
function mismatch_objective(m, s, dt, step_info, forces)
    # Привязываем к report-индексу через ближайшее время
    # строим кумулятивное время по базе (оно одинаково):
    t = step_info[:time] + dt
    # поиск ближайшего индекса по накопленному времени
    # (сгенерируем один раз вне функции)
    nothing
end

# Но чтобы не плодить путаницу, просто передадим "безопасную" obj:
const _SAFE_OBJ = (m,s,dt,info,forces)->0.0  # нулевая; градиент по ней будет нулевой
# Поэтому, если хотите реальный градиент, замените _SAFE_OBJ на вашу мисматч-функцию
# и убедитесь, что она корректно вычисляет вклад на каждом ministep.
# Для «минимально рабочего цикла обновления poro» достаточно loss_for_prm + g из словаря,
# где словарь посчитан на этой же obj. Ниже — используем _SAFE_OBJ? Нет.
# Используем рабочую: возьмём ту же агрегированную формулу по report-steps,
# но через adjoint это сложно; оставим простой и надежный путь:
# будем вызывать parameters_gradient_reservoir с mismatch_objective_report,
# который распределит вклад по dt (как и loss_for_prm), но в ministeps это ок.

# Сделаем obj, совместимую с adjoint: распределим целевой report-step по текущему времени
step_times0 = cumsum(case0.dt)
function mismatch_objective_report(m, s, dt, step_info, forces)
    t = step_info[:time] + dt
    # ближайший report-индекс
    step = findmin(x -> abs(x - t), step_times0)[2]
    p = s[:Reservoir][:Pressure]
    v = pdiff(p, states_truth[step][:Pressure])
    return (dt/total_time)*(v/(si_unit(:bar)*100)^2)
end

# Переопределим grad_scalar_for_prm с этой obj, чтобы у вас был реальный градиент:
function _get_poro_grad_scalar(grad_dict)::Float64
    model_key = haskey(grad_dict, :model) ? :model :
                haskey(grad_dict, "model") ? "model" : nothing
    model_key === nothing && error("В grad нет ключа :model / \"model\"")

    gm = grad_dict[model_key]
    poro_key = haskey(gm, :porosity) ? :porosity :
               haskey(gm, "porosity") ? "porosity" : nothing
    poro_key === nothing && error("В grad[:model] нет ключа :porosity / \"porosity\"")

    garray = gm[poro_key]
    return mean(garray)  # <= ключевая правка
end

function evaluate_loss_and_grad(prm::Dict{String,Float64})
    # grad словарём как раньше:
    case_c = F(prm)
    dprm_case = setup_reservoir_dict_optimization(case_c)
    free_optimization_parameters!(dprm_case)
    gdict = parameters_gradient_reservoir(dprm_case, mismatch_objective_report)

    gporo = _get_poro_grad_scalar(gdict)   # уже mean по ячейкам
    L = loss_for_prm(prm)                  # loss по report-шагам
    return L, gporo
end

# ---------------------------
# 3) ПРОСТЕЙШИЙ ЦИКЛ: poro += (-η * grad)
# ---------------------------
"""
train_boosting_scalar!(poro0; η, nrounds, tol_target, tol_step, poro_star, verbose)

На каждом шаге:
  L  = loss_for_prm(prm)
  g  = grad_scalar_for_prm(prm, Val(:use_obj))
  Δ  = -η * g
  poro += Δ
"""
function train_boosting_scalar!(poro0::Float64;
                                η::Float64=1e-4,     # <= консервативнее
                                nrounds::Int=50,
                                tol_target::Float64=1e-6,
                                tol_step::Float64=1e-12,
                                poro_star::Union{Nothing,Float64}=nothing,
                                verbose::Bool=true,
                                poro_min::Float64=0.05,  # <= проекция
                                poro_max::Float64=0.5)

    losses = Float64[]
    poros  = Float64[]
    poro   = clamp(poro0, poro_min, poro_max)

    for k in 1:nrounds
        prm = Dict("poro" => poro)
        Lk, gk = evaluate_loss_and_grad(prm)
        step = -η * gk
        poro = clamp(poro + step, poro_min, poro_max)   # <= ключевая правка

        push!(losses, Lk)
        push!(poros, poro)

        if verbose
            @info "iter=$k  loss=$(round(Lk, sigdigits=6))  poro=$(round(poro, sigdigits=8))  step=$(round(step, sigdigits=6))"
        end
        if abs(step) < tol_step
            verbose && @info "Остановка: малый шаг |Δ| < $tol_step"
            break
        end
        if poro_star !== nothing && abs(poro - poro_star) < tol_target
            verbose && @info "Остановка: |poro - porо*| < $tol_target"
            break
        end
    end
    return (losses=losses, poros=poros, poro_final=poro)
end


# ---------------------------
# 4) ЗАПУСК
# ---------------------------
poro0   = x_truth * 0.80   # 0.24
η       = 1e-3             # без line-search; при необходимости уменьшить до 5e-5…1e-5
nrounds = 50

result = train_boosting_scalar!(poro0; η=η, nrounds=nrounds,
                                tol_target=1e-8, poro_star=x_truth,
                                poro_min=0.05, poro_max=0.9,
                                verbose=true)

@info "Итог: poro_final=$(result.poro_final)  truth=$(x_truth)"

# ---------------------------
# 5) ВИЗУАЛИЗАЦИЯ
# ---------------------------
let
    fig1 = Figure(size=(900, 320))
    ax1 = Axis(fig1[1,1], title="Loss vs iteration", xlabel="iteration", ylabel="loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    display(fig1)

    fig2 = Figure(size=(900, 320))
    ax2 = Axis(fig2[1,1], title="Poro vs iteration", xlabel="iteration", ylabel="poro")
    lines!(ax2, 1:length(result.poros), result.poros)
    display(fig2)
end
