#############################
# NORNE_NOHYST: adjoint-GD по поячеечной PORO
# Версия с расширенными метриками времени и счётчиками прогонов.
# - PORO строго по пути [:model][:porosity]
# - Градиент читается из grad_dict[:model][:porosity]
# - Пер-шаговый учёт forward-времени при adjoint-оценке; backward распределяется равномерно
# - Названия артефактов включают модель, QOI и подбираемый параметр
#############################

using Jutul
using JutulDarcy
using GeoEnergyIO
using Statistics: mean, std
using ForwardDiff: value
using Printf
using CUDA

# ---------------------------
# CONFIG / META (для подписей и имён файлов)
# ---------------------------
const MODEL_NAME   = "EGG"
const QOI_BASIS    = "debites_well_rates" # "дебиты"
const TARGET_PARAM = "PORO"
const ARTIFACT_TAG = "model=$(MODEL_NAME)__qoi=$(QOI_BASIS)__target=$(TARGET_PARAM)"

# ---------------------------
# Замеры времени (long-формат, как было)
# ---------------------------
const TimingRow = NamedTuple{(:label, :iteration, :seconds), Tuple{String, Int, Float64}}

@inline function push_timing!(rows::Vector{TimingRow}, label::String, iteration::Int, seconds::Float64)
    push!(rows, (label=label, iteration=iteration, seconds=seconds))
    return nothing
end

# ---------------------------
# Плоские счётчики прогонов
# ---------------------------
const RUN_CNT = Dict{Symbol, Int}(
    :forward_truth => 0,
    :forward_eval  => 0,
    :backward_eval => 0,
    :grad_eval     => 0,
    :param_update  => 0,
    :forward_final => 0,
)

# ---------------------------
# Таблица "шаги × типы времени" (wide-формат)
# ---------------------------
function write_step_timing_table_csv(path::AbstractString;
                                     t_report::Vector{Float64},
                                     fwd_mean::Vector{Float64},
                                     bwd_mean::Vector{Float64},
                                     total_mean::Vector{Float64})
    @assert length(t_report) == length(fwd_mean) == length(bwd_mean) == length(total_mean)
    open(path, "w") do io
        println(io, "step_index,t_report,forward_eval_mean_sec,backward_eval_mean_sec,total_eval_mean_sec")
        for i in eachindex(t_report)
            @printf(io, "%d,%.9f,%.6f,%.6f,%.6f\n",
                    i, t_report[i], fwd_mean[i], bwd_mean[i], total_mean[i])
        end
    end
    return nothing
end

# ---------------------------
# Пользовательские флаги
# ---------------------------
USE_COARSE      = true
COARSE_SHAPE    = (20, 20, 3)
USE_FIRST_STEPS = true
N_STEPS_TO_USE  = 2

PLOT          = false
SAVE_TIMINGS  = true
OUTDIR        = joinpath(pwd(), "fig")
isdir(OUTDIR) || mkpath(OUTDIR)
if PLOT
    using GLMakie
end

STEP_TABLE_CSV_PATH   = joinpath(OUTDIR, "timing_steps__$(ARTIFACT_TAG).csv")

# ---------------------------
# 0) NORNE + базовый кейс (единожды)
# ---------------------------
data_pth = joinpath(GeoEnergyIO.test_input_file_path("EGG"), "EGG.DATA")
@assert isfile(data_pth) "Файл кейса не найден: $data_pth"
data_raw  = parse_data_file(data_pth)

function slice_steps(case::JutulCase)
    USE_FIRST_STEPS || return case
    @assert N_STEPS_TO_USE ≥ 1
    nsteps = length(case.dt)
    return case[1:min(N_STEPS_TO_USE, nsteps)]
end

function coarse_case(case::JutulCase)
    USE_COARSE || return case
    return coarsen_reservoir_case(case, COARSE_SHAPE, method = :ijk)
end

BASE_CASE = setup_case_from_data_file(data_raw) |> slice_steps |> coarse_case

# Базовый прогон (truth) на CUDA
const SIMULATOR_ARGS = (
    output_substates = true,
    linear_solver_backend = :cuda,
    precond = :ilu0,
)
timing_rows = TimingRow[]
t_truth_start = time_ns()
res_true = simulate_reservoir(BASE_CASE; linear_solver_backend=:cuda, precond=:ilu0)
push_timing!(timing_rows, "truth_forward", 0, 1e-9 * (time_ns() - t_truth_start))
RUN_CNT[:forward_truth] += 1

# ---------------------------
# 1) Истинные ряды QOI и масштабы per-well
# ---------------------------
function pick_qoi_key(wres)::Tuple{Symbol,Bool}
    for k in (:orat, :orate, :qo, :oil); haskey(wres, k) && return (k, false); end
    for k in (:wrat, :grat, :lrat);      haskey(wres, k) && return (k, true);  end
    return (:missing, true)
end

function build_truth_per_well_all(res)
    wells = res.wells.wells
    @assert !isempty(wells) "В результате симуляции не найдено скважин"
    t_true = Float64.(res.time)
    @assert !isempty(t_true) "Пустой временной вектор в результате симуляции"

    ALL_WELLS = collect(keys(wells))
    qkey_by   = Dict{Symbol,Tuple{Symbol,Bool}}()
    scale_by  = Dict{Symbol,Float64}()
    truth_by  = Dict{Symbol,Vector{Float64}}()

    for w in ALL_WELLS
        wres = wells[w]
        qk, use_abs = pick_qoi_key(wres)
        @assert qk != :missing "У скважины $w нет подходящих QOI (:orat/:orate/:qo/:oil/:wrat/:grat/:lrat)."
        v = Float64.(wres[qk])
        @assert length(v) == length(t_true) "Длина QOI для $w не равна длине времени"
        truth_by[w] = use_abs ? abs.(v) : v
        qkey_by[w]  = (qk, use_abs)
        scale_by[w] = max(1e-12, maximum(abs.(truth_by[w])))
    end
    return t_true, truth_by, ALL_WELLS, qkey_by, scale_by
end

t_true, q_true_map, ALL_WELLS, QOI_PER_WELL, Q_SCALE_PER_WELL =
    build_truth_per_well_all(res_true)
@assert !isempty(ALL_WELLS)

N_W   = length(ALL_WELLS)
W_INV = 1.0 / N_W
T_INV = 1.0 / max(eps(), (t_true[end] - t_true[1]))

@inline function nearest_report_index(t::Float64, times::AbstractVector{<:Real})
    i = searchsortedfirst(times, t)
    if i ≤ 1
        return 1
    elseif i > length(times)
        return length(times)
    else
        (t - times[i-1] ≤ times[i] - t) ? (i - 1) : i
    end
end

# ---------------------------
# 2) Подготовка PORO как ВЕКТОРА
# ---------------------------
function keep_only_porosity!(dopt::DictParameters)
    mdl = dopt.parameters[:model]
    for k in collect(keys(mdl))
        k == :porosity && continue
        delete!(mdl, k)
    end
    dopt.possible_targets = Any[[:model, :porosity]]
    if !isempty(dopt.parameter_targets)
        for key in collect(keys(dopt.parameter_targets))
            key == [:model, :porosity] && continue
            delete!(dopt.parameter_targets, key)
        end
    end
    return dopt
end

function make_poro_template_and_len()
    dopt = setup_reservoir_dict_optimization(BASE_CASE)
    free_optimization_parameters!(dopt)
    keep_only_porosity!(dopt)
    @assert haskey(dopt.parameters, :model)
    @assert haskey(dopt.parameters[:model], :porosity)
    poro_vec = dopt.parameters[:model][:porosity]
    @assert poro_vec isa AbstractVector
    return Array{Float64}(poro_vec), length(poro_vec)
end

PORO_TEMPLATE, PORO_LEN = make_poro_template_and_len()
@info "PORO_TEMPLATE: длина = $PORO_LEN (вектор)"

function dict_with_poro_vec(poro_vec::AbstractVector{<:Real})
    @assert length(poro_vec) == PORO_LEN "Длина PORO=$(length(poro_vec)) ≠ $PORO_LEN"
    dopt = setup_reservoir_dict_optimization(BASE_CASE; do_copy=true)
    keep_only_porosity!(dopt)
    free_optimization_parameters!(dopt)
    target = dopt.parameters[:model][:porosity]
    @assert length(target) == PORO_LEN
    copyto!(target, Float64.(poro_vec))
    return dopt
end

case_with_poro_vec(poro_vec::AbstractVector{<:Real}) =
    dict_with_poro_vec(poro_vec).setup_function(dict_with_poro_vec(poro_vec).parameters, missing)

# ---------------------------
# 3) Adjoint objective (пер-скважинная нормированная MSE)
# ---------------------------
_loss_acc = Ref(0.0)

# --- Вспомогательное состояние для пер-шагового времени при adjoint-оценке
const _IN_TIMED_EVAL      = Ref(false)
const _CALL_COUNTER       = Ref(0)
const _LAST_NS            = Ref{UInt64}(0)
const _CURR_FORWARD_VEC   = Ref(Vector{Float64}())
const FORWARD_BY_ITER     = Vector{Vector{Float64}}()  # список векторов (по шагам) для каждой итерации
const TOTAL_EVAL_SECS     = Float64[]                  # total (forward+backward) на итерацию

function _start_eval_timing!(nsteps::Int)
    _IN_TIMED_EVAL[] = true
    _CALL_COUNTER[]  = 0
    _LAST_NS[]       = time_ns()
    _CURR_FORWARD_VEC[] = zeros(nsteps)
    return nothing
end

function _finalize_eval_timing!()
    # Завершим последний шаг (между последним callback и концом adjoint-оценки)
    if _IN_TIMED_EVAL[] && _CALL_COUNTER[] ≥ 1
        now = time_ns()
        dt = 1e-9 * (now - _LAST_NS[])
        idx = min(_CALL_COUNTER[], length(_CURR_FORWARD_VEC[]))
        _CURR_FORWARD_VEC[][idx] += dt
        _LAST_NS[] = now
    end
    push!(FORWARD_BY_ITER, copy(_CURR_FORWARD_VEC[]))
    _IN_TIMED_EVAL[] = false
end

function per_well_objective_report(m, s, dt, step_info, forces)
    # Пер-шаговое forward-время: учитываем дельту со времени прошлого вызова
    if _IN_TIMED_EVAL[]
        now = time_ns()
        if _CALL_COUNTER[] ≥ 1
            prev_idx = _CALL_COUNTER[] # время между (prev_idx) и текущим вызовом — это вклад в prev_idx
            if prev_idx ≤ length(_CURR_FORWARD_VEC[])
                _CURR_FORWARD_VEC[][prev_idx] += 1e-9 * (now - _LAST_NS[])
            end
        end
        _CALL_COUNTER[] += 1
        _LAST_NS[] = now
    end

    # Основной функционал
    t = value(step_info[:time]) + dt
    k = nearest_report_index(t, t_true)

    acc = zero(dt)
    for w in ALL_WELLS
        (qk, use_abs) = QOI_PER_WELL[w]
        q_truth = q_true_map[w][k]
        q_curr  = JutulDarcy.compute_well_qoi(m, s, forces, w, qk)
        q_curr  = use_abs ? abs(q_curr) : q_curr
        z       = (q_curr - q_truth) / Q_SCALE_PER_WELL[w]
        acc    += z*z
    end
    contrib = dt * W_INV * acc * T_INV
    _loss_acc[] += value(contrib)
    return contrib
end

# ---------------------------
# 4) Извлечение ∂L/∂PORO
# ---------------------------
function poro_grad_vector(grad_dict)
    @assert haskey(grad_dict, :model)
    mdl = grad_dict[:model]
    @assert haskey(mdl, :porosity)
    gd = mdl[:porosity]
    @assert gd isa AbstractVector
    @assert length(gd) == PORO_LEN
    return collect(Float64.(gd))
end

# ---------------------------
# 5) Оценка L и ∂L/∂PORO + тайминг по шагам
# ---------------------------
function evaluate_L_and_g_adjoint(poro_vec::AbstractVector{<:Real};
                                  timing_cb::Union{Nothing,Function}=nothing)
    t0 = time_ns()
    dopt = dict_with_poro_vec(poro_vec)
    _loss_acc[] = 0.0

    # включаем пер-шаговый учёт времени
    nsteps = length(t_true)
    _start_eval_timing!(nsteps)

    gdict = parameters_gradient_reservoir(dopt, per_well_objective_report;
        simulator_arg = SIMULATOR_ARGS)

    # финализируем пер-шаговую часть
    _finalize_eval_timing!()

    elapsed = 1e-9 * (time_ns() - t0)
    !(timing_cb === nothing) && timing_cb(elapsed)
    push!(TOTAL_EVAL_SECS, elapsed)

    # счётчики
    RUN_CNT[:grad_eval]     += 1
    RUN_CNT[:forward_eval]  += 1
    RUN_CNT[:backward_eval] += 1

    L = _loss_acc[]
    @assert L ≥ 0 "Loss получился отрицательным: $L"
    g = poro_grad_vector(gdict)
    return L, g
end

# ---------------------------
# 6) Обучение PORO: GD + проекция
# ---------------------------
function train_porosity_adjoint!(poro0_vec::AbstractVector{<:Real};
                                 eta::Float64=1e-4,
                                 nrounds::Int=10,
                                 poro_min::Float64=1e-6,
                                 poro_max::Float64=0.999)
    @assert eta > 0
    @assert nrounds ≥ 1
    @assert poro_min < poro_max
    @assert length(poro0_vec) == PORO_LEN

    x = clamp.(Float64.(poro0_vec), poro_min, poro_max)

    losses   = Float64[]
    poro_avg = Float64[]
    timings  = TimingRow[]

    for k in 1:nrounds
        iter_t0 = time_ns()
        eval_cb = (secs) -> push_timing!(timings, "adjoint_eval", k, secs)
        Lk, gk = evaluate_L_and_g_adjoint(x; timing_cb = eval_cb)
        @assert length(gk) == length(x)
        @. x = clamp(x - eta * gk, poro_min, poro_max)
        RUN_CNT[:param_update] += 1
        push!(losses, Lk)
        push!(poro_avg, mean(x))
        iter_elapsed = 1e-9 * (time_ns() - iter_t0)
        push_timing!(timings, "train_iteration", k, iter_elapsed)
        @info "[$MODEL_NAME | $QOI_BASIS | $TARGET_PARAM] iter=$k loss=$(round(Lk, sigdigits=7)) mean(PORO)=$(round(poro_avg[end], sigdigits=7)) eta=$eta"
    end

    return (losses=losses, poro_mean=poro_avg, poro_final=x, timings=timings)
end

# ---------------------------
# 7) Старт и обучение
# ---------------------------
start_poro = clamp.(0.7 .* PORO_TEMPLATE, 1e-6, 0.999)
train_start = time_ns()
result = train_porosity_adjoint!(start_poro; eta=1e-2, nrounds=4, poro_min=1e-6, poro_max=0.999)
train_elapsed = 1e-9 * (time_ns() - train_start)
append!(timing_rows, result.timings)
push_timing!(timing_rows, "train_total", 0, train_elapsed)

# ---------------------------
# 8) Финальный прогон и сравнение пред/истина по скважинам
# ---------------------------
function per_well_series(res, wells::Vector{Symbol}, qoi_map::Dict{Symbol,Tuple{Symbol,Bool}})
    raw = res.wells.wells
    t = Float64.(res.time)
    map = Dict{Symbol,Vector{Float64}}()
    for w in wells
        (qk, use_abs) = qoi_map[w]
        v = Float64.(raw[w][qk])
        map[w] = use_abs ? abs.(v) : v
    end
    return t, map
end

t_fit_start = time_ns()
res_fit = simulate_reservoir(case_with_poro_vec(result.poro_final);
    linear_solver_backend = :cuda, precond = :ilu0)
push_timing!(timing_rows, "final_forward", length(result.losses), 1e-9 * (time_ns() - t_fit_start))
RUN_CNT[:forward_final] += 1

t_fit, q_fit_map = per_well_series(res_fit, ALL_WELLS, QOI_PER_WELL)

function final_normalized_mse(t_ref, q_truth_map, t_pred, q_pred_map, wells, scales)
    K = min(length(t_ref), length(t_pred))
    @assert K ≥ 1 "Нет пересечения по времени для сравнения"
    s = 0.0
    for w in wells
        sc = scales[w]
        qT = q_truth_map[w]
        qP = q_pred_map[w]
        @assert length(qT) ≥ K && length(qP) ≥ K "Недостаточно точек для $w"
        @inbounds for k in 1:K
            z = (qP[k] - qT[k]) / sc
            s += z*z
        end
    end
    return s / (K * length(wells))
end

final_mse = final_normalized_mse(t_true, q_true_map, t_fit, q_fit_map, ALL_WELLS, Q_SCALE_PER_WELL)

println("===============================================")
@printf("[%s | %s | %s]\n", MODEL_NAME, QOI_BASIS, TARGET_PARAM)
@printf("FINAL LOSS (last iter): %.6g\n", result.losses[end])
@printf("FINAL normalized MSE   : %.6g\n", final_mse)
@printf("FINAL mean PORO        : %.6g\n", result.poro_mean[end])
println("Артефакты сохраняются в: $OUTDIR")
println("===============================================")

# ---------------------------
# 9) Wide-таблица времени: шаги × типы времени (по adjoint-оценкам)
# ---------------------------
# fwd_per_iter: список векторов длины nsteps; total_eval_per_iter — скаляры
@assert length(FORWARD_BY_ITER) == length(TOTAL_EVAL_SECS) == length(result.losses)

nsteps = length(t_true)
@assert all(length(v) == nsteps for v in FORWARD_BY_ITER)

# средние по forward-времени на шаг
fwd_mean = [ mean(getindex.(FORWARD_BY_ITER, i)) for i in 1:nsteps ]

# оценка backward: на каждой итерации берём (total - sum(forward)), делим поровну по шагам
bwd_per_step_iters = Vector{Vector{Float64}}(undef, length(TOTAL_EVAL_SECS))
for k in eachindex(TOTAL_EVAL_SECS)
    fsum = sum(FORWARD_BY_ITER[k])
    bwd_total = max(0.0, TOTAL_EVAL_SECS[k] - fsum)
    bwd_per_step_iters[k] = fill(bwd_total / nsteps, nsteps)
end
bwd_mean = [ mean(getindex.(bwd_per_step_iters, i)) for i in 1:nsteps ]

total_mean = [ fwd_mean[i] + bwd_mean[i] for i in 1:nsteps ]

if SAVE_TIMINGS
    write_step_timing_table_csv(STEP_TABLE_CSV_PATH;
        t_report = t_true, fwd_mean=fwd_mean,
        bwd_mean=bwd_mean, total_mean=total_mean)

    # Итоговые агрегаты времени (для COUNTERS_CSV)
    extra = Dict{Symbol,Float64}(
        :truth_forward_secs => sum(r.seconds for r in timing_rows if r.label == "truth_forward"),
        :final_forward_secs => sum(r.seconds for r in timing_rows if r.label == "final_forward"),
        :train_total_secs   => sum(r.seconds for r in timing_rows if r.label == "train_total"),
        :adjoint_eval_mean_secs => mean(TOTAL_EVAL_SECS),
    )
end

# ---------------------------
# 10) (Опционально) Визуализация (с расширенными подписями)
# ---------------------------
if PLOT
    title_prefix = "Модель: $(MODEL_NAME) | QOI: дебиты | Подбираем: пористость (PORO)"
    fig1 = Figure(size=(900, 320))
    ax1  = Axis(fig1[1,1], title=title_prefix * " — Loss vs iteration", xlabel="iteration", ylabel="loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    save(joinpath(OUTDIR, "loss_vs_iter__$(ARTIFACT_TAG).png"), fig1); display(fig1)

    fig2 = Figure(size=(900, 320))
    ax2  = Axis(fig2[1,1], title=title_prefix * " — Mean PORO vs iteration", xlabel="iteration", ylabel="mean(poro)")
    lines!(ax2, 1:length(result.poro_mean), result.poro_mean)
    save(joinpath(OUTDIR, "poro_mean_vs_iter__$(ARTIFACT_TAG).png"), fig2); display(fig2)

    show_wells = ALL_WELLS[1:min(6, length(ALL_WELLS))]
    fig3 = Figure(size=(1100, 380))
    grid = fig3[1,1] = GridLayout()
    for (i, w) in enumerate(show_wells)
        ax = Axis(grid[1, i], title=String(w) * " — " * title_prefix, xlabel="report step", ylabel="rate")
        qT = q_true_map[w]; qF = q_fit_map[w]
        Kp = min(length(qT), length(qF))
        lines!(ax, 1:Kp, qT[1:Kp], label="truth")
        lines!(ax, 1:Kp, qF[1:Kp], label="pred")
        axislegend(ax; position=:rt)
    end
    save(joinpath(OUTDIR, "per_well_compare__$(ARTIFACT_TAG).png"), fig3); display(fig3)
end
