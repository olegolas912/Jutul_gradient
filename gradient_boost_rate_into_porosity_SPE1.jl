#############################
# Per-well ALL (producers + injectors) truth + adjoint
# MSE per well over time, ∂L/∂porosity (per-cell), GD training
#############################

using Jutul, JutulDarcy, GeoEnergyIO
using Statistics: mean
using ForwardDiff: value

const PLOT = true
if PLOT
    using GLMakie
end

# -------------------------------------------
# 0) Данные и базовый кейс (SPE1)
# -------------------------------------------
const data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")

using FilePathsBase: basename
root, _ = splitext(basename(data_pth))          # <- тут раскладываем кортеж
const MODEL_TAG = replace(root, r"\W+" => "_")  # очищаем до [A-Za-z0-9_]
const OUTDIR    = joinpath(pwd(), "fig")
isdir(OUTDIR) || mkpath(OUTDIR)

const data_raw  = parse_data_file(data_pth)
const data_work = deepcopy(data_raw)  # сюда будем класть актуальную PORO

# -------------------------------------------
# Классификация скважин и выбор QOI
# -------------------------------------------

# эвристика для производителей (отрицательный :lrat или имя начинается с PROD)
is_prod_name(sym::Symbol) = startswith(uppercase(String(sym)), "PROD")

function classify_wells(wells::Dict{Symbol,<:Any})
    prods = Symbol[]
    injs  = Symbol[]
    for w in keys(wells)
        wres = wells[w]
        if haskey(wres, :lrat) && !isempty(wres[:lrat])
            first_val = wres[:lrat][1]
            if first_val < 0
                push!(prods, w)
            elseif first_val > 0
                push!(injs, w)
            else
                # нулевой старт — дойдём именем
                if is_prod_name(w)
                    push!(prods, w)
                else
                    push!(injs, w)  # по умолчанию в инжекторы
                end
            end
        else
            # нет :lrat — ориентируемся по имени
            if is_prod_name(w)
                push!(prods, w)
            else
                push!(injs, w)
            end
        end
    end
    return prods, injs
end

# подбираем oil-QOI для продюсеров
pick_oil_qoi_key(wres) = begin
    for k in (:orat, :orate, :qo, :oil)
        if haskey(wres, k)
            return k
        end
    end
    return nothing
end

# подбираем inject-QOI для инжекторов
pick_inj_qoi_key(wres) = begin
    for k in (:wrat, :grat, :lrat)
        if haskey(wres, k)
            return k
        end
    end
    return nothing
end

"""
build_truth_per_well_all(res)
→ (t_true::Vector{Float64},
   q_true_map::Dict{Symbol,Vector{Float64}},
   ALL_WELLS::Vector{Symbol},
   QOI_PER_WELL::Dict{Symbol,Tuple{Symbol,Bool}},   # (qkey, use_abs)
   Q_SCALE_PER_WELL::Dict{Symbol,Float64},
   PROD_WELLS::Vector{Symbol},
   INJ_WELLS::Vector{Symbol})

Для продюсеров: oil-QOI, иначе abs(lrat).
Для инжекторов: wrat или grat, иначе abs(lrat).
"""
function build_truth_per_well_all(res)
    wells = res.wells.wells
    t_true = Float64.(res.time)

    prods, injs = classify_wells(wells)
    ALL_WELLS = vcat(prods, injs)
    @assert !isempty(ALL_WELLS) "Не найдены скважины."

    QOI_PER_WELL = Dict{Symbol,Tuple{Symbol,Bool}}()
    q_true_map   = Dict{Symbol,Vector{Float64}}()

    # добывающие
    for w in prods
        wres = wells[w]
        qk = pick_oil_qoi_key(wres)
        if qk === nothing
            @assert haskey(wres, :lrat) "У $w нет oil-QOI и нет :lrat"
            v = Float64.(wres[:lrat])     # < 0 обычно
            @assert length(v) == length(t_true)
            q_true_map[w] = abs.(v)       # делаем положительным дебит добычи
            QOI_PER_WELL[w] = (:lrat, true)
        else
            v = Float64.(wres[qk])
            @assert length(v) == length(t_true)
            q_true_map[w] = v
            QOI_PER_WELL[w] = (qk, false)
        end
    end

    # нагнетательные
    for w in injs
        wres = wells[w]
        qk = pick_inj_qoi_key(wres)
        @assert qk !== nothing "У $w нет ни wrat, ни grat, ни lrat"
        v = Float64.(wres[qk])
        @assert length(v) == length(t_true)
        # для стабильности сравнения используем положительный ряд
        q_true_map[w] = abs.(v)
        QOI_PER_WELL[w] = (qk, true)     # use_abs=true
    end

    # пер-скважинные масштабы нормировки
    Q_SCALE_PER_WELL = Dict{Symbol,Float64}()
    for w in ALL_WELLS
        Q_SCALE_PER_WELL[w] = max(1e-12, maximum(abs.(q_true_map[w])))
    end

    return t_true, q_true_map, ALL_WELLS, QOI_PER_WELL, Q_SCALE_PER_WELL, prods, injs
end

# -------------------------------------------
# Построение кейса с поячеечной пористостью
# -------------------------------------------
@inline function build_case_with_poro(poro::AbstractArray{<:Real})
    @assert haskey(data_work["GRID"], "PORO")
    @assert size(poro) == size(data_work["GRID"]["PORO"]) "Размерности poro и GRID.PORO не совпадают"
    data_work["GRID"]["PORO"] .= poro
    return setup_case_from_data_file(data_work)
end

@inline function build_case_with_poro(poro::Real)
    fill!(data_work["GRID"]["PORO"], Float64(poro))
    return setup_case_from_data_file(data_work)
end

# -------------------------------------------
# 1) Истина: пер-скважинные ряды для всех скважин
# -------------------------------------------
const case_true = setup_case_from_data_file(data_raw)
res_true = simulate_reservoir(case_true)

const t_true,
      q_true_map,
      ALL_WELLS,
      QOI_PER_WELL,
      Q_SCALE_PER_WELL,
      PROD_WELLS,
      INJ_WELLS = build_truth_per_well_all(res_true)

const N_W = length(ALL_WELLS)
const W_INV = 1.0 / N_W
const T_INV = 1.0 / max(eps(), (t_true[end] - t_true[1]))

@inline function nearest_report_index(t::Float64, times::AbstractVector{<:Real})
    i = searchsortedfirst(times, t)
    if i <= 1
        return 1
    elseif i > length(times)
        return length(times)
    else
        (t - times[i-1] <= times[i] - t) ? (i - 1) : i
    end
end

# -------------------------------------------
# 2) Целевая функция: MSE по каждой скважине, затем среднее
# -------------------------------------------
const _loss_acc = Ref(0.0)

function per_well_objective_report(m, s, dt, step_info, forces)
    t = value(step_info[:time]) + dt
    k = nearest_report_index(t, t_true)

    acc = zero(dt)  # Dual
    for w in ALL_WELLS
        (qk, use_abs) = QOI_PER_WELL[w]
        q_truth = q_true_map[w][k]                      # Float64
        q_curr  = JutulDarcy.compute_well_qoi(m, s, forces, w, qk)
        q_curr  = use_abs ? abs(q_curr) : q_curr
        z       = (q_curr - q_truth) / Q_SCALE_PER_WELL[w]
        acc    += z*z
    end

    contrib = dt * W_INV * acc * T_INV
    _loss_acc[] += value(contrib)
    return contrib
end

# -------------------------------------------
# 3) Переупаковка градиента в тензор PORO
# -------------------------------------------
@inline function poro_grad_array(grad_dict)
    @assert haskey(grad_dict, :model) && haskey(grad_dict[:model], :porosity)
    gd = Float64.(grad_dict[:model][:porosity])
    tgt = size(data_work["GRID"]["PORO"])
    if size(gd) == tgt
        return Array{Float64}(gd)
    elseif length(gd) == prod(tgt)
        return reshape(collect(gd), tgt)
    else
        error("Gradient :porosity size $(size(gd)) incompatible with PORO size $(tgt)")
    end
end

# -------------------------------------------
# 4) Оценка L и ∂L/∂poro (adjoint)
# -------------------------------------------
function evaluate_L_and_g_adjoint(poro::AbstractArray{<:Real})
    case_c    = build_case_with_poro(poro)
    dprm_case = setup_reservoir_dict_optimization(case_c)
    free_optimization_parameters!(dprm_case)

    t_fwd_est = @elapsed simulate_reservoir(case_c)
    _loss_acc[] = 0.0
    t_grad_total = @elapsed gdict = parameters_gradient_reservoir(dprm_case, per_well_objective_report)

    L = _loss_acc[]
    g = poro_grad_array(gdict)
    t_adj_est = max(t_grad_total - t_fwd_est, 0.0)

    return L, g, (t_fwd=t_fwd_est, t_grad=t_grad_total, t_adj=t_adj_est)
end

# -------------------------------------------
# 5) Обучение пористости (покомпонентный GD)
# -------------------------------------------
function train_porosity_adjoint!(poro0::AbstractArray{<:Real};
                                 η::Float64=1e-4,
                                 nrounds::Int=20,
                                 poro_min::Float64=1e-6,
                                 poro_max::Float64=1.0)

    x = clamp.(Float64.(poro0), poro_min, poro_max)

    losses   = Float64[]
    poro_avg = Float64[]
    fwd_t    = Float64[]; adj_t = Float64[]; grad_t = Float64[]

    for k in 1:nrounds
        Lk, gk, tms = evaluate_L_and_g_adjoint(x)
        @. x = clamp(x - η * gk, poro_min, poro_max)   # шаг + проекция

        push!(losses, Lk)
        push!(poro_avg, mean(x))
        push!(fwd_t, tms.t_fwd); push!(adj_t, tms.t_adj); push!(grad_t, tms.t_grad)

        @info "iter=$(k) | L=$(round(Lk, sigdigits=7)) | poro_mean=$(round(poro_avg[end], sigdigits=7)) | " *
              "η=$(round(η, sigdigits=6)) | t_fwd≈$(round(tms.t_fwd, digits=3)) s  t_adj≈$(round(tms.t_adj, digits=3)) s  t_grad=$(round(tms.t_grad, digits=3)) s"
    end

    return (losses=losses, poro_mean=poro_avg, poro_final=x,
            fwd_times=fwd_t, adj_times=adj_t, grad_total_times=grad_t)
end

# -------------------------------------------
# 6) Запуск и сравнение
# -------------------------------------------
const poro_true0 = Array{Float64}(data_raw["GRID"]["PORO"])
poro0 = clamp.(0.7 .* poro_true0, 1e-6, 1.0)

result = train_porosity_adjoint!(poro0; η=1e-4, nrounds=12, poro_min=1e-6, poro_max=1.0)
@info "Итог: poro_mean_final = $(result.poro_mean[end]) | producers=$(length(PROD_WELLS)) injectors=$(length(INJ_WELLS))"

# собираем предсказанные пер-скважинные ряды
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

res_fit = simulate_reservoir(build_case_with_poro(result.poro_final))
t_fit, q_fit_map = per_well_series(res_fit, ALL_WELLS, QOI_PER_WELL)

# -------------------------------------------
# 7) Визуализации
# -------------------------------------------
if PLOT
    # 1) Loss
    fig1 = Figure(size=(900, 300))
    ax1  = Axis(fig1[1,1],
        title = "[$MODEL_TAG] Loss vs iteration",
        xlabel = "iteration", ylabel = "loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    display(fig1)
    save(joinpath(OUTDIR, "$(MODEL_TAG)_loss.png"), fig1)
    @info "Saved: $(joinpath(OUTDIR, "$(MODEL_TAG)_loss.png"))"

    # 2) Средняя пористость
    fig2 = Figure(size=(900, 300))
    ax2  = Axis(fig2[1,1],
        title = "[$MODEL_TAG] Mean porosity vs iteration",
        xlabel = "iteration", ylabel = "mean(poro)")
    lines!(ax2, 1:length(result.poro_mean), result.poro_mean)
    display(fig2)
    save(joinpath(OUTDIR, "$(MODEL_TAG)_poro_mean.png"), fig2)
    @info "Saved: $(joinpath(OUTDIR, "$(MODEL_TAG)_poro_mean.png"))"

    # 3) Пер-скважинное сравнение (producers + injectors)
    show_prods = PROD_WELLS[1:min(3, length(PROD_WELLS))]
    show_injs  = INJ_WELLS[1:min(3, length(INJ_WELLS))]
    show_wells = vcat(show_prods, show_injs)

    fig3 = Figure(size=(1100, 380))
    grid = fig3[1,1] = GridLayout()
    for (i, w) in enumerate(show_wells)
        ax = Axis(grid[1, i],
            title  = "[$MODEL_TAG] $(String(w))",
            xlabel = "report step", ylabel = "rate")
        qT = q_true_map[w]; qF = q_fit_map[w]
        Kp = min(length(qT), length(qF))
        lines!(ax, 1:Kp, qT[1:Kp], label="truth")
        lines!(ax, 1:Kp, qF[1:Kp], label="fitted")
        axislegend(ax; position=:rt)
    end
    display(fig3)
    save(joinpath(OUTDIR, "$(MODEL_TAG)_per_well.png"), fig3)
    @info "Saved: $(joinpath(OUTDIR, "$(MODEL_TAG)_per_well.png"))"
end

