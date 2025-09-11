#############################
# SPE1 truth + adjoint (oil-rate mismatch) + ∂L/∂porosity
#############################

using Jutul, JutulDarcy, GeoEnergyIO
using Statistics: mean
using ForwardDiff: value

const PLOT = true
if PLOT
    using GLMakie
end

# -------------------------------
# 0) SPE1 и постройка кейса с пористостью
# -------------------------------


data_pth = joinpath(JutulDarcy.GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data_pth = joinpath(JutulDarcy.GeoEnergyIO.test_input_file_path("SPE9"), "SPE9.DATA")
data_pth = joinpath(JutulDarcy.GeoEnergyIO.test_input_file_path("EGG"), "EGG.DATA")
data_pth = joinpath(JutulDarcy.GeoEnergyIO.test_input_file_path("NORNE_NOHYST"), "NORNE_NOHYST.DATA")

data = parse_data_file(data_pth);
case = setup_case_from_data_file(data);


const data_pth  = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
const data_raw  = parse_data_file(data_pth)
const data_work = deepcopy(data_raw)

@inline function build_case_with_poro(poro::Float64)
    fill!(data_work["GRID"]["PORO"], poro)
    return setup_case_from_data_file(data_work)
end

# -------------------------------
# 1) «Истина»: один прогон и ряды из res_true.wells[:PROD]
# -------------------------------
const case_true = setup_case_from_data_file(data_raw)
@time res_true = simulate_reservoir(case_true)

const PROD = :PROD
const q_oil_true = Float64.(res_true.wells[PROD][:orat])  # нефте-дебит (по отчётам)
const t_true     = res_true.time
@assert length(q_oil_true) == length(t_true)

# нормирующий масштаб (безразмерность лосса)
const Q_SCALE = max(1e-8, maximum(abs.(q_oil_true)))
const T_INV   = 1.0 / (t_true[end] - t_true[1])

# ближайший отчётный индекс для времени t
@inline function nearest_report_index(t::Float64, times)
    i = searchsortedfirst(times, t)
    if i <= 1
        return 1
    elseif i > length(times)
        return length(times)
    else
        (t - times[i-1] <= times[i] - t) ? (i - 1) : i
    end
end

# -------------------------------
# 2) Целевая функция для adjoint: MSE по q_oil (интегрально по времени)
# -------------------------------
const _loss_acc = Ref(0.0)

function oilrate_objective_report(m, s, dt, step_info, forces)
    # время текущей точки интегрирования
    t = value(step_info[:time]) + dt
    k = nearest_report_index(t, t_true)

    # «истина» по отчётам
    q_true = q_oil_true[k]            # Float64

    # Живой дебит ИЗ ТЕКУЩЕГО состояния (оставляем тип Dual!)
    q_curr = JutulDarcy.compute_well_qoi(m, s, forces, :PROD, :orat)

    # безразмерная ошибка; z – Dual
    z = (q_curr - q_true) / Q_SCALE

    # интегральный вклад по времени; contrib – Dual
    contrib = dt * (z*z) * T_INV

    # в аккумулятор кладём численное значение (Float64),
    # но наружу возвращаем Dual для корректного градиента
    _loss_acc[] += value(contrib)
    return contrib
end


# -------------------------------
# 3) Извлечение ∂L/∂porosity из словаря градиента
# -------------------------------
@inline function poro_grad_scalar(grad_dict)::Float64
    v = grad_dict[:model][:porosity]
    return v isa AbstractArray ? mean(v) : Float64(v)
end

# -------------------------------
# 4) Оценка лосса и градиента через встроенный adjoint
# -------------------------------
function evaluate_L_and_g_adjoint(poro::Float64)
    case_c    = build_case_with_poro(poro)
    dprm_case = setup_reservoir_dict_optimization(case_c)
    free_optimization_parameters!(dprm_case)

    # опционально: чистый forward для тайминга
    t_fwd_est = @elapsed simulate_reservoir(case_c)

    _loss_acc[] = 0.0
    t_grad_total = @elapsed gdict = parameters_gradient_reservoir(dprm_case, oilrate_objective_report)

    L = _loss_acc[]
    g = poro_grad_scalar(gdict)
    t_adj_est = max(t_grad_total - t_fwd_est, 0.0)

    return L, g, (t_fwd=t_fwd_est, t_grad=t_grad_total, t_adj=t_adj_est)
end

# -------------------------------
# 5) Градиентный спуск (фиксированный η, без BB)
# -------------------------------
function train_porosity_adjoint!(poro0::Float64;
                                 η::Float64=1e-4,
                                 nrounds::Int=20,
                                 poro_min::Float64=1e-6,
                                 poro_max::Float64=1.0)

    losses = Float64[]; poros = Float64[]
    fwd_t  = Float64[]; adj_t = Float64[]; grad_t = Float64[]

    x = clamp(poro0, poro_min, poro_max)

    for k in 1:nrounds
        Lk, gk, tms = evaluate_L_and_g_adjoint(x)

        step = -η * gk
        x = clamp(x + step, poro_min, poro_max)

        push!(losses, Lk); push!(poros, x)
        push!(fwd_t, tms.t_fwd); push!(adj_t, tms.t_adj); push!(grad_t, tms.t_grad)

        @info "iter=$(k)  L=$(round(Lk, sigdigits=7))  poro=$(round(x, sigdigits=8))  " *
              "step=$(round(step, sigdigits=6))  grad=$(round(gk, sigdigits=6))  η=$(round(η, sigdigits=6))  |  " *
              "t_fwd≈$(round(tms.t_fwd, digits=3)) s  t_adj≈$(round(tms.t_adj, digits=3)) s  t_grad=$(round(tms.t_grad, digits=3)) s"
    end

    return (losses=losses, poros=poros, poro_final=x,
            fwd_times=fwd_t, adj_times=adj_t, grad_total_times=grad_t)
end

# -------------------------------
# 6) Запуск и сравнение
# -------------------------------
const poro_guess = only(unique(data_raw["GRID"]["PORO"]))
poro0 = clamp(poro_guess * 0.7, 1e-6, 1.0)

result = train_porosity_adjoint!(poro0; η=3e-2, nrounds=12)

@info "Итог: poro_final = $(result.poro_final)"

# постфактум сравнение рядов q_oil
case_fit = build_case_with_poro(result.poro_final)
res_fit  = simulate_reservoir(case_fit)
q_oil_fit = Float64.(res_fit.wells[PROD][:orat])
Kp = min(length(q_oil_true), length(q_oil_fit))

if PLOT
    fig1 = Figure(size=(900, 300))
    ax1  = Axis(fig1[1,1], title="Loss vs iteration (adjoint, q_oil mismatch)", xlabel="iteration", ylabel="loss")
    lines!(ax1, 1:length(result.losses), result.losses)
    #display(fig1)

    fig2 = Figure(size=(900, 300))
    ax2  = Axis(fig2[1,1], title="Porosity vs iteration", xlabel="iteration", ylabel="poro")
    lines!(ax2, 1:length(result.poros), result.poros)
    display(fig2)

    fig3 = Figure(size=(1000, 360))
    ax3  = Axis(fig3, title="q_oil per step: truth vs fitted", xlabel="report step", ylabel="q_oil (m^3/day)")
    lines!(ax3, 1:Kp, q_oil_true[1:Kp], label="truth")
    lines!(ax3, 1:Kp, q_oil_fit[1:Kp],  label="fitted")
    axislegend(ax3; position=:rt)
    #display(fig3)
end
