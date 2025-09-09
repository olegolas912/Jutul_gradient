#############################
# SPE1 truth + oil-rate mismatch (reports) + porosity fit (FD gradient)
#############################

using Jutul, JutulDarcy, GeoEnergyIO
using Statistics: mean

const PLOT = true
if PLOT
    using GLMakie
end

# -------------------------------
# 0) SPE1: чтение и базовый кейс
# -------------------------------
const data_pth  = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
const data_raw  = parse_data_file(data_pth)
const data_work = deepcopy(data_raw)

@inline function build_case_with_poro(poro::Float64)
    fill!(data_work["GRID"]["PORO"], poro)
    return setup_case_from_data_file(data_work)
end

# -------------------------------
# 1) Истина: один прогон и ряды из res.wells[:PROD]
# -------------------------------
const case_true = setup_case_from_data_file(data_raw)
@time res_true = simulate_reservoir(case_true)

const PROD = :PROD
const q_oil_true  = Float64.(res_true.wells[PROD][:orat])  # нефте-дебит
const bhp_true_Pa = Float64.(res_true.wells[PROD][:bhp])   # давление (Па)
const t_true      = res_true.time

@assert length(q_oil_true) == length(t_true)

# -------------------------------
# 2) Лосс по дебиту нефти (с отчётов)
# -------------------------------
function oilrate_mse_loss(res_pred, q_true::AbstractVector{<:Real}; scale::Float64=100.0)
    q_pred = Float64.(res_pred.wells[PROD][:orat])   # <-- фиксированная конвертация
    K = min(length(q_true), length(q_pred))
    s = 0.0
    @inbounds @simd for i in 1:K
        z = (q_pred[i] - q_true[i]) / scale
        s += z*z
    end
    return s / K
end

# -------------------------------
# 3) Оценка лосса и градиента (FD, центральная разность)
# -------------------------------
function evaluate_L_and_g_fd(poro::Float64; eps_rel::Float64=1e-4, scale::Float64=100.0)
    # базовый прогон
    case0 = build_case_with_poro(poro)
    t_fwd = @elapsed res0 = simulate_reservoir(case0)
    L0 = oilrate_mse_loss(res0, q_oil_true; scale=scale)

    # центральная разность
    eps = eps_rel * max(1.0, abs(poro))

    case_p = build_case_with_poro(poro + eps)
    res_p  = simulate_reservoir(case_p)
    Lp     = oilrate_mse_loss(res_p, q_oil_true; scale=scale)

    case_m = build_case_with_poro(poro - eps)
    res_m  = simulate_reservoir(case_m)
    Lm     = oilrate_mse_loss(res_m, q_oil_true; scale=scale)

    g = (Lp - Lm) / (2eps)
    return L0, g, t_fwd
end

# -------------------------------
# 4) Градиентный спуск по пористости (без BB)
# -------------------------------
function train_porosity_fd!(poro0::Float64;
                            η::Float64=1e-2,
                            nrounds::Int=12,
                            tol_step::Float64=1e-8,
                            poro_min::Float64=1e-6,
                            poro_max::Float64=1.0,
                            eps_rel::Float64=1e-4,
                            scale::Float64=100.0)

    losses = Float64[]; poros = Float64[]; times = Float64[]
    x = clamp(poro0, poro_min, poro_max)

    for k in 1:nrounds
        Lk, gk, t_fwd = evaluate_L_and_g_fd(x; eps_rel=eps_rel, scale=scale)

        step = -η * gk
        x_new = clamp(x + step, poro_min, poro_max)

        push!(losses, Lk); push!(poros, x_new); push!(times, t_fwd)
        @info "iter=$(k)  L=$(round(Lk, sigdigits=7))  poro=$(round(x_new, sigdigits=8))  " *
              "step=$(round(step, sigdigits=6))  grad=$(round(gk, sigdigits=6))  η=$(round(η, sigdigits=6))  " *
              "| t_fwd≈$(round(t_fwd, digits=3)) s"

        if abs(step) < tol_step
            x = x_new
            break
        end
        x = x_new
    end

    return (losses=losses, poros=poros, poro_final=x, fwd_times=times)
end

# -------------------------------
# 5) Запуск и графики
# -------------------------------
const poro_guess = only(unique(data_raw["GRID"]["PORO"]))
poro0 = clamp(poro_guess * 0.7, 1e-6, 1.0)

const Q_SCALE = max(1e-8, maximum(abs.(q_oil_true)))


result = train_porosity_fd!(poro0; η=1e-4, nrounds=12, eps_rel=1e-4, scale=Q_SCALE)

@info "Итог: poro_final = $(result.poro_final)"

# Итоговый прогон и сравнение рядов q_oil
case_fit = build_case_with_poro(result.poro_final)
res_fit  = simulate_reservoir(case_fit)
q_oil_fit = Float64.(res_fit.wells[PROD][:orat])
Kp = min(length(q_oil_true), length(q_oil_fit))

if PLOT
    fig1 = Figure(size=(900, 300))
    ax1  = Axis(fig1[1,1], title="Loss vs iteration (q_oil mismatch via reports)", xlabel="iteration", ylabel="loss")
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
