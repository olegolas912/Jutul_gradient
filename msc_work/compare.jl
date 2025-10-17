using Jutul
using JutulDarcy
using GLMakie
using Statistics

# --- две модели ---
case1 = setup_case_from_data_file(joinpath("/home/oleg/Github/Jutul_gradient", "SPE1.DATA"))
ws1, states1 = simulate_reservoir(case1, output_substates = true)

case2 = setup_case_from_data_file(joinpath("/home/oleg/Github/Jutul_gradient", "SPE1 copy.DATA"))
# здесь при желании меняешь case2 (perm/рокфлюиды/управления), затем считаешь
ws2, states2 = simulate_reservoir(case2, output_substates = true)

# --- извлечь ряды времени и дебитов (берём PROD; знак минус, чтобы дебит добычи был >0) ---
t1 = ws1.time ./ si_unit(:day)
q1 = -ws1.wells[:PROD][:orat] .* si_unit(:day)   # м^3/сут

t2 = ws2.time ./ si_unit(:day)
q2 = -ws2.wells[:PROD][:orat] .* si_unit(:day)   # м^3/сут

# --- синхронизация длины (для одинакового DATA обычно сетка времени совпадает) ---
n = min(length(t1), length(t2))
t  = t1[1:n]
q1 = q1[1:n]
q2 = q2[1:n]

# --- накопленная добыча (м^3) и кумулятивные кривые ---
Δt = diff(t)                                # сутки
Np1 = sum(0.5 .* (q1[1:end-1] .+ q1[2:end]) .* Δt)
Np2 = sum(0.5 .* (q2[1:end-1] .+ q2[2:end]) .* Δt)

cum1 = cumsum(vcat(0.0, 0.5 .* (q1[1:end-1] .+ q1[2:end]) .* Δt))  # м^3
cum2 = cumsum(vcat(0.0, 0.5 .* (q2[1:end-1] .+ q2[2:end]) .* Δt))  # м^3

# --- метрики MAE/RMSE для дебитов и для кумулятивов ---
mae_rates = mean(abs.(q1 .- q2))
rmse_rates = sqrt(mean((q1 .- q2).^2))

mae_cum = mean(abs.(cum1 .- cum2))
rmse_cum = sqrt(mean((cum1 .- cum2).^2))

@info "Np1 = $(round(Np1, digits=3)) м^3 | Np2 = $(round(Np2, digits=3)) м^3 | Δ = $(round(Np2 - Np1, digits=3)) м^3"
@info "MAE дебитов = $(round(mae_rates, digits=3)) м^3/сут | RMSE дебитов = $(round(rmse_rates, digits=3)) м^3/сут"
@info "MAE кумулятивов = $(round(mae_cum, digits=3)) м^3 | RMSE кумулятивов = $(round(rmse_cum, digits=3)) м^3"

# --- график дебитов ---
fig = Figure()
ax  = Axis(fig[1, 1], xlabel = "Время, сутки", ylabel = "Дебит нефти, м³/сут")
lines!(ax, t, q1, label = "Model 1")
lines!(ax, t, q2, label = "Model 2")
axislegend(ax)
display(fig)
