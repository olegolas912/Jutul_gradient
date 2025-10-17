using Jutul
using JutulDarcy
using GLMakie


case = setup_case_from_data_file(joinpath("/home/oleg/Github/Jutul_gradient", "SPE1.DATA"))
ws, states = simulate_reservoir(case, output_substates = true)
plot_reservoir(case.model, states)

# --- по доке: ws.time и ws.wells[:PROD][:orat] ---
t_days   = ws.time ./ si_unit(:day)                 # дни
q_o_day  = -1 * ws.wells[:PROD][:orat] .* si_unit(:day)  # м³/сут (из м³/с)

# накопленная добыча Np (м³) по трапециям в координатах "сутки / м³/сут"
Np_m3 = sum(0.5 .* (q_o_day[1:end-1] .+ q_o_day[2:end]) .* diff(t_days))
@info "Np (накопленная нефть) = $(round(Np_m3, digits=3)) м^3"

# график дебита нефти
fig = Figure()
ax  = Axis(fig[1,1], xlabel = "Время, сутки", ylabel = "Дебит нефти, м³/сут")
lines!(ax, t_days, q_o_day, label = "PROD: orat")
axislegend(ax)
display(fig)