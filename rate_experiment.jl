# min_npv3.jl — три кейса, сравнение их NPV и бар-график (GLMakie)

using Jutul, JutulDarcy, GeoEnergyIO, GLMakie

# --- 1) База: EGG → 50 шагов → укрупнение 20×20×3
data_dir = GeoEnergyIO.test_input_file_path("EGG")
case_fine = setup_case_from_data_file(joinpath(data_dir, "EGG.DATA"))
case_fine = case_fine[1:50]
base_case = coarsen_reservoir_case(case_fine, (20, 20, 3), method = :ijk)

# --- 2) Параметры экономики
const OIL_PRICE   = 100.0     # $/stb
const WATER_PRICE = -10.0     # $/stb (штраф за добытую воду)
const WATER_COST  =  5.0      # $/stb (стоимость закачки воды)
const DISC        =  0.05     # 5% годовых

ctrl = base_case.forces[1][:Facility].control
base_rate = ctrl[:INJECT1].target.value

function optimize_case(case, steps)
    setup = JutulDarcy.setup_rate_optimization_objective(case, base_rate;
        max_rate_factor = 10.0,
        oil_price = OIL_PRICE,
        water_price = WATER_PRICE,
        water_cost = WATER_COST,
        discount_rate = DISC,
        maximize = true,
        sim_arg = (rtol = 1e-5, tol_cnv = 1e-5),
        steps = steps              # :first — одна доля на весь период, :each — новая на каждый отчётный шаг
    )
    obj_best, x_best, hist = Jutul.unit_box_bfgs(setup.x0, setup.obj; maximize = true, lin_eq = setup.lin_eq)
    final_obj = hist.val[end]      # NPV оптимизатора на последней итерации (для справки)
    return setup.case, final_obj
end

# --- 3) Три кейса
const_case, npv_const_opt = optimize_case(base_case, :first)
var_case,   npv_var_opt   = optimize_case(base_case, :each)

# --- 4) Прогоны
ws_base,  _ = simulate_reservoir(base_case,  info_level = -1)
ws_const, _ = simulate_reservoir(const_case, info_level = -1)
ws_var,   _ = simulate_reservoir(var_case,   info_level = -1)

# --- 5) Единый расчёт NPV по результатам
bbl = si_unit(:stb)
function npv_from_ws(model, ws; po = OIL_PRICE, pw = WATER_PRICE, ci = WATER_COST, r = DISC)
    f = reservoir_measurables(model, ws)
    t_years = (ws.time ./ si_unit(:day)) ./ 365.0
    fopt = f[:fopt].values ./ bbl   # нефть, cum stb
    fwpt = f[:fwpt].values ./ bbl   # вода добытая, cum stb
    fwit = f[:fwit].values ./ bbl   # вода закачанная, cum stb

    dOil = diff(vcat(0.0, fopt))
    dWpr = diff(vcat(0.0, fwpt))
    dWin = diff(vcat(0.0, fwit))
    disc = (1 .+ r).^(-t_years)

    return sum( (po .* dOil .+ pw .* dWpr .- ci .* dWin) .* disc )
end

npv_base  = npv_from_ws(base_case.model,  ws_base)
npv_const = npv_from_ws(const_case.model, ws_const)
npv_var   = npv_from_ws(var_case.model,   ws_var)

# --- 6) Текстовый вывод
vals = [
    ("Base (as DATA)",      npv_base),
    ("Optimized: constant", npv_const),
    ("Optimized: per-step", npv_var),
]
println("\nNPV, USD:")
for (name, v) in vals
    println(rpad("  "*name*":", 28), round(v, sigdigits = 7))
end
println("\nObjective at optimizer last iter (USD):")
println("  constant: ", round(npv_const_opt, sigdigits = 7))
println("  per-step: ", round(npv_var_opt,   sigdigits = 7))

# --- 7) Бар-график NPV (в млн $)
labels = ["Base", "Constant", "Per-step"]
npv_vals = [npv_base, npv_const, npv_var] ./ 1e6  # млн USD

fig = Figure(size = (720, 420))
ax  = Axis(fig[1, 1], ylabel = "NPV (million USD)", title = "NPV comparison")
barplot!(ax, 1:length(npv_vals), npv_vals)
ax.xticks = (1:length(labels), labels)

# подписи над столбцами
for (i, v) in enumerate(npv_vals)
    text!(ax, i, v, text = string(round(v, digits = 2)), align = (:center, :bottom))
end

fig  # показать окно
# при необходимости можно сохранить:
# save("npv_bar.png", fig)
