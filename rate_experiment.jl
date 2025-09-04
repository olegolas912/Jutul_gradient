# ==========================================
# Producer-rate optimization on EGG example
# (control = production well rates instead of injectors)
# ==========================================
using Jutul, JutulDarcy, GLMakie, GeoEnergyIO, HYPRE, LBFGSB
using Statistics: mean

# Если в проекте есть локальные утилиты/цели — подключаем
# (в них обычно лежат compute_well_qoi, reservoir_measurables, npv_objective, report_timesteps)
include("utils.jl")
include("objectives.jl")

# ---------- Load and coarsen EGG case (same as original example) ----------
data_dir  = GeoEnergyIO.test_input_file_path("EGG")
data_pth  = joinpath(data_dir, "EGG.DATA")
fine_case = setup_case_from_data_file(data_pth)
coarse_case = coarsen_reservoir_case(fine_case[1:50], (20, 20, 3); method = :ijk)

# ---------- Helpers ----------
const EPS_RATE = 1e-12

function get_well_lists(case::JutulCase)
    f1 = case.forces[1][:Facility]
    wells = collect(keys(f1.control))
    injectors = Symbol[]
    producers = Symbol[]
    for w in wells
        c = f1.control[w]
        if c isa InjectorControl
            push!(injectors, w)
        elseif c isa ProducerControl
            push!(producers, w)
        end
    end
    return injectors, producers
end

function liquid_rate_of_well(model, state, forces, w::Symbol)
    # Предпочтительно surface liquid rate; если нет — oil + water
    try
        return abs(compute_well_qoi(model, state, forces, w, SurfaceLiquidRateTarget))
    catch
        lr = 0.0
        try
            lr += abs(compute_well_qoi(model, state, forces, w, SurfaceOilRateTarget))
        catch
        end
        try
            lr += abs(compute_well_qoi(model, state, forces, w, SurfaceWaterRateTarget))
        catch
        end
        return lr
    end
end

function base_rate_from_producers(case::JutulCase; default_if_zero=convert_to_si(100.0, :stb)/si_unit(:day))
    ws, states = simulate_reservoir(case; info_level = -1)
    model   = case.model
    forces1 = case.forces[1]
    _, producers = get_well_lists(case)
    isempty(producers) && error("В кейсе не найдены добывающие скважины.")
    # берём средний дебит по продюсерам на первом отчётном шаге
    s1   = states[1]
    vals = [liquid_rate_of_well(model, s1, forces1, w) for w in producers]
    br   = mean(vals)
    return br > EPS_RATE ? br : default_if_zero
end

# ---------- Objective builder: control PRODUCER rates ----------
"""
    setup_producer_rate_optimization_objective(case, base_rate; kwargs...)

Как setup_rate_optimization_objective, но управляем добывающими скважинами:
- управление: ProducerControl → TotalRateTarget
- NPV: выручка с продюсеров и (как обычно) стоимость закачки инжекторов
- градиент: по целям продюсеров (TotalRateTarget)
"""
function setup_producer_rate_optimization_objective(case::JutulCase, base_rate;
        max_rate_factor = 10.0,
        steps = :first,                  # :first = один набор, :each = на каждый репорт-шаг
        use_ministeps = true,
        limits_enabled = true,
        verbose = true,
        # NPV economics (per barrel / per Mscf)
        oil_price   = 100.0,
        gas_price   = 0.0,
        water_price = -10.0,
        water_cost  = 5.0,
        oil_cost    = 0.0,
        gas_cost    = 0.0,
        discount_rate = 0.05,
        sim_arg = NamedTuple(),
        kwarg...
    )
    steps in (:first, :each) || error("steps must be :first or :each")
    myprint(s) = (verbose && println(s))

    # Свободно редактируем forces поминутно
    nstep  = length(case.dt)
    forces = case.forces isa Vector ? [deepcopy(f) for f in case.forces] : [deepcopy(case.forces) for _ in 1:nstep]
    case   = JutulCase(case.model, case.dt, forces; state0=case.state0, parameters=case.parameters)

    # Инжекторы/продюсеры
    injectors, producers = get_well_lists(case)
    isempty(producers) && error("Не найдены добывающие скважины.")
    isempty(injectors) && myprint("Внимание: инжекторов нет — затраты на закачку будут нулевыми.")

    control_wells = producers
    n_ctrl = length(control_wells)
    myprint("Управляемые скважины (добывающие): $control_wells; всего = $n_ctrl")

    eachstep      = steps == :each
    nstep_unique  = eachstep ? nstep : 1
    max_rate      = max_rate_factor*base_rate

    # По желанию — «сбросить» лимиты (оставляя корректный знак)
    if !limits_enabled
        for f in case.forces
            fF = f[:Facility]
            for (k, c) in fF.control
                fF.limits[k] = default_limits(c)
            end
        end
    end

    cache = Dict{Symbol,Any}()

    function f!(x; grad=true)
        X = reshape(x, n_ctrl, nstep_unique)

        # Применяем цели дебита для продюсеров
        for stepno in 1:nstep
            for (i, w) in enumerate(control_wells)
                x_i = X[i, eachstep ? stepno : 1]
                new_rate = max(x_i*max_rate, EPS_RATE)
                fF = case.forces[stepno][:Facility]
                old_ctrl = fF.control[w]
                new_tgt  = TotalRateTarget(new_rate)
                fF.control[w] = replace_target(old_ctrl, new_tgt)
                # синхронизируем лимиты с таргетом
                lims = get(fF.limits, w, default_limits(old_ctrl))
                fF.limits[w] = merge(lims, as_limit(new_tgt))
            end
        end

        # Прямой прогон
        simres = simulate_reservoir(case; output_substates = use_ministeps, info_level = -1, sim_arg...)
        r = simres.result
        dt_mini = report_timesteps(r.reports; ministeps=use_ministeps)

        # Целевая: NPV на шаг
        function npv_obj(model, state, dt, step_info, forces_here)
            return npv_objective(model, state, dt, step_info, forces_here;
                timesteps = dt_mini,
                injectors = injectors,  # стоимость закачки
                producers = producers,  # выручка с добычи
                maximize  = true,
                oil_price = oil_price,
                gas_price = gas_price,
                water_price = water_price,
                oil_cost  = oil_cost,
                gas_cost  = gas_cost,
                water_cost = water_cost,
                discount_rate = discount_rate,
                kwarg...
            )
        end

        obj = Jutul.evaluate_objective(npv_obj, case.model, r.states, case.dt, case.forces)

        if !grad
            return obj
        end

        # Обратный прогон: градиент по управляющим воздействиями (целям продюсеров)
        if !haskey(cache, :storage)
            targets = Jutul.force_targets(case.model)
            targets[:Facility][:control] = :control
            targets[:Facility][:limits]  = nothing
            cache[:storage] = Jutul.setup_adjoint_forces_storage(
                case.model, r.states, case.forces, case.dt, npv_obj;
                state0 = case.state0, parameters = case.parameters, targets = targets,
                eachstep = eachstep, di_sparse = true
            )
        end

        dF, _, _ = Jutul.solve_adjoint_forces!(
            cache[:storage], case.model, r.states, r.reports, npv_obj, case.forces;
            state0 = case.state0, parameters = case.parameters
        )

        dX = zeros(n_ctrl, nstep_unique)
        for stepno in 1:nstep_unique
            fF = dF[stepno][:Facility]
            for (i, w) in enumerate(control_wells)
                # чувствительность цели по значению таргета скважины
                ctrl = fF.control[w]
                ∂obj_∂q = ctrl.target.value
                dX[i, stepno] = ∂obj_∂q * max_rate
            end
        end
        return (obj, vec(dX))
    end

    x0 = fill(base_rate/max_rate, n_ctrl*nstep_unique)

    F! = x -> f!(x; grad=false)
    function F_and_dF!(g, x)
        obj, grad = f!(x; grad=true)
        g .= grad
        return obj
    end
    function dF!(g, x)
        _, grad = f!(x; grad=true)
        g .= grad
        return g
    end

    return (case=case, x0=x0, obj=F!, F_and_dF=F_and_dF!, dF=dF!, lin_eq=NamedTuple())
end

# ---------- High-level driver (like original) ----------
base_rate = base_rate_from_producers(coarse_case)
println("Base rate (from producers) = $(base_rate) [SI units of volume/time]")

function optimize_producer_rates(steps; use_box_bfgs = true)
    setup = setup_producer_rate_optimization_objective(coarse_case, base_rate;
        max_rate_factor = 10.0,
        oil_price = 100.0, water_price = -10.0, water_cost = 5.0,
        discount_rate = 0.05,
        steps = steps,
        use_ministeps = true,
        verbose = true,
        sim_arg = (precond = :cprw, linear_solver = :bicgstab, info_level = -1),
    )
    if use_box_bfgs
        # Bound-constrained BFGS на unit-box (из Jutul)
        obj_best, x_best, hist = Jutul.unit_box_bfgs(setup.x0, setup.obj;
            maximize = true, lin_eq = setup.lin_eq
        )
        H = hist.val
    else
        lower = zeros(length(setup.x0)); upper = ones(length(setup.x0))
        results, x_best = lbfgsb(setup.F_and_dF, setup.dF, setup.x0;
            lb=lower, ub=upper, iprint=1, factr=1e12, maxfun=20, maxiter=20, m=20
        )
        H = results
    end
    return (setup.case, H, x_best)
end

# ---------- Run two strategies: one-set and per-step ----------
case1, hist1, x1 = optimize_producer_rates(:first)
case2, hist2, x2 = optimize_producer_rates(:each)

# ---------- Plots ----------
# Allocation (constant)
fig = Figure()
ax = Axis(fig[1,1], xlabel="Producer index", ylabel="Rate fraction (of max producer rate)", title="Constant producer-rate allocation")
scatter!(ax, 1:length(x1), x1)
fig

# Allocation per step
allocs = reshape(x2, length(x1), :)
fig = Figure()
ax = Axis(fig[1,1], xlabel="Report step", ylabel="Rate fraction (of max producer rate)", title="Producer-rate per step")
for i in axes(allocs, 1)
    lines!(ax, allocs[i, :], label = "Producer #$i")
end
axislegend()
fig

# NPV evolution
fig = Figure()
ax = Axis(fig[1,1], xlabel="LBFGS iteration", ylabel="Net present value (million USD)")
scatter!(ax, 1:length(hist1), hist1./1e6, label="Constant producer rates")
scatter!(ax, 1:length(hist2), hist2./1e6, marker=:x, label="Per-step producer rates")
axislegend(position=:rb)
fig

# ---------- Simulate and compare field curves ----------
ws0, states0 = simulate_reservoir(coarse_case; info_level = -1)
ws1, states1 = simulate_reservoir(case1;       info_level = -1)
ws2, states2 = simulate_reservoir(case2;       info_level = -1)

# Field measurables
f0 = reservoir_measurables(coarse_case, ws0, states0; type=:field)
f1 = reservoir_measurables(case1,       ws1, states1; type=:field)
f2 = reservoir_measurables(case2,       ws2, states2; type=:field)

bbl = si_unit(:stb)

# Cumulative oil
fig = Figure()
ax = Axis(fig[1,1], xlabel="Time / days", ylabel="Field oil production (accumulated, bbl)")
t = ws0.time ./ si_unit(:day)
lines!(ax, t, f0[:fopt].values./bbl, label="Base")
lines!(ax, t, f1[:fopt].values./bbl, label="Const prod-rate")
lines!(ax, t, f2[:fopt].values./bbl, label="Per-step prod-rate")
axislegend(position=:rb)
fig

# Cumulative water
fig = Figure()
ax = Axis(fig[1,1], xlabel="Time / days", ylabel="Field water production (accumulated, bbl)")
lines!(ax, t, f0[:fwpt].values./bbl, label="Base")
lines!(ax, t, f1[:fwpt].values./bbl, label="Const prod-rate")
lines!(ax, t, f2[:fwpt].values./bbl, label="Per-step prod-rate")
axislegend(position=:rb)
fig
