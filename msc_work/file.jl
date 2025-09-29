############### TOP-LEVEL IMPORTS ###############
using JutulDarcy
using Jutul
using Random
using Statistics
using GLMakie

# Базовые юниты из Jutul/JutulDarcy
Darcy, bar, kg, meter, day_unit = si_units(:darcy, :bar, :kilogram, :meter, :day) # OK по докам

############### СЕТКА/ГЕОМЕТРИЯ ###############
nx, ny, nz = 50, 50, 1
Lx, Ly, Lz = 100.0, 100.0, 1.0
cart_dims   = (nx, ny, nz)
physical_sz = (Lx, Ly, Lz) .* meter
g    = CartesianMesh(cart_dims, physical_sz)
nc  = prod(cart_dims)

# Если пористость ещё не определена где-то раньше, возьмём 0.2
poro = @isdefined(poro) ? poro : fill(0.2, nc)

# Индексации
cell_index(i, j; nx=nx) = (j-1)*nx + i
left_cells  = [cell_index(1,  j)   for j in 1:ny]
right_cells = [cell_index(nx, j)   for j in 1:ny]

############### НАСТРОЙКА ФИЗИКИ ###############
phases = (LiquidPhase(), VaporPhase())
rhoLS  = 1000.0 * kg/meter^3
rhoGS  = 100.0  * kg/meter^3
sys = ImmiscibleSystem(phases, reference_densities=[rhoLS, rhoGS])

# ВЯЗКОСТИ (Па·с)
μL = 1.0e-3
μG = 1.0e-5

############### ФУНКЦИИ ДЛЯ ПРОНИЦАЕМОСТИ И СЛУЧАЯ ###############
# ДОЛЖНА существовать у вас:
# k_from_zones(theta)::Vector{Float64}  # длина = nc, единицы: м^2

function k_from_zones(theta::AbstractMatrix{<:Real})
    sx, sy = size(theta)
    sx == 5 || throw(ArgumentError("theta must have 5 rows"))
    sy == 5 || throw(ArgumentError("theta must have 5 columns"))
    fx = nx ÷ sx
    fy = ny ÷ sy
    k = Vector{Float64}(undef, nc)
    @inbounds for j in 1:ny
        jj = ((j - 1) ÷ fy) + 1
        for i in 1:nx
            ii = ((i - 1) ÷ fx) + 1
            idx = cell_index(i, j)
            k[idx] = exp(theta[ii, jj]) * 1.0e-3 * Darcy
        end
    end
    return k
end

const zone_dims = (5, 5)
const zone_fx = nx ÷ zone_dims[1]
const zone_fy = ny ÷ zone_dims[2]

const zone_cells = let cells = Vector{Vector{Int}}(undef, prod(zone_dims))
    idx = 1
    for jj in 1:zone_dims[2], ii in 1:zone_dims[1]
        block = Vector{Int}(undef, zone_fx * zone_fy)
        pos = 1
        for fy_idx in 1:zone_fy, fx_idx in 1:zone_fx
            x = (ii - 1) * zone_fx + fx_idx
            y = (jj - 1) * zone_fy + fy_idx
            block[pos] = cell_index(x, y)
            pos += 1
        end
        cells[idx] = block
        idx += 1
    end
    cells
end

function make_case(k_m2::Vector{Float64})
    @assert length(k_m2) == nc "Длина k_m2 должна равняться числу ячеек"

    # 1) Домен
    domain = reservoir_domain(g;
        permeability = k_m2,
        porosity     = poro
    )

    # 2) Модель + параметры
    model, parameters = setup_reservoir_model(domain, sys)

    # 3) Вязкости как параметры (по фазам и ячейкам)
    parameters[:Reservoir][:PhaseViscosities] = [fill(μL, nc) fill(μG, nc)]'

    # 4) Начальное состояние
    state0 = setup_reservoir_state(model;
        Pressure    = 100.0 * bar,
        Saturations = [1.0, 0.0]
    )

    # 5) Граничные условия по давлению (левая/правая грани)
    hL, hR = 102.0, 100.0    # м водяного столба
    ρg = 9.81e3               # Па/м
    pL = hL * ρg
    pR = hR * ρg
    bcL = flow_boundary_condition(left_cells,  domain, pL)
    bcR = flow_boundary_condition(right_cells, domain, pR)
    forces = setup_reservoir_forces(model, bc = vcat(bcL, bcR))

    # 6) Временные шаги
    nstep = 60
    dt = fill(1.0 * day_unit, nstep)

    return JutulCase(model, dt, forces; state0 = state0, parameters = parameters)
end

############### «СКВАЖИНЫ»/НАБЛЮДАТЕЛИ (ячейки) ###############
Random.seed!(42)
theta_true = [log(50.0 + 20.0*sin(π*i/5)*cos(π*j/5)) for i=1:5, j=1:5] # мД (лог-шкала)
k_true = k_from_zones(theta_true)  # в м^2, длина = nc

cell_idx(ix, iy; nx=nx, ny=ny) = (iy-1)*nx + ix
obs_ij = [(5,5),(25,5),(45,5),
          (5,25),(25,25),(45,25),
          (5,45),(25,45)]
obs_cells = [cell_idx(i,j) for (i,j) in obs_ij]

############### «ИСТИНА» И МОДЕЛИРОВАНИЕ ###############
case_true = make_case(k_true)

# OK-вызов: (case::JutulCase)
res_true = simulate_reservoir(case_true)
wd_true, states_true, time_true = res_true

# FIX: Давление берём как [:Pressure]
p_series_true = [states_true[t][:Pressure][obs_cells] for t in 1:length(states_true)]

# Шум в тренировочной части (дни 1–40)
obs_train = [p .*(1 .+ 0.005*randn(length(p))) for p in p_series_true[1:40]]
obs_blind = p_series_true[41:end]


struct PressureHistoryObjective
    obs::Vector{Vector{Float64}}
    cells::Vector{Int}
end

@inline function step_counter(info)
    for key in (:step, :report_step, :report_step_index)
        haskey(info, key) && return info[key]
    end
    error("step index not found")
end

@inline function reservoir_pressure(state)
    if :Reservoir in keys(state)
        return state[:Reservoir][:Pressure]
    elseif :Pressure in keys(state)
        return state[:Pressure]
    else
        error("Pressure not found in state")
    end
end

function (obj::PressureHistoryObjective)(model, state, dt, step_info, forces)
    idx = step_counter(step_info)
    idx > length(obj.obs) && return zero(dt)
    p = reservoir_pressure(state)[obj.cells]
    d = p .- obj.obs[idx]
    return 0.5 * sum(d .* d)
end


function loss_rmse(theta_vec)
    theta = reshape(theta_vec, (5,5))
    k_try = k_from_zones(theta)
    case  = make_case(k_try)

    # FIX: правильный вызов симуляции + получаем states
    res = try
        simulate_reservoir(case; info_level = -1)
    catch err
        @warn "Simulation failed in loss_rmse" exception = (err, catch_backtrace())
        return Inf
    end
    wd, st, tt = res

    # сравнение по дням 1–40
    s = 0.0
    M = length(obs_cells)
    T = 40
    nstep = min(T, length(st))
    nstep == 0 && return Inf
    @inbounds for t in 1:nstep
        # FIX: путь к давлению
        p = st[t][:Pressure][obs_cells]
        y = obs_train[t]
        @simd for m in 1:M
            s += (p[m] - y[m])^2
        end
    end
    return sqrt(s / (nstep*M))
end

function zones_gradient(g_perm::AbstractVector, theta::AbstractMatrix)
    grad = similar(theta, Float64)
    k = k_from_zones(theta)
    for jj in 1:zone_dims[2], ii in 1:zone_dims[1]
        idx = (ii - 1) + (jj - 1) * zone_dims[1] + 1
        acc = 0.0
        cells = zone_cells[idx]
        @inbounds for cell in cells
            acc += g_perm[cell] * k[cell]
        end
        grad[ii, jj] = acc
    end
    return grad
end

function gradient_step!(θ, θ_shape, objective, η, θ_min, θ_max)
    θ_mat = reshape(θ, θ_shape)
    case = make_case(k_from_zones(θ_mat))
    dopt = setup_reservoir_dict_optimization(case)
    free_optimization_parameter!(dopt, [:model, :permeability])
    grad_dict = parameters_gradient_reservoir(dopt, objective)
    δθ = vec(zones_gradient(grad_dict[:model][:permeability], θ_mat))
    any(!isfinite, δθ) && return false
    θ .= clamp.(θ .- η .* δθ, θ_min, θ_max)
    return true
end

# Инициализация
theta0 = fill(log(50.0), 5, 5)
struct HMLog
    it::Int
    rmse::Float64
    secs::Float64
end

function adjoint_gradient(theta0; niter = 10, η = 1.0e-7)
    θ = vec(theta0)
    logs = HMLog[]
    snaps = [k_from_zones(theta0)]
    bestθ = copy(θ)
    best_rmse = loss_rmse(bestθ)
    objective = PressureHistoryObjective(obs_train, obs_cells)
    θ_min, θ_max = log(5.0), log(500.0)
    t0 = time()
    sizeθ = size(theta0)
    for it in 1:niter
        ok = try
            gradient_step!(θ, sizeθ, objective, η, θ_min, θ_max)
        catch err
            @warn "Adjoint gradient failed" exception = (err, catch_backtrace())
            false
        end
        ok || break
        rmse = loss_rmse(θ)
        push!(logs, HMLog(it, rmse, time() - t0))
        push!(snaps, k_from_zones(reshape(θ, sizeθ)))
        if rmse < best_rmse
            best_rmse = rmse
            bestθ = copy(θ)
        end
        @info "Adjoint iter=$it  rmse=$(round(rmse, digits=4))"
    end
    return bestθ, logs, best_rmse, time() - t0, snaps
end

bestθ, hm_logs, best_rmse, total_secs, k_snaps = adjoint_gradient(theta0)

# Итоговая модель
k_est = k_from_zones(reshape(bestθ, (5, 5)))
case_est = make_case(k_est)

# FIX: корректный вызов симуляции и states
wd_est, states_est, time_est = simulate_reservoir(case_est)

# Метрики
function rmse_series(pred, ref)
    s = 0.0; n = 0
    for t in eachindex(pred)
        p = pred[t]; r = ref[t]
        s += mean((p .- r).^2); n += 1
    end
    sqrt(s/n)
end

pick_steps(n, count) = sort(unique(round.(Int, range(1, n, length = min(count, n)))))

function plot_reservoir_fields(mesh, states, times; count = 3)
    GLMakie.activate!()
    idxs = pick_steps(length(states), count)
    fig = Figure(size = (360 * length(idxs), 600))
    for (col, idx) in enumerate(idxs)
        tday = times[idx] / day_unit
        axp = Axis(fig[1, col], title = "Pressure t=$(round(tday, digits=1)) d")
        plot_cell_data!(axp, mesh, states[idx][:Pressure] ./ bar; colormap = :thermal)
        axs = Axis(fig[2, col], title = "Water sat t=$(round(tday, digits=1)) d")
        plot_cell_data!(axs, mesh, states[idx][:Saturations][1, :]; colormap = :tempo, colorrange = (0.0, 1.0))
    end
    fig
end

function plot_permeability_maps(mesh, snaps)
    GLMakie.activate!()
    init_md = snaps[1] ./ Darcy .* 1.0e3
    final_md = snaps[end] ./ Darcy .* 1.0e3
    delta_pct = 100 .* (final_md .- init_md) ./ init_md
    cr = maximum(abs.(delta_pct))
    cr = cr == 0 ? 1.0 : cr
    fig = Figure(size = (1080, 400))
    ax1 = Axis(fig[1, 1], title = "Initial k (mD)")
    plot_cell_data!(ax1, mesh, init_md; colormap = :solar)
    ax2 = Axis(fig[1, 2], title = "Estimated k (mD)")
    plot_cell_data!(ax2, mesh, final_md; colormap = :solar)
    ax3 = Axis(fig[1, 3], title = "Δk (%)")
    plot_cell_data!(ax3, mesh, delta_pct; colormap = :balance, colorrange = (-cr, cr))
    fig
end

# FIX: путь к давлению
pred_train = [states_est[t][:Pressure][obs_cells] for t in 1:40]
pred_blind = [states_est[t][:Pressure][obs_cells] for t in 41:60]

RMSE_train = rmse_series(pred_train, obs_train)
RMSE_blind = rmse_series(pred_blind, obs_blind)

println("== METRICS ==")
println("Total time (s): ", round(total_secs, digits=2))
println("RMSE train:     ", round(RMSE_train, digits=4))
println("RMSE blind:     ", round(RMSE_blind, digits=4))
println("Iters used:     ", length(hm_logs))

fields_fig = plot_reservoir_fields(g, states_est, time_est)
perm_fig = plot_permeability_maps(g, k_snaps)
display(fields_fig)
display(perm_fig)
