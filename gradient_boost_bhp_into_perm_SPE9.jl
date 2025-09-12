using Jutul, JutulDarcy, GeoEnergyIO
using LinearAlgebra, Statistics
using Printf: @printf

# ========== 0) ДАННЫЕ ==========
const DATA_FILE = joinpath(GeoEnergyIO.test_input_file_path("SPE9"), "SPE9.DATA")
raw0 = parse_data_file(DATA_FILE)
base_case = setup_case_from_data_file(raw0)

# ========== 1) ПАРАМЕТРИЗАЦИЯ K ==========
"""
_get_grid_dict(d)
Возвращает словарь GRID вне зависимости от типа ключа (String/Symbol).
"""
function _get_grid_dict(d)
    for k in ("GRID", :GRID, "Grid", :grid)
        if haskey(d, k)
            return d[k]
        end
    end
    error("В raw-данных отсутствует раздел GRID")
end
"""
_get_grid_array(gd, name)
Безопасно достаёт массив из раздела GRID по имени ключа (e.g. "PERMX").
Пробует String и Symbol варианты, а также регистры.
"""
function _get_grid_array(gd, name::AbstractString)
    cand = (name, uppercase(name), lowercase(name))
    for c in cand
        for k in (c, Symbol(c))
            if haskey(gd, k)
                return copy(gd[k])
            end
        end
    end
    error("В GRID нет ключа $(name)")
end
grid0 = _get_grid_dict(raw0)
Kx0 = _get_grid_array(grid0, "PERMX")
Ky0 = _get_grid_array(grid0, "PERMY")
Kz0 = _get_grid_array(grid0, "PERMZ")

@assert size(Kx0) == size(Ky0) == size(Kz0) "PERM-тензоры имеют разные размеры"
@assert ndims(Kx0) == 3 "Ожидается 3D массив PERM, получено ndims=$(ndims(Kx0))"
const SHAPE = size(Kx0)::NTuple{3,Int}
const NCELL = prod(SHAPE)

pack3(θx, θy, θz) = vcat(vec(θx), vec(θy), vec(θz))
function unpack3(θ::AbstractVector, shape::NTuple{3,Int})
    n = prod(shape)
    θx = reshape(@view(θ[1:n]), shape)
    θy = reshape(@view(θ[n+1:2n]), shape)
    θz = reshape(@view(θ[2n+1:3n]), shape)
    return θx, θy, θz
end

function apply_perm!(raw_base, θ::AbstractVector)
    θx, θy, θz = unpack3(θ, SHAPE)
    raw = deepcopy(raw_base)
    gd = _get_grid_dict(raw)
    gd["PERMX"] = Kx0 .* exp.(θx)
    gd["PERMY"] = Ky0 .* exp.(θy)
    gd["PERMZ"] = Kz0 .* exp.(θz)
    return raw
end

# единицы
const M2_PER_mD = 9.869233e-16
_to_mD(x) = x ./ M2_PER_mD

# лог-RMSE по положительным, с векторизацией маски
_log_rmse_masked(a, b) = begin
    va = vec(a); vb = vec(b)
    m = (va .> 0) .& (vb .> 0)
    any(m) ? sqrt(mean((log.(va[m]) .- log.(vb[m])).^2)) : NaN
end

function eval_K_short(θ; Kx_true=Kx0, Ky_true=Ky0, Kz_true=Kz0)
    θx, θy, θz = unpack3(θ, SHAPE)
    Kx = Kx0 .* exp.(θx); Ky = Ky0 .* exp.(θy); Kz = Kz0 .* exp.(θz)

    # RMSE в мД (векторизуем всё)
    rx = sqrt(mean((vec(_to_mD(Kx)) .- vec(_to_mD(Kx_true))).^2))
    ry = sqrt(mean((vec(_to_mD(Ky)) .- vec(_to_mD(Ky_true))).^2))
    rz = sqrt(mean((vec(_to_mD(Kz)) .- vec(_to_mD(Kz_true))).^2))

    rall = sqrt(mean(vcat(
        vec(_to_mD(Kx)) .- vec(_to_mD(Kx_true)),
        vec(_to_mD(Ky)) .- vec(_to_mD(Ky_true)),
        vec(_to_mD(Kz)) .- vec(_to_mD(Kz_true))
    ).^2))

    # log-RMSE по каждой оси
    lx = _log_rmse_masked(Kx, Kx_true)
    ly = _log_rmse_masked(Ky, Ky_true)
    lz = _log_rmse_masked(Kz, Kz_true)

    # log-RMSE "ALL": собираем общий вектор и соответствующую маску
    va = vcat(vec(Kx), vec(Ky), vec(Kz))
    vb = vcat(vec(Kx_true), vec(Ky_true), vec(Kz_true))
    m_all = (va .> 0) .& (vb .> 0)
    lall = any(m_all) ? sqrt(mean((log.(va[m_all]) .- log.(vb[m_all])).^2)) : NaN

    @printf("\nRMSE (мД):   X=%.3g  Y=%.3g  Z=%.3g  | ALL=%.3g\n", rx, ry, rz, rall)
    @printf("log-RMSE:     X=%.3g  Y=%.3g  Z=%.3g  | ALL=%.3g\n", lx, ly, lz, lall)
    return (Kx_rec=Kx, Ky_rec=Ky, Kz_rec=Kz)
end

# ========== 2) ПРОГОН И BHP ==========
"""
simulate_bhp(raw_mod) -> (well_names::Vector{String}, times::Vector{Float64}, B::Matrix{Float64})
B: (n_wells, n_steps), строки соответствуют well_names.
Подправь доступ к отчётам под свою структуру sim.
"""
function simulate_bhp(raw_mod)
    case = setup_case_from_data_file(raw_mod)
    sim = simulate_reservoir(case)
    # В актуальной структуре JutulDarcy результаты по скважинам лежат в sim.wells.wells
    wells_dict = sim.wells.wells  # Dict{Symbol, Dict{Symbol, Vector}}
    well_syms = sort(collect(keys(wells_dict)); by = x -> String(x))
    well_names = String.(well_syms)
    times = sim.time
    nW = length(well_syms)
    nT = length(times)
    B  = zeros(nW, nT)
    for (wi, wsym) in enumerate(well_syms)
        series = wells_dict[wsym][:bhp]
        @assert length(series) == nT
        @views B[wi, :] .= series
    end
    return well_names, times, B
end

# Наблюдения (для синтетики — возьмём из базового прогона)
θ_truth = clamp.(0.2 .* randn(3*NCELL), THETA_MIN, THETA_MAX)
raw_truth = apply_perm!(raw0, θ_truth)
well_order_obs, times_obs, BHP_obs = simulate_bhp(raw_truth)

# Сопоставление порядка скважин
function align_rows(B_pred_names::Vector{String}, B_pred::AbstractMatrix,
                    obs_names::Vector{String})
    dict = Dict(name => i for (i, name) in enumerate(B_pred_names))
    idx = [dict[n] for n in obs_names]  # бросит ошибку, если имя не найдено
    return B_pred[idx, :]
end

# Ошибка по BHP (нормировка по σ каждой скважины)
function bhp_loss(B_pred::AbstractMatrix, B_obs::AbstractMatrix)
    @assert size(B_pred) == size(B_obs)
    nW, nT = size(B_obs)
    σ = vec(std(B_obs; dims=2)) .+ 1e-6
    L = 0.0
    for w in 1:nW
        @views r = B_pred[w, :] .- B_obs[w, :]
        L += sum((r ./ σ[w]).^2)
    end
    return L / (nW * nT)
end

# ========== 3) ЦЕЛЕВАЯ ФУНКЦИЯ ==========
const LAMBDA_PRIOR  = 1e-3   # слабый каркас к θ=0  (k≈k0)
const THETA_MIN     = -3.0   # множители ~ [e^-3, e^3] ≈ [0.05, 20]
const THETA_MAX     = +3.0

project!(θ) = (θ .= clamp.(θ, THETA_MIN, THETA_MAX); θ)

function objective_only(θ::AbstractVector)
    raw_mod = apply_perm!(raw0, θ)
    pred_names, _, B_pred_raw = simulate_bhp(raw_mod)
    B_pred = align_rows(pred_names, B_pred_raw, well_order_obs)
    L = bhp_loss(B_pred, BHP_obs)
    L += LAMBDA_PRIOR * dot(θ, θ)
    return L
end

# ========== 4) ТВОЙ ГРАДИЕНТ (adjoint) ==========
# ИДЕЯ: часто adjoint отдаёт ∂L/∂Kx, ∂L/∂Ky, ∂L/∂Kz на текущем прогоне.
# Тогда при K = K0 .* exp(θ) имеем: ∂L/∂θx = (∂L/∂Kx) ⊙ Kx   и т.д.

"""
adjoint_grad_K!(gKx, gKy, gKz, raw_mod, BHP_obs, well_order_obs)
Считает градиент по проницаемости через adjoint API JutulDarcy.
Возвращает ∂L/∂Kx, ∂L/∂Ky, ∂L/∂Kz в массивах gKx, gKy, gKz той же размерности, что и SHAPE.

Функционал L — средневзвешенная (по времени и по скважинам) квадратичная невязка BHP
между текущим прогоном и эталоном (BHP_obs), нормированная на σ каждой скважины.
"""
function adjoint_grad_K!(gKx, gKy, gKz, raw_mod, BHP_obs, well_order_obs)
    # 1) Сбор вспомогательных данных из наблюдений
    obs_wells_syms = Symbol.(well_order_obs)
    nW, nT = size(BHP_obs)
    @assert nW == length(obs_wells_syms)
    T = times_obs
    @assert nT == length(T)
    σ = vec(std(BHP_obs; dims=2)) .+ 1e-6

    # Быстрый помощник: ближайший индекс по времени
    nearest_index(t::Real, times::AbstractVector{<:Real}) = begin
        i = searchsortedfirst(times, t)
        if i <= 1
            return 1
        elseif i > length(times)
            return length(times)
        else
            (t - times[i-1] <= times[i] - t) ? (i - 1) : i
        end
    end

    # 2) Подготовка оптимизации по словарю параметров
    case = setup_case_from_data_file(raw_mod)
    dopt = setup_reservoir_dict_optimization(case)
    free_optimization_parameters!(dopt)

    # 3) Определяем целевую функцию для adjoint
    W_INV = 1.0 / nW
    T_INV = 1.0 / max(eps(), (T[end] - T[1]))
    function obj(m, s, dt, info, forces)
        t = info[:time] + dt
        k = nearest_index(t, T)
        acc = zero(dt)
        @inbounds for wi in 1:nW
            w = obs_wells_syms[wi]
            # BHP текущего состояния
            bhp_cur = JutulDarcy.compute_well_qoi(m, s, forces, w, :bhp)
            z = (bhp_cur - BHP_obs[wi, k]) / max(σ[wi], 1e-6)
            acc += z*z
        end
        return dt * W_INV * acc * T_INV
    end

    # 4) Градиент словарём
    gdict = parameters_gradient_reservoir(dopt, obj)
    gm = gdict[:model]
    haskey(gm, :permeability) || error("Adjoint не вернул градиент по :permeability")
    gperm = Array(gm[:permeability])  # ожидаем (3, NCELL)
    @assert size(gperm, 2) == prod(SHAPE)

    # 5) Раскладываем по осям и возвращаем
    gKx .= reshape(@view(gperm[1, :]), SHAPE)
    gKy .= reshape(@view(gperm[2, :]), SHAPE)
    gKz .= reshape(@view(gperm[3, :]), SHAPE)
    return nothing
end

"""
loss_and_grad(θ) -> (L, gθ)
Считает значение L и градиент по θ (через ∂L/∂K и цепное правило).
"""
function loss_and_grad(θ::Vector{Float64})
    # прямой прогон под текущим θ
    raw_mod = apply_perm!(raw0, θ)
    pred_names, _, B_pred_raw = simulate_bhp(raw_mod)
    B_pred = align_rows(pred_names, B_pred_raw, well_order_obs)

    # значение функционала
    L_data = bhp_loss(B_pred, BHP_obs)
    L_reg  = LAMBDA_PRIOR * dot(θ, θ)
    L = L_data + L_reg

    # запрос градиента по K от твоего adjoint
    θx, θy, θz = unpack3(θ, SHAPE)
    Kx = Kx0 .* exp.(θx); Ky = Ky0 .* exp.(θy); Kz = Kz0 .* exp.(θz)

    gKx = similar(Kx); gKy = similar(Ky); gKz = similar(Kz)
    adjoint_grad_K!(gKx, gKy, gKz, raw_mod, BHP_obs, well_order_obs)  # <--- ТВОЯ ФУНКЦИЯ

    # цепное правило: dL/dθ = dL/dK ⊙ K
    gθx = gKx .* Kx
    gθy = gKy .* Ky
    gθz = gKz .* Kz

    gθ = pack3(gθx, gθy, gθz)

    # + градиент от L2-prior
    @inbounds @simd for i in eachindex(gθ)
        gθ[i] += 2 * LAMBDA_PRIOR * θ[i]
    end

    return L, gθ
end

# ========== 5) ПРОСТОЙ ГРАДИЕНТНЫЙ СПУСК ==========
η = 5e-3            # фиксированный шаг (начни с 1e-3..1e-2)
maxiter = 4         # сократим для тестового прогона
θ = zeros(3*NCELL)   # старт: k = k0

println("iter  L           |g|        step")
for it in 1:maxiter
    L, g = loss_and_grad(θ)
    θ .-= η .* g
    project!(θ)  # держим в боксах
    @printf("%4d  %.6e  %.3e  %.2e\n", it, L, norm(g), η)
end

# ========== 6) РЕЗУЛЬТАТ ==========
θx, θy, θz = unpack3(θ, SHAPE)
Kx_rec = Kx0 .* exp.(θx)
Ky_rec = Ky0 .* exp.(θy)
Kz_rec = Kz0 .* exp.(θz)
θx_t, θy_t, θz_t = unpack3(θ_truth, SHAPE)
Kx_true = Kx0 .* exp.(θx_t)
Ky_true = Ky0 .* exp.(θy_t)
Kz_true = Kz0 .* exp.(θz_t)

# 2) печать mean/min/max в мД: rec | true
@printf("Kx (mD): rec mean=%.3f min=%.3f max=%.3f  |  true mean=%.3f min=%.3f max=%.3f\n",
        mean(_to_mD(Kx_rec)), minimum(_to_mD(Kx_rec)), maximum(_to_mD(Kx_rec)),
        mean(_to_mD(Kx_true)), minimum(_to_mD(Kx_true)), maximum(_to_mD(Kx_true)))

@printf("Ky (mD): rec mean=%.3f min=%.3f max=%.3f  |  true mean=%.3f min=%.3f max=%.3f\n",
        mean(_to_mD(Ky_rec)), minimum(_to_mD(Ky_rec)), maximum(_to_mD(Ky_rec)),
        mean(_to_mD(Ky_true)), minimum(_to_mD(Ky_true)), maximum(_to_mD(Ky_true)))

@printf("Kz (mD): rec mean=%.3f min=%.3f max=%.3f  |  true mean=%.3f min=%.3f max=%.3f\n",
        mean(_to_mD(Kz_rec)), minimum(_to_mD(Kz_rec)), maximum(_to_mD(Kz_rec)),
        mean(_to_mD(Kz_true)), minimum(_to_mD(Kz_true)), maximum(_to_mD(Kz_true)))

# 3) метрики считаем относительно этой "истины"
val = eval_K_short(θ; Kx_true=Kx0, Ky_true=Ky0, Kz_true=Kz0)