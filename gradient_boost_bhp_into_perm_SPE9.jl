using Jutul, JutulDarcy, GeoEnergyIO
using LinearAlgebra, Statistics, Random
using Printf: @printf

# ========== 0) ДАННЫЕ ==========
const DATA_FILE = joinpath(GeoEnergyIO.test_input_file_path("SPE9"), "SPE9.DATA")
raw0 = parse_data_file(DATA_FILE)

# ========== 1) УТИЛИТЫ И СЕТКА ==========
function _get_grid_dict(d)
    for k in ("GRID", :GRID, "Grid", :grid)
        if haskey(d, k); return d[k]; end
    end
    error("В raw-данных отсутствует раздел GRID")
end
function _get_grid_array(gd, name::AbstractString)
    for c in (name, uppercase(name), lowercase(name))
        for k in (c, Symbol(c))
            if haskey(gd, k); return copy(gd[k]); end
        end
    end
    error("В GRID нет ключа $(name)")
end

grid0 = _get_grid_dict(raw0)
Kx0 = _get_grid_array(grid0, "PERMX")
Ky0 = _get_grid_array(grid0, "PERMY")
Kz0 = _get_grid_array(grid0, "PERMZ")
@assert size(Kx0) == size(Ky0) == size(Kz0)
@assert ndims(Kx0) == 3
const SHAPE = size(Kx0)::NTuple{3,Int}
const NCELL = prod(SHAPE)

# единицы
const M2_PER_mD = 9.869233e-16
_to_mD(x) = x ./ M2_PER_mD

# ========== 2) ПАРАМЕТРИЗАЦИЯ κ ∈ [0.05, 10] ==========
const LAMBDA_PRIOR = 1e-3
const KAPPA_MIN = 0.05
const KAPPA_MAX = 10.0
project!(κ) = (κ .= clamp.(κ, KAPPA_MIN, KAPPA_MAX); κ)

pack3(a, b, c) = vcat(vec(a), vec(b), vec(c))
function unpack3(v::AbstractVector, shape::NTuple{3,Int})
    n = prod(shape)
    v1 = reshape(@view(v[1:n]), shape)
    v2 = reshape(@view(v[n+1:2n]), shape)
    v3 = reshape(@view(v[2n+1:3n]), shape)
    return v1, v2, v3
end

function apply_perm!(raw_base, κ::AbstractVector)
    κx, κy, κz = unpack3(κ, SHAPE)
    raw = deepcopy(raw_base)
    gd = _get_grid_dict(raw)
    gd["PERMX"] = Kx0 .* κx
    gd["PERMY"] = Ky0 .* κy
    gd["PERMZ"] = Kz0 .* κz
    return raw
end

# метрика по логам (по положительным)
_log_rmse_masked(a, b) = begin
    va = vec(a); vb = vec(b)
    m = (va .> 0) .& (vb .> 0)
    any(m) ? sqrt(mean((log.(va[m]) .- log.(vb[m])).^2)) : NaN
end

# ========== 3) ПРОГОН И BHP ==========
function simulate_bhp(raw_mod)
    case = setup_case_from_data_file(raw_mod)
    sim = simulate_reservoir(case)
    wells_dict = sim.wells.wells
    well_syms = sort(collect(keys(wells_dict)); by = x -> String(x))
    well_names = String.(well_syms)
    times = sim.time
    nW = length(well_syms); nT = length(times)
    B = zeros(nW, nT)
    for (wi, wsym) in enumerate(well_syms)
        series = wells_dict[wsym][:bhp]
        @assert length(series) == nT
        @views B[wi, :] .= series
    end
    return well_names, times, B
end

# наблюдения из синтетической "истины"
Random.seed!(1234)
κ_truth = KAPPA_MIN .+ (KAPPA_MAX - KAPPA_MIN) .* rand(3*NCELL)
project!(κ_truth)
raw_truth = apply_perm!(raw0, κ_truth)
well_order_obs, times_obs, BHP_obs = simulate_bhp(raw_truth)

# сопоставление порядка скважин
function align_rows(B_pred_names::Vector{String}, B_pred::AbstractMatrix,
                    obs_names::Vector{String})
    dict = Dict(name => i for (i, name) in enumerate(B_pred_names))
    idx = [dict[n] for n in obs_names]
    return B_pred[idx, :]
end

# ошибка по BHP (нормировка по σ скважины)
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

# ========== 4) ADJOINT: ∂L/∂K ==========
function adjoint_grad_K!(gKx, gKy, gKz, raw_mod, BHP_obs, well_order_obs)
    obs_wells_syms = Symbol.(well_order_obs)
    nW, nT = size(BHP_obs)
    @assert nW == length(obs_wells_syms)
    T = times_obs
    @assert nT == length(T)
    σ = vec(std(BHP_obs; dims=2)) .+ 1e-6
    nearest_index(t::Real, times::AbstractVector{<:Real}) = begin
        i = searchsortedfirst(times, t)
        if i <= 1; return 1
        elseif i > length(times); return length(times)
        else; (t - times[i-1] <= times[i] - t) ? (i - 1) : i
        end
    end
    case = setup_case_from_data_file(raw_mod)
    dopt = setup_reservoir_dict_optimization(case)
    free_optimization_parameters!(dopt)
    W_INV = 1.0 / nW
    T_INV = 1.0 / max(eps(), (T[end] - T[1]))
    function obj(m, s, dt, info, forces)
        t = info[:time] + dt
        k = nearest_index(t, T)
        acc = zero(dt)
        @inbounds for wi in 1:nW
            w = obs_wells_syms[wi]
            bhp_cur = JutulDarcy.compute_well_qoi(m, s, forces, w, :bhp)
            z = (bhp_cur - BHP_obs[wi, k]) / max(σ[wi], 1e-6)
            acc += z*z
        end
        return dt * W_INV * acc * T_INV
    end
    gdict = parameters_gradient_reservoir(dopt, obj)
    gm = gdict[:model]
    haskey(gm, :permeability) || error("Adjoint не вернул градиент по :permeability")
    gperm = Array(gm[:permeability])  # (3, NCELL)
    @assert size(gperm, 2) == prod(SHAPE)
    gKx .= reshape(@view(gperm[1, :]), SHAPE)
    gKy .= reshape(@view(gperm[2, :]), SHAPE)
    gKz .= reshape(@view(gperm[3, :]), SHAPE)
    return nothing
end

# ========== 5) L И ГРАДИЕНТ ПО κ ==========
function loss_and_grad(κ::Vector{Float64})
    raw_mod = apply_perm!(raw0, κ)
    pred_names, _, B_pred_raw = simulate_bhp(raw_mod)
    B_pred = align_rows(pred_names, B_pred_raw, well_order_obs)
    L_data = bhp_loss(B_pred, BHP_obs)
    L_reg  = LAMBDA_PRIOR * sum((κ .- 1.0).^2)
    L = L_data + L_reg
    κx, κy, κz = unpack3(κ, SHAPE)
    gKx = similar(κx); gKy = similar(κy); gKz = similar(κz)
    adjoint_grad_K!(gKx, gKy, gKz, raw_mod, BHP_obs, well_order_obs)
    gκx = gKx .* Kx0; gκy = gKy .* Ky0; gκz = gKz .* Kz0
    gκ = pack3(gκx, gκy, gκz)
    @inbounds @simd for i in eachindex(gκ)
        gκ[i] += 2 * LAMBDA_PRIOR * (κ[i] - 1.0)
    end
    return L, gκ
end

# ========== 6) ОПТИМИЗАЦИЯ ==========
η = 5e-3
maxiter = 4
Random.seed!(42)
κ = KAPPA_MIN .+ (KAPPA_MAX - KAPPA_MIN) .* rand(3*NCELL)
project!(κ)

println("iter  L           |g|        step")
for it in 1:maxiter
    L, g = loss_and_grad(κ)
    κ .-= η .* g
    project!(κ)
    @printf("%4d  %.6e  %.3e  %.2e\n", it, L, norm(g), η)
end

# ========== 7) РЕЗУЛЬТАТ И МЕТРИКИ ==========
κx, κy, κz = unpack3(κ, SHAPE)
Kx_rec = Kx0 .* κx; Ky_rec = Ky0 .* κy; Kz_rec = Kz0 .* κz
κx_t, κy_t, κz_t = unpack3(κ_truth, SHAPE)
Kx_true = Kx0 .* κx_t; Ky_true = Ky0 .* κy_t; Kz_true = Kz0 .* κz_t

@printf("Kx (mD): rec mean=%.3f min=%.3f max=%.3f  |  true mean=%.3f min=%.3f max=%.3f\n",
    mean(_to_mD(Kx_rec)), minimum(_to_mD(Kx_rec)), maximum(_to_mD(Kx_rec)),
    mean(_to_mD(Kx_true)), minimum(_to_mD(Kx_true)), maximum(_to_mD(Kx_true)))
@printf("Ky (mD): rec mean=%.3f min=%.3f max=%.3f  |  true mean=%.3f min=%.3f max=%.3f\n",
    mean(_to_mD(Ky_rec)), minimum(_to_mD(Ky_rec)), maximum(_to_mD(Ky_rec)),
    mean(_to_mD(Ky_true)), minimum(_to_mD(Ky_true)), maximum(_to_mD(Ky_true)))
@printf("Kz (mD): rec mean=%.3f min=%.3f max=%.3f  |  true mean=%.3f min=%.3f max=%.3f\n",
    mean(_to_mD(Kz_rec)), minimum(_to_mD(Kz_rec)), maximum(_to_mD(Kz_rec)),
    mean(_to_mD(Kz_true)), minimum(_to_mD(Kz_true)), maximum(_to_mD(Kz_true)))

rx = sqrt(mean((vec(_to_mD(Kx_rec)) .- vec(_to_mD(Kx_true))).^2))
ry = sqrt(mean((vec(_to_mD(Ky_rec)) .- vec(_to_mD(Ky_true))).^2))
rz = sqrt(mean((vec(_to_mD(Kz_rec)) .- vec(_to_mD(Kz_true))).^2))
rall = sqrt(mean(vcat(
    vec(_to_mD(Kx_rec)) .- vec(_to_mD(Kx_true)),
    vec(_to_mD(Ky_rec)) .- vec(_to_mD(Ky_true)),
    vec(_to_mD(Kz_rec)) .- vec(_to_mD(Kz_true))
).^2))

@printf("\nRMSE (мД):   X=%.3g  Y=%.3g  Z=%.3g  | ALL=%.3g\n", rx, ry, rz, rall)
