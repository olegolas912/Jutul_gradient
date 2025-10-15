# === History matching PERM по дебитам через adjoint + L-BFGS ===
using Jutul, JutulDarcy
using GeoEnergyIO
using Statistics, Printf
using Random

const RATE_SCALE = 1.0 / si_unit(:day)

# -------------------- 0) Вводные --------------------
# Укажи свой путь к SPE1.DATA
datafile = joinpath("/home/oleg/Github/Jutul_gradient/", "SPE1.DATA")

# Загружаем DATA и базовый кейс ("истина" для наблюдений)
data = GeoEnergyIO.parse_data_file(datafile)
# function relax_rate_limits!(deck::AbstractDict)
#     steps = deck["SCHEDULE"]["STEPS"]
#     for step in steps
#         if haskey(step, "WCONPROD")
#             for ctrl in step["WCONPROD"]
#                 ctrl[4] = Inf  # ORAT
#                 ctrl[5] = Inf  # WRAT
#                 ctrl[6] = Inf  # GRAT
#                 ctrl[7] = Inf  # LRAT
#                 ctrl[8] = Inf  # RESV
#             end
#         end
#         if haskey(step, "WCONINJE")
#             for ctrl in step["WCONINJE"]
#                 ctrl[5] = Inf  # RATE
#             end
#         end
#     end
#     return deck
# end
# relax_rate_limits!(data)
case_truth = setup_case_from_data_file(data)
res_truth  = simulate_reservoir(case_truth; info_level=-1)

# Скважины и наблюдения (истина) по дебитам
wells = [:PROD]
obs_oil = Dict(w => collect(res_truth.wells[w][:orat]) ./ RATE_SCALE for w in wells)
obs_wat = Dict(w => collect(res_truth.wells[w][:wrat]) ./ RATE_SCALE for w in wells)

# Временные шаги (используем фактическое время модели)
step_times = collect(res_truth.time)
total_time = step_times[end]

const RATE_WEIGHT = 1.0
const PERM_LOWER_SCALE = 0.1
const PERM_UPPER_SCALE = 1.05


# -------------------- 1) Целевая функция (MSE по дебитам) --------------------
function rates_mismatch(model, state, dt, step_info, forces)
    t = step_info[:time] + dt
    step = searchsortedlast(step_times, t)
    step = clamp(step, 1, length(step_times))

    s = 0.0
    for w in wells
        qo = JutulDarcy.compute_well_qoi(model, state, forces, w, :orat) / RATE_SCALE
        qw = JutulDarcy.compute_well_qoi(model, state, forces, w, :wrat) / RATE_SCALE

        qoref = obs_oil[w][step]
        qwref = obs_wat[w][step]

        err_o = qo - qoref
        err_w = qw - qwref

        s += RATE_WEIGHT * 0.5 * (err_o^2 + err_w^2)
    end

    return (dt / total_time) * s / length(wells)
end

# -------------------- 2) Setup-функция F_perm --------------------
# Внутри оптимизации мы будем передавать prm["perm"] (в СИ: м^2) длины = числу ячеек,
# передаём три вектора такой же длины, как PERMX в DATA.
function F_perm(prm::AbstractDict, step_info = missing)
    data_c = deepcopy(data)
    sz = size(data_c["GRID"]["PERMX"])  # например (5,5,1)

    data_c["GRID"]["PERMX"] = reshape(prm["kx"], sz)
    data_c["GRID"]["PERMY"] = reshape(prm["ky"], sz)
    data_c["GRID"]["PERMZ"] = reshape(prm["kz"], sz)

    return setup_case_from_data_file(data_c)
end

# -------------------- 3) Начальная точка: берём PERMX/PERMY/PERMZ из DATA и их меняем --------------------
# Значения в DATA у JutulDarcy — в СИ (м^2).
md_per_SI = 1000.0 / si_unit(:darcy)  # м² -> мДарси

perm_true_x_SI = vec(data["GRID"]["PERMX"])
perm_true_y_SI = vec(data["GRID"]["PERMY"])
perm_true_z_SI = vec(data["GRID"]["PERMZ"])

# Задаём «плохой» старт в мДарси:
kx_start_mD = 120.0   # << истина 500
ky_start_mD = 80.0    # << истина 500
kz_start_mD = 50.0    # берём true kz для стабильности

# Конвертируем в СИ и заполняем все ячейки константой
kx0_SI = fill(kx_start_mD / md_per_SI, length(perm_true_x_SI))
ky0_SI = fill(ky_start_mD / md_per_SI, length(perm_true_y_SI))
kz0_SI = fill(kz_start_mD / md_per_SI, length(perm_true_z_SI))

# Стартовый словарь параметров для оптимизации
prm0 = Dict("kx" => kx0_SI, "ky" => ky0_SI, "kz" => kz0_SI)

# Быстрый отчёт, чтобы видеть реальный старт (в мД)
@info @sprintf("START (мД): kx=%.1f, ky=%.1f, kz=%.1f", kx_start_mD, ky_start_mD, kz_start_mD)

# -------------------- 4) Настройка оптимизации --------------------
Random.seed!(0)
dprm = setup_reservoir_dict_optimization(prm0, F_perm)
for (p, vals, init) in zip(
    ("kx", "ky", "kz"),
    (perm_true_x_SI, perm_true_y_SI, perm_true_z_SI),
    (kx0_SI, ky0_SI, kz0_SI),
)
    lo = PERM_LOWER_SCALE .* min.(vals, init)
    hi = PERM_UPPER_SCALE .* max.(vals, init)
    free_optimization_parameter!(dprm, p; abs_min = lo, abs_max = hi, scaler = :log)
end

perm_tuned = optimize_reservoir(
    dprm,
    rates_mismatch;
    max_it = 120,
    step_init = 0.1,
    max_initial_update = 0.2,
)

@info "Оптимизация завершена."

# -------------------- 5) Достаём оценённую проницаемость и переводим в мДарси --------------------
kx_SI = perm_tuned["kx"]              # в м^2, длина = Nc
ky_SI = perm_tuned["ky"]
kz_SI = perm_tuned["kz"]

md_per_SI = 1000.0 / si_unit(:darcy)  # м^2 -> мДарси
kx_mD = kx_SI .* md_per_SI
ky_mD = ky_SI .* md_per_SI
kz_mD = kz_SI .* md_per_SI
true_kx_mD = perm_true_x_SI .* md_per_SI
true_ky_mD = perm_true_y_SI .* md_per_SI
true_kz_mD = perm_true_z_SI .* md_per_SI
function rmse(a, b)
    return sqrt(mean((a .- b).^2))
end
@info @sprintf("RMSE (мД): kx=%.2f, ky=%.2f, kz=%.2f", rmse(kx_mD, true_kx_mD), rmse(ky_mD, true_ky_mD), rmse(kz_mD, true_kz_mD))

# (опционально) сохраним CSV
function save_perm_csv(path::AbstractString, kx, ky, kz, sz::NTuple{3,Int})
    nx, ny, nz = sz
    open(path, "w") do io
        write(io, "i,j,k,kx_mD,ky_mD,kz_mD\n")
        for k in 1:nz, j in 1:ny, i in 1:nx
            idx = (k-1)*nx*ny + (j-1)*nx + i
            @printf(io, "%d,%d,%d,%.8f,%.8f,%.8f\n", i, j, k, kx[idx], ky[idx], kz[idx])
        end
    end
end

sz = size(data["GRID"]["PERMX"])  # например (5,5,1)
save_perm_csv("perm_lbfsg_mD.csv", kx_mD, ky_mD, kz_mD, sz)
@info "Сохранил kx,ky,kz (мДарси) в perm_lbfsg_mD.csv"
kx_mean_mD = mean(kx_mD)
ky_mean_mD = mean(ky_mD)
kz_mean_mD = mean(kz_mD)

@info @sprintf("TUNED MEAN (мД): kx=%.1f, ky=%.1f, kz=%.1f",
               kx_mean_mD, ky_mean_mD, kz_mean_mD)