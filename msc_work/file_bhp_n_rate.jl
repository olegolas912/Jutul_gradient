# === History matching PERM по дебитам через adjoint + L-BFGS ===
using Jutul, JutulDarcy
using GeoEnergyIO
using Statistics, Printf
using Random

const RATE_SCALE = 1.0 / si_unit(:day)

# -------------------- 0) Вводные --------------------
# Укажи свой путь к SPE1.DATA
datafile = joinpath("/media/oleg/E8C0040CC003E024/Tnavigator_models/msc_spe1", "SPE1.DATA")

# Загружаем DATA и базовый кейс ("истина" для наблюдений)
data = GeoEnergyIO.parse_data_file(datafile)
case_truth = setup_case_from_data_file(data)
res_truth  = simulate_reservoir(case_truth; info_level=-1)

# Скважины и наблюдения (истина) по дебитам
wells = [:INJ, :PROD]
obs_oil = Dict(w => collect(res_truth.wells[w][:orat]) ./ RATE_SCALE for w in wells)
obs_wat = Dict(w => collect(res_truth.wells[w][:wrat]) ./ RATE_SCALE for w in wells)
BHP_SCALE = si_unit(:bar)
obs_bhp = Dict(w => collect(res_truth.wells[w][:bhp]) ./ BHP_SCALE for w in wells)

# Временные шаги (как в доках: используем cumsum(dt) + шаг по ближайшему времени)
step_times  = collect(res_truth.time)
total_time  = step_times[end]

const RATE_REL_FLOOR = 1.0        # см^3/сут для нормировки относительной ошибки
const RATE_WEIGHT = 1.0e3         # усиливаем вклад дебитов
const BHP_WEIGHT = 1.0e3         # работаем в барах
const PERM_LOWER_SCALE = 0.1
const PERM_UPPER_SCALE = 1.1

# -------------------- 1) Целевая функция (MSE по дебитам) --------------------
function rates_mismatch(model, state, dt, step_info, forces)
    t = step_info[:time] + dt
    step = searchsortedlast(step_times, t)
    step = clamp(step, 1, length(step_times))

    s = 0.0
    for w in wells
        qo = JutulDarcy.compute_well_qoi(model, state, forces, w, :orat) / RATE_SCALE
        qw = JutulDarcy.compute_well_qoi(model, state, forces, w, :wrat) / RATE_SCALE
        qb = JutulDarcy.compute_well_qoi(model, state, forces, w, :bhp) / BHP_SCALE

        qoref = obs_oil[w][step]
        qwref = obs_wat[w][step]
        qbref = obs_bhp[w][step]

        err_o = (qo - qoref) / max(abs(qoref), RATE_REL_FLOOR)
        err_w = (qw - qwref) / max(abs(qwref), RATE_REL_FLOOR)
        err_b = qb - qbref

        s += RATE_WEIGHT * 0.5 * (err_o^2 + err_w^2)
        s += BHP_WEIGHT * err_b^2
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

# -------------------- 4) Настройка оптимизации с L-BFGS --------------------
dprm = setup_reservoir_dict_optimization(prm0, F_perm)

# Коробочные ограничения по каждой ячейке
for (p, vals, init) in zip(
    ("kx", "ky", "kz"),
    (perm_true_x_SI, perm_true_y_SI, perm_true_z_SI),
    (kx0_SI, ky0_SI, kz0_SI),
)
    lo = PERM_LOWER_SCALE .* min.(vals, init)
    hi = PERM_UPPER_SCALE .* max.(vals, init)
    free_optimization_parameter!(dprm, p; abs_min = lo, abs_max = hi, scaler = :log)
end

# -------------------- 5) Запуск оптимизации --------------------
Random.seed!(0)
perm_tuned = optimize_reservoir(
    dprm,
    rates_mismatch;
    max_it = 150,
    step_init = 1e-2,
    max_initial_update = 5e-2,
)

# dprm.history.val содержит историю значений цели по итерациям LBFGS
@info "Оптимизация завершена."

# -------------------- 6) Достаём оценённую проницаемость и переводим в мДарси --------------------
kx_SI = perm_tuned["kx"]              # в м^2, длина = Nc
ky_SI = perm_tuned["ky"]
kz_SI = perm_tuned["kz"]

md_per_SI = 1000.0 / si_unit(:darcy)  # м^2 -> мДарси
kx_mD = kx_SI .* md_per_SI
ky_mD = ky_SI .* md_per_SI
kz_mD = kz_SI .* md_per_SI

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
@info @sprintf("TUNED (мД): kx=%.1f, ky=%.1f, kz=%.1f", kx_mD[1], ky_mD[1], kz_mD[1])
