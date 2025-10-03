# === History matching PERM по дебитам через adjoint + L-BFGS ===
using Jutul, JutulDarcy
using GeoEnergyIO
using Statistics, Printf
using Random

# -------------------- 0) Вводные --------------------
# Укажи свой путь к SPE1.DATA
datafile = joinpath("/media/oleg/E8C0040CC003E024/Tnavigator_models/msc_spe1", "SPE1.DATA")

# Загружаем DATA и базовый кейс ("истина" для наблюдений)
data = GeoEnergyIO.parse_data_file(datafile)
case_truth = setup_case_from_data_file(data)
res_truth  = simulate_reservoir(case_truth; info_level=-1)

# Скважины и наблюдения (истина) по дебитам
wells = [:INJ, :PROD]
obs_oil = Dict(w => collect(res_truth.wells[w][:orat]) for w in wells)
obs_wat = Dict(w => collect(res_truth.wells[w][:wrat]) for w in wells)

# Временные шаги (как в доках: используем cumsum(dt) + шаг по ближайшему времени)
step_times  = cumsum(case_truth.dt)
total_time  = step_times[end]

# -------------------- 1) Целевая функция (MSE по дебитам) --------------------
function rates_mismatch(model, state, dt, step_info, forces)
    # время на конце текущего шага
    t = step_info[:time] + dt
    # ближайший отчетный шаг
    step = searchsortedlast(step_times, t)
    step = clamp(step, 1, length(step_times))

    s = 0.0
    for w in wells
        qo = JutulDarcy.compute_well_qoi(model, state, forces, w, :orat)
        qw = JutulDarcy.compute_well_qoi(model, state, forces, w, :wrat)
        err_o = qo - obs_oil[w][step]
        err_w = qw - obs_wat[w][step]
        s += 0.5*(err_o^2 + err_w^2)
    end
    return s / length(wells)
end

# -------------------- 2) Setup-функция F_perm --------------------
# Внутри оптимизации мы будем передавать prm["perm"] (в СИ: м^2) длины = числу ячеек,
# передаём три вектора такой же длины, как PERMX в DATA.
function F_perm(prm::AbstractDict, step_info = missing)
    data_c = deepcopy(data)
    sz = size(data_c["GRID"]["PERMX"])  # например (5,5,1)

    kx = reshape(prm["kx"], sz)
    ky = reshape(prm["ky"], sz)
    kz = reshape(prm["kz"], sz)

    data_c["GRID"]["PERMX"] = kx
    data_c["GRID"]["PERMY"] = ky
    data_c["GRID"]["PERMZ"] = kz

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
kz_start_mD = 15.0    # << истина 50

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

# Полная свобода по всем ячейкам и направлениям (относительные коробочные ограничения)
for p in ("kx", "ky", "kz")
    free_optimization_parameter!(dprm, p; rel_min = 0.2, rel_max = 20.0)
end

# -------------------- 5) Запуск оптимизации --------------------
Random.seed!(0)
perm_tuned = optimize_reservoir(dprm, rates_mismatch;max_it = 50)

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
