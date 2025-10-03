# === History matching PERM по дебитам через adjoint + L-BFGS (как в доках) ===
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
    # ближайший отчетный шаг (устойчивее и быстрее)
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

# -------------------- 2) Setup-функция F_perm (ровно как в доках) --------------------
# Внутри оптимизации мы будем передавать prm["perm"] (в СИ: м^2) длины = числу ячеек,
# и расширять это в PERMX/PERMY/PERMZ (равные — пример из документации).
function F_perm(prm::AbstractDict, step_info = missing)
    data_c = deepcopy(data)
    # Базовый размер массива PERMX из DATA (обычно 5x5x1 для SPE1)
    sz = size(data_c["GRID"]["PERMX"])
    perm_vec = prm["perm"]
    length(perm_vec) == prod(sz) || error("Длина prm[\"perm\"]=$(length(perm_vec)) != prod(sz)=$(prod(sz))")
    permxyz = reshape(perm_vec, sz)
    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = permxyz
    return setup_case_from_data_file(data_c)
end

# -------------------- 3) Начальная точка: берем PERMX из DATA --------------------
# ВНИМАНИЕ: значения в DATA у JutulDarcy — в СИ (м^2).
perm0_SI = vec(data["GRID"]["PERMX"])        # длина = Nc
prm0 = Dict("perm" => copy(perm0_SI))

# -------------------- 4) Настройка оптимизации с L-BFGS (как в доках) --------------------
# Готовим структуру параметров-оптимизации из (prm0, F_perm)
dprm = setup_reservoir_dict_optimization(prm0, F_perm)

# Освобождаем параметр "perm" с коробочными относительными ограничениями
# (например, позволим варьировать от 0.2x до 5x относительно стартового значения)
free_optimization_parameter!(dprm, "perm"; rel_min = 0.2, rel_max = 5.0)

# Если хочешь "лумпинг" (аггрегировать ячейки по слоям - как в доках):
# rmesh   = physical_representation(reservoir_domain(case_truth.model))
# layerno = map(i -> cell_ijk(rmesh, i)[3], 1:number_of_cells(rmesh))
# free_optimization_parameter!(dprm, "perm"; rel_min=0.2, rel_max=5.0, lumping=layerno)

# -------------------- 5) Запуск оптимизации --------------------
Random.seed!(0)
perm_tuned = optimize_reservoir(dprm, rates_mismatch)

# dprm.history.val содержит историю значений цели по итерациям LBFGS
@info "Оптимизация завершена."

# -------------------- 6) Достаём оценённую проницаемость и переводим в мДарси --------------------
# Возвращается в том же формате, в каком передавали: prm["perm"] в СИ (м^2)
perm_SI = perm_tuned["perm"]                # Vector{Float64} длины Nc, в м^2
md_per_SI = 1000.0 / si_unit(:darcy)        # множитель перевода: м^2 -> мДарси
kx_mD = perm_SI .* md_per_SI                # равные для x,y,z — как в примере доков
ky_mD = copy(kx_mD)
kz_mD = copy(kx_mD)

# Печатаем первые 10 ячеек
println("\nПервые 10 ячеек (мДарси):")
for i in 1:min(10, length(kx_mD))
    @printf("%3d: kx=%.6f  ky=%.6f  kz=%.6f\n", i, kx_mD[i], ky_mD[i], kz_mD[i])
end

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
