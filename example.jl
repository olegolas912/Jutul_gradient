using Jutul, JutulDarcy, GeoEnergyIO, LBFGSB
using LinearAlgebra: dot, norm  # <-- нужно для dot(), norm()

# 1) Загружаем и коэрсим тестовый кейс EGG (50 отчётных шагов)
data_pth  = joinpath(GeoEnergyIO.test_input_file_path("EGG"), "EGG.DATA")
fine_case = setup_case_from_data_file(data_pth)
case      = coarsen_reservoir_case(fine_case[1:50], (20,20,3), method = :ijk)

# 2) Базовый таргет закачки (будет верхней шкалой для долей)
ctrl      = case.forces[1][:Facility].control
base_rate = ctrl[:INJECT1].target.value

# 3) Строим цель (NPV) и «проводку» управлений
setup = JutulDarcy.setup_rate_optimization_objective(
    case, base_rate;
    max_rate_factor = 10,
    oil_price   = 100.0,   # $/bbl
    water_price = -10.0,   # $/bbl (штраф за воду в продукции)
    water_cost  = 5.0,     # $/bbl (стоимость закачки)
    discount_rate = 0.05,  # 5%/год
    steps = :each          # доли можно менять на каждом отчётном шаге
)

# 4) Значение цели и градиент в стартовой точке x0
x  = copy(setup.x0)                # x: доли ∈ [0,1]^m
f0 = setup.F!(x)                   # f0 = J(x)
g  = zeros(length(x))
setup.dF!(g, x)                    # g = ∇J(x)

println("NPV в x0: ", f0)
println("||∇J(x)||: ", norm(g))
println("Первые 8 компонент ∇J(x): ", g[1:min(length(g), 8)])

# 5) Проверка градиента (направленная производная)
ε = 1e-6
d = zeros(length(x)); d[1] = 1.0          # тестовое направление
f_plus = setup.F!(x .+ ε .* d)
approx = (f_plus - f0) / ε                # конечная разность
dotgrad = dot(g, d)                       # ∇J⋅d
println("Проверка: finite-diff = $approx, grad·d = $dotgrad, |diff| = ", abs(approx - dotgrad))
