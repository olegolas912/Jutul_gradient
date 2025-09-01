Ниже — «как рождаются» градиенты в твоём примере, что именно надо вызвать, какие переменные где лежат и зачем.

# Что такое градиент здесь

Это адъюнктная чувствительность цели $J$ по параметрам пласта $\theta$ (пористость, проницаемость и т.п.). Схема: один прямой прогон (сохраняем состояния по времени) + один обратный прогон (решаем адъюнкт и собираем $\partial J/\partial \theta$ за все шаги).

Математически (по-простому):
состояния $s_{k+1}=F(s_k,\theta,u_k)$, цель $J=\sum_k \phi(s_{k+1})$.
Адъюнкт $\lambda_k = (\partial F/\partial s_k)^\top \lambda_{k+1} + \partial\phi/\partial s_k$,
градиент $\displaystyle \frac{\partial J}{\partial \theta}=\sum_k (\partial F/\partial \theta)^\top \lambda_{k+1}$.

# Мини-пайплайн (код, который реально нужен)

```julia
# 0) Кейс и "истина" (как у тебя)
data = parse_data_file(data_pth)
case = setup_case_from_data_file(data)
ws, states = simulate_reservoir(case)  # "истинные" состояния, нужны цели

# 1) Целевая функция на шагах (у тебя уже есть mismatch_objective)
# function mismatch_objective(m, s, dt, step_info, forces) ... end

# 2) Подготовка набора настраиваемых параметров (что по чему хотим градиенты)
dprm_case = setup_reservoir_dict_optimization(case)

# Открыть конкретные параметры:
free_optimization_parameter!(dprm_case, "porosity")          # поячейковая φ
# free_optimization_parameter!(dprm_case, "permeability")    # isotropic k
# free_optimization_parameter!(dprm_case, "permx")           # анизотропный вариант
# free_optimization_parameters!(dprm_case)                   # открыть всё типовое

# 3) Посчитать градиенты (один forward + один adjoint)
dprm_grad = parameters_gradient_reservoir(dprm_case, mismatch_objective)

# 4) Достать нужные массивы
g_poro = dprm_grad[:model][:porosity]        # ∂J/∂φ, размер = числу ячеек
# g_perm = dprm_grad[:model][:permeability]  # ∂J/∂k (или :permx/:permy/:permz)
```

# Что делает каждая функция и зачем

* `setup_case_from_data_file(data)` — собирает численную модель (сетка, PVT, скважины, расписание).
* `simulate_reservoir(case)` — прямой прогон; даёт «истинные» `states` по времени, которые твоя цель сравнивает с текущим расчётом.
* `mismatch_objective(m, s, dt, step_info, forces)` — **локальная цель на шаге**.
  Входы:
  `m` — модель шага, `s` — состояние в конце шага, `dt` — длительность шага,
  `step_info[:time]` — время начала шага, `forces` — нагрузки/скважинные воздействия.
  Выход: вклад в $J$ за этот шаг (скаляр).
* `setup_reservoir_dict_optimization(case)` — формирует описание «настраиваемых» параметров, по которым можно брать градиенты/оптимизировать.
* `free_optimization_parameter!(dprm_case, name; kwargs...)` — «разморозить» параметр `name` (дать ему свободу). Тут ты решаешь **по чему именно** нужен градиент (φ, k, WI, и т.д.). Доп.параметры: пределы, масштабирование, lumping.
* `parameters_gradient_reservoir(dprm_case, mismatch_objective)` — запускает **адъюнкт**: пробегает модель, вызывает твою цель на каждом шаге, строит $\partial J/\partial \text{state}$, решает обратную задачу и возвращает словарь градиентов по всем «размороженным» параметрам.
* (Опционально) `plot_cell_data(m, dprm_grad[:model][:porosity])` — отрисовка карты чувствительности.

# Где лежат результаты (структура `dprm_grad`)

* Уровень 1: блоки `:model`, `:wells`, `:controls`.
* Уровень 2: имена параметров. Примеры:

  * `dprm_grad[:model][:porosity] :: Vector{Float64}` — поячейковая $\partial J/\partial \phi$.
  * `dprm_grad[:model][:permeability]` или `:permx/:permy/:permz` — $\partial J/\partial k$.
  * `dprm_grad[:wells][:well_index]` и пр. — по скважинным параметрам (если их «разморозил»).

# Важные нюансы, чтобы градиенты были корректными

* **Единицы и нормализация** в цели: ты делишь на $(100\ \text{бар})^2$ и умножаешь на $dt/T$. Это делает цель безразмерной и аккумулирует вклад по времени → устойчивые величины градиента.
* **Сопоставление по времени**: сейчас берёшь «ближайший» шаг `states[step]`. Если нужно более гладко — переходи на линейную интерполяцию по времени.
* **Выбор параметров**: открывай только то, что действительно нужно; иначе размерность огромная, и сигнал градиента размывается.
* **Lumping**: если хочешь градиент **по слоям/фациям**, открой поячейковый параметр и затем агрегируй суммой по группам (или задай `lumping=` при «разморозке»).

# Быстрый чек на своих данных

```julia
println(keys(dprm_grad))                  # => [:model, :wells, :controls] (зависит от кейса)
println(keys(dprm_grad[:model]))          # какие именно параметры посчитаны
println(size(dprm_grad[:model][:porosity]))
println(maximum(abs, dprm_grad[:model][:porosity]))
```

Если нужно — покажу, как за 5 строк добавить интерполяцию по времени в `mismatch_objective` и как вывести топ-N ячеек по |∂J/∂φ| с их (i,j,k).
