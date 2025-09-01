# 0) Точка входа: кто вообще решает, какие α ставить

**Файл:** `rate_optimization.jl`
**Функция:** `optimize_rates(steps; use_box_bfgs=true)`

* Тут создаётся пакет интерфейсов:

  ```julia
  setup = setup_rate_optimization_objective(coarse_case, base_rate; ... , steps=steps)
  ```

  В `setup` лежат:

  * `x0` — начальный вектор долей α,
  * `obj` / `F!` / `dF!` / `F_and_dF!` — обёртки цели/градиента,
  * `lin_eq` — линейные равенства Ax=b (сумма α по шагу).
* **Затем оптимизатор** (он и “решает”, какие α лучше):

  ```julia
  obj_best, x_best, hist = Jutul.unit_box_bfgs(setup.x0, setup.obj;
      maximize=true, lin_eq=setup.lin_eq)
  # или lbfgsb(setup.F!, setup.dF!, setup.x0; lb=0, ub=1, ...)
  ```

  Он будет многократно вызывать `setup.obj` (или `F!`/`dF!`) с разными `x`, сравнивать NPV и двигать `x` (α).

---

# 1) Где программа «понимает», **что поставить на инжекторах** (α → целевые дебиты)

**Файл:** `objectives.jl`
**Функция:** `setup_rate_optimization_objective(...)` → внутренняя **`f!(x; grad=true)`**

1. `x` приводится к матрице долей:

   ```julia
   x_mat = reshape(x, ninj, nstep_unique)   # α[j,i]
   ```
2. Для **каждого шага i** и **каждого инжектора j** строится уставка:

   ```julia
   max_rate = max_rate_factor*base_rate     # Qmax per well
   new_rate   = max(x_mat[j,i]*max_rate, MIN_INITIAL_WELL_RATE)
   new_target = TotalRateTarget(new_rate)
   f_forces   = case.forces[i][:Facility]
   ctrl       = f_forces.control[injectors[j]]
   f_forces.control[injectors[j]] = replace_target(ctrl, new_target)
   f_forces.limits[injectors[j]]  = merge(lims, as_limit(new_target))
   ```

   👉 **Здесь** и происходит «поставить α»: доля умножается на `Qmax`, и целевой дебит инжектора на шаге заменяется на `new_target`.

**Число (твой кейс):** `base_rate≈0.0009201389 м³/с` (=79.5 м³/сут), `max_rate_factor=10` →
`Qmax≈0.0092013889 м³/с` (=795 м³/сут). Если `α=0.12`, то `new_rate=0.12*Qmax=0.00110417 м³/с` (=**95.4 м³/сут**).

---

# 2) Где запускается **прямой прогон** (forward model)

**Там же, внутри `f!`:**

```julia
sim  = simulate_reservoir(case; output_substates=use_ministeps, info_level=-1, sim_arg...)
r    = sim.result
```

* На выходе `r.states`/`r.reports`: давления, насыщенности, фактические дебиты по скважинам/полю (`:fwir`, `:fopr`, и т.п.).
* По твоему логу: `:fwir ≈ 0.00736111 м³/с` = **636 м³/сут** (что равно `0.8 * Qmax`).

---

# 3) Где считается и **сравнивается** цель (NPV)

**Внутри `f!`** формируется **объект цели** (замыкание) через `npv_objective(...)`, а потом:

```julia
obj = Jutul.evaluate_objective(npv_obj, case.model, r.states, case.dt, case.forces)
```

* Внутри `npv_objective(...)` (тот же файл) на каждом шаге берутся QoI скважин:

  ```julia
  orat = compute_well_qoi(..., SurfaceOilRateTarget)   / liquid_unit
  wrat = compute_well_qoi(..., SurfaceWaterRateTarget) / liquid_unit
  grat = compute_well_qoi(..., SurfaceGasRateTarget)   / gas_unit
  ```

  и собирается денежный поток за шаг с дисконтом:

  ```julia
  cash = ( + oil_price*orat + gas_price*grat + water_price*wrat  # продюсеры
           - (oil_cost*orat + gas_cost*grat + water_cost*wrat) ) # инжекторы
  J_i  = sgn * dt * cash * (1+discount_rate)^(-time/discount_unit) / scale
  ```

  Сумма по шагам = **NPV**.

👉 **“Сравнение” вариантов** делает **оптимизатор** (BFGS/L-BFGS-B): он вызывает `f!` с разными `x`, получает **число NPV**, сравнивает и двигает `x` дальше. В логе это колонка `Objective`.

---

# 4) Где запускается **обратный прогон** (adjoint / reverse) и где лежит градиент

**Снова внутри `f!`** (если `grad=true`):

1. Готовим хранилище и решаем сопряжённую задачу «по силам»:

   ```julia
   storage = Jutul.setup_adjoint_forces_storage(case.model, r.states, case.forces, case.dt, npv_obj; ...)
   dforces, t_to_f, grad_adj =
       Jutul.solve_adjoint_forces!(storage, case.model, r.states, r.reports, npv_obj, case.forces; ...)
   ```

   * **Где “сырые” чувствительности:**
     `dforces[step][:Facility].control[well].target.value` = ∂NPV/∂(target\_rate).

2. Переводим в градиент по долям α:

   ```julia
   df = zeros(ninj, nstep_unique)
   for step in 1:nstep_unique, (j, inj) in enumerate(injectors)
       do_dq      = dforces[step][:Facility].control[inj].target.value  # ∂NPV/∂q_target
       df[j,step] = do_dq * max_rate                                    # ∂NPV/∂α = (∂NPV/∂q)*Qmax
   end
   grad = vec(df)   # вектор для оптимизатора
   ```

   * **Где «живут» градиенты:**
     • по таргет-дебитам — в `dforces[...]` (см. выше),
     • по долям α — в `df[j, step]`,
     • плоский вектор для оптимизатора — `vec(df)`.

**Число:** при `Qmax=795 м³/сут` и `do_dq=2000 $/(м³/сут)` → `∂NPV/∂α=2000*795=**1.59e6 $**`.

---

# 5) Где **оптимизатор** использует это и «решает» новые α

**Файл:** `rate_optimization.jl`
**Функция:** `optimize_rates(...)`

* Запуск:

  ```julia
  obj_best, x_best, hist = Jutul.unit_box_bfgs(setup.x0, setup.obj;
      maximize=true, lin_eq=setup.lin_eq)
  ```

  или

  ```julia
  results, x_best = lbfgsb(setup.F!, setup.dF!, setup.x0; lb=0, ub=1, ...)
  ```
* Что **получает** оптимизатор на каждом шаге:

  * **значение цели** `F(x)` = NPV,
  * **градиент** `∇F(x)` = `vec(df)`,
  * **ограничения:** бокс `0≤x≤1`, равенства `A x = b`.
* Что **делает**:

  * строит направление из текущего градиента (BFGS/L-BFGS-B),
  * подбирает шаг (line search),
  * проектирует на допустимую область (`[0,1]` и `A x=b`),
  * **“сравнивает”** `Objective` старого/нового `x` (это то, что ты видишь в таблице),
  * повторяет до остановки.
* Что **возвращает**: `x_best` (лучшая найденная α), `obj_best` (лучший NPV), и `hist` (таблица итераций).

---

# 6) Где формируются **обёртки** `setup.obj` / `setup.F!` / `setup.dF!` / `setup.F_and_dF!`

**Файл:** `objectives.jl` → `setup_rate_optimization_objective(...)`

* Внутри создаётся **`f!(x; grad)`** (описан выше) и на его основе — обёртки:

  * `setup.obj` — интерфейс «значение + градиент» для `unit_box_bfgs`
    (вызовет `f!(x; grad=true)` и отдаст оба).
  * `setup.F!` — «только значение» (для L-BFGS-B).
  * `setup.dF!` — «только градиент» (копирует `vec(df)` в буфер оптимизатора).
  * `setup.F_and_dF!` — «оба сразу» (если оптимизатор умеет).

---

# 7) Куда “вплетаются” **равенства** и **границы 0..1**

* **Равенства `A x = b`:** собраны в `setup.lin_eq` (там же, в `objectives.jl`)
  и переданы в `Jutul.unit_box_bfgs(..., lin_eq=setup.lin_eq)`.
  В `:each`: `A` — по строке на шаг, `b[i]=ninj/max_rate_factor` (=**0.8** у тебя).
* **Бокс `0..1`:** встроен в `unit_box_bfgs`; в `lbfgsb` задаётся как `lb=zeros(...)`, `ub=ones(...)`.

---

## «Один цикл» в цифрах (сверка с твоим логом)

* **Старт:** `x0 .= 0.1` (8 инжекторов × 50 шагов), на каждом шаге сумма α = **0.8**.
  Подставили → forward → `:fwir ≈ 0.00736111 м³/с` (**636 м³/сут**) — это и видишь в отчётах.
  Посчитали NPV: `Objective ≈ 2.1311e+08`.
* **Adjoint:** получили `dforces` → `df = do_dq * Qmax` → `vec(df)`.
* **BFGS-шаг:** оптимизатор двигает `x` (сохраняя `Ax=b` и `0≤x≤1`), снова вызывает `f!`.
  В логе: NPV растёт (`2.1342e+08` → … → **`2.1931e+08`**), печатается `Proj. grad`, `Linesearch-its`.
* **Финиш:** `x_best` (400 чисел) — оптимальные доли; если их поставить и прогнать, получишь финальные `:fopr`, `:fwir`, `:fwpr` и т.п.

---

если хочешь, дам короткий “диагностический” патч (несколько `@info`) **ровно в те места**:

* перед `simulate_reservoir` — печатать сумму α по шагам и ожидаемый `Q_field`;
* после `evaluate_objective` — печатать текущее `Objective`;
* сразу после `solve_adjoint_forces!` — печатать пару `do_dq` и соответствующий `df[j,i]`;
* в `optimize_rates` — печатать размерности `x`, `A`, норму нарушения `‖Ax−b‖`.
