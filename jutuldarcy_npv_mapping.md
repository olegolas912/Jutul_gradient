# JutulDarcy NPV Optimization — Math ↔ Code Mapping (with Numbers)

**Files referenced:** `rate_optimization.jl`, `objectives.jl`, `utils.jl`  
**Goal:** показать, куда именно «заходят числа» (дебиты, объёмы закачки), и дать прямую аналогию «математика = переменная/функция в коде», с числовым примером.

---

## 0) Чёткие исходные данные для численного примера

- Кол-во инжекторов: $n_{\text{inj}} = 8$.
- Базовый таргет на одну скважину: `base_rate = 79.5` м³/сут.
- Умножитель потолка: `max_rate_factor = 10`  $\Rightarrow Q_{\max} = 79.5 \cdot 10 = 795$ м³/сут.
- Минимальный дебит одной скважины: `MIN_INITIAL_WELL_RATE = 1` м³/сут.
- 50 отчётных шага одинаковой длины: $\Delta t_1 = \Delta t_2 = ?$ суток. 
- Дисконт: `discount_rate = 0.05` (5%/год), дисконт-единица — *год*.
- Цены/затраты: `oil_price = 100`, `water_price = -10`, `water_cost = 5` (условные \$/м³).

**Замечание по единицам:** внутри Jutul/JutulDarcy всё в строгих SI; в примере для прозрачности используем м³/сут и дни, но в коде есть перевод через `liquid_unit`, `gas_unit`, `discount_unit` и т.п.

---

## 1) Управления (u ≡ α) → уставки инжекторов (таргеты дебита)

### Математика

- Управления (фракции закачки по инжекторам и шагам):
  $
  \alpha_{j,i} \in [0,1].
  $

- Верхний предел на одну скважину:
  $
  Q_{\max} = \texttt{max\_rate\_factor} \cdot \texttt{base\_rate}.
  $

- Таргет дебита инжектора $j$ на шаге $i$:
  $
  q^{\text{inj}}_{j,i}(\alpha) = \max\!\big(\alpha_{j,i} \, Q_{\max},\; Q_{\min}\big), \quad Q_{\min}=\texttt{MIN\_INITIAL\_WELL\_RATE}.
  $

- (Часто) Линейное равенство по шагу (фиксируем суммарную закачку):
  $
  \sum_{j=1}^{n_{\text{inj}}} \alpha_{j,i} \;=\; \frac{n_{\text{inj}} \cdot \texttt{base\_rate}}{Q_{\max}}
  \;=\; \frac{n_{\text{inj}}}{\texttt{max\_rate\_factor}}.
  $

### Код (ключевые места)

- В `objectives.jl` (внутри `setup_rate_optimization_objective(...)`):
  ```julia
  max_rate = max_rate_factor*base_rate        # = Q_max
  x_i = x[inj_no, stepno]                     # = α_{j,i}
  new_rate = max(x_i*max_rate, MIN_INITIAL_WELL_RATE)  # = q^{inj}_{j,i}
  new_target = TotalRateTarget(new_rate)
  f_forces = case.forces[stepno][:Facility]
  ctrl = f_forces.control[inj]
  f_forces.control[inj] = replace_target(ctrl, new_target)
  f_forces.limits[inj]  = merge(lims, as_limit(new_target))
  ```

- Начальный вектор управления:
  ```julia
  x0 = fill(base_rate/max_rate, ninj, nsteps) # α^0 = 1/max_rate_factor
  ```

- Линейное равенство:
  ```julia
  lin_eq = (A = A, b = b) # Σ_j α_{j,i} = ninj*base_rate/max_rate = ninj/max_rate_factor
  ```

### Числа (шаг 1 и шаг 2)

- Шаг 1: возьмём $\alpha_{:,1} = [0.10, 0.15, 0.05]$.  
  Проверка ограничения: $0.10 + 0.15 + 0.05 = 0.30 = \frac{3}{10}$.  
  Таргеты:
  $
  (q^{\text{inj}}_{1,1}, q^{\text{inj}}_{2,1}, q^{\text{inj}}_{3,1})
  = (100,\,150,\,50)\ \text{м}^3/\text{сут}.
  $

- Шаг 2: $\alpha_{:,2} = [0.05, 0.20, 0.05]$, сумма $=0.30$.  
  Таргеты: $(50, 200, 50)$ м³/сут.

- Матрица ограничения для 2 шагов и 3 инжекторов:
  $
  A=\begin{bmatrix}
  1&1&1&0&0&0\\
  0&0&0&1&1&1
  \end{bmatrix},\quad
  b=\begin{bmatrix}0.3\\0.3\end{bmatrix}.
  $

---

## 2) Прямой прогон (forward) и извлечение дебитов продюсеров/инжекторов

### Математика

- Динамика состояний:
  $
  S_{i+1} = \Phi\!\big(S_i,\; q^{\text{inj}}_{:,i}(\alpha),\; \text{прочие силы}\big).
  $

- Фактические поверхностные дебиты по продюсерам на шаге $i$:
  $
  q_o(i;\alpha),\quad q_w(i;\alpha),\quad q_g(i;\alpha).
  $
  По инжекторам — фактическая закачка (суммарно): $q_{\text{inj}}(i;\alpha)$.

### Код

- Запуск симуляции:
  ```julia
  simulated = simulate_reservoir(case; output_substates=use_ministeps, info_level=-1, sim_arg...)
  r = simulated.result
  ```

- Извлечение QoI для скважины `w`:
  ```julia
  orat = compute_well_qoi(model, state, forces, w, SurfaceOilRateTarget)   / liquid_unit
  wrat = compute_well_qoi(model, state, forces, w, SurfaceWaterRateTarget) / liquid_unit
  grat = compute_well_qoi(model, state, forces, w, SurfaceGasRateTarget)   / gas_unit
  ```

### Числа (суммарно по всем продюсерам)

- Шаг 1: $q_o = 500$ м³/сут, $q_w = 800$ м³/сут; сумма закачки: $q_{\text{inj}} = 100+150+50 = 300$ м³/сут.
- Шаг 2: $q_o = 520$ м³/сут, $q_w = 780$ м³/сут; $q_{\text{inj}} = 300$ м³/сут.

---

## 3) Целевая функция NPV (с дисконтированием)

### Переменные (и где в коде)

- $p_o$ — `oil_price = 100` (\$/м³),  
  $p_w$ — `water_price = -10` (\$/м³),  
  $c_w$ — `water_cost = 5` (\$/м³).
- $\Delta t_i$ — длительность шага: `dt` (в сутках/сек — зависит от настроек).
- $t_i$ — календарное время конца шага в годах:  
  `time = step_info[:time] + dt` → `discount = (1.0 + discount_rate)^(-time/discount_unit)`.
- $r$ — `discount_rate = 0.05`.
- Масштаб `scale` (для единиц; в примере считаем равным 1).
- Знак `maximize` → `sgn = +1` (максимизация прибыли).

### Математика (вклад одного шага)

$J_i(\alpha)=\Big(p_o \underbrace{\sum_{w\in\mathcal P} q_o^{(w)}(i;\alpha)}_{\text{нефть}}+ p_g \underbrace{\sum_{w\in\mathcal P} q_g^{(w)}(i;\alpha)}_{\text{газ}}+ p_w \underbrace{\sum_{w\in\mathcal P} q_w^{(w)}(i;\alpha)}_{\text{вода на выкиде}}\;-\;c_w \underbrace{\sum_{w\in\mathcal I} q_{\text{inj}}^{(w)}(i;\alpha)}_{\text{закачка воды}}
\Big)\cdot \Delta t_i \cdot (1+r)^{-t_i}.
$

Итоговая цель: $\mathrm{NPV}(\alpha) = \sum_i J_i(\alpha)$.

### Код

- Суммирование по продюсерам (доходы/штрафы):
  ```julia
  obj += oil_price*orat + gas_price*grat + water_price*wrat
  ```

- Затраты по инжекторам:
  ```julia
  obj -= (oil_cost*orat + gas_cost*grat + water_cost*wrat)
  ```

- Дисконт и вклад шага:
  ```julia
  return sgn * dt * obj * ((1.0 + discount_rate)^(-time/discount_unit)) / scale
  ```

### Числовой пример NPV (2 шага по 30 суток)

- $\Delta t_1=\Delta t_2=30$ сут $\Rightarrow$ в годах: $30/365 \approx 0.08219$.
- Время конца шагов: $t_1 \approx 0.08219$ г., $t_2 \approx 0.16438$ г.
- Дисконт: $D_1 = (1.05)^{-0.08219} \approx 0.9960$, $D_2 \approx 0.9920$.

Шаг 1:
$
\begin{aligned}
C_1 &= 100\cdot 500 \;+\; (-10)\cdot 800 \;-\; 5\cdot 300 \;=\; 40500\ \$/\text{сут},\\
J_1 &= 40500 \times 30 \times 0.9960 \;\approx\; \mathbf{1\,210\,137.42}.
\end{aligned}
$

Шаг 2:
$
\begin{aligned}
C_2 &= 100\cdot 520 \;+\; (-10)\cdot 780 \;-\; 5\cdot 300 \;=\; 42700\ \$/\text{сут},\\
J_2 &= 42700 \times 30 \times 0.9920 \;\approx\; \mathbf{1\,270\,767.08}.
\end{aligned}
$

Сумма: $\mathrm{NPV} \approx \mathbf{2\,480\,904.51}$.

---

## 4) Градиент $\partial \mathrm{NPV}/\partial \alpha$ (adjoint)

### Математика (цепочка)

$
\frac{\partial \mathrm{NPV}}{\partial \alpha_{j,i}}
= \underbrace{\frac{\partial \mathrm{NPV}}{\partial q^{\text{inj}}_{j,i}}}_{\text{adjoint даёт}}
\cdot
\underbrace{\frac{\partial q^{\text{inj}}_{j,i}}{\partial \alpha_{j,i}}}_{=\,Q_{\max}\;\text{(или 0 на пороге)}}.
$

### Код

- Подготовка и запуск сопряжённой задачи по силам:
  ```julia
  cache[:storage] = Jutul.setup_adjoint_forces_storage(case.model, r.states, forces, case.dt, npv_obj; ...)
  dforces, t_to_f, grad_adj = Jutul.solve_adjoint_forces!(cache[:storage], case.model, r.states, r.reports, npv_obj, forces; ...)
  ```

- Извлечение чувствительности по таргету инжектора и преобразование к $\partial/\partial \alpha$:
  ```julia
  ctrl = dforces[stepno][:Facility].control[inj]
  do_dq = ctrl.target.value              # ≈ ∂NPV/∂(target_rate)
  df[inj_no, stepno] = do_dq * max_rate  # ∂NPV/∂α = (∂NPV/∂q)*Q_max
  ```

### Числовой пример

Пусть adjoint дал на шаге 1:  
$(\partial \mathrm{NPV}/\partial q^{\text{inj}}_{1,1}, \partial \mathrm{NPV}/\partial q^{\text{inj}}_{2,1}, \partial \mathrm{NPV}/\partial q^{\text{inj}}_{3,1}) = (2000,\; -500,\; 1000)$ $[\$/(\text{м}^3/\text{сут})]$.  
Тогда при $Q_{\max}=1000$ м³/сут:
$
\left(\tfrac{\partial \mathrm{NPV}}{\partial \alpha_{1,1}},
\tfrac{\partial \mathrm{NPV}}{\partial \alpha_{2,1}},
\tfrac{\partial \mathrm{NPV}}{\partial \alpha_{3,1}}\right)
= (2.0\!\times\!10^6,\; -5.0\!\times\!10^5,\; 1.0\!\times\!10^6).
$

Это прямо то, что делает строка `df[inj_no, stepno] = do_dq * max_rate`.

---

## 5) Оптимизация (поиск $u$)

### Математика

$
\max_{\alpha\in[0,1]^n,\; A\alpha=b} \ \mathrm{NPV}(\alpha).
$

### Код

- Конструктор цели возвращает интерфейс для оптимизатора:
  ```julia
  return (
    x0 = x0,
    lin_eq = (A=A, b=b),
    obj = f!,
    F! = F!, dF! = dF!, F_and_dF! = F_and_dF!,
    case = case
  )
  ```

- В `rate_optimization.jl` запускается, например:
  ```julia
  Jutul.unit_box_bfgs(setup.x0, setup.obj; maximize=true, lin_eq=setup.lin_eq)
  ```

- Стартовое управление: $\alpha^0 = \texttt{base\_rate}/Q_{\max} = 100/1000 = 0.1$ в каждой ячейке, а $\sum_j \alpha_{j,i} = 0.3$ удерживает суммарный таргет $= 300$ м³/сут на каждом шаге.

---

## Быстрая «шпаргалка» соответствий (с числами)

| Матем. объект | Значение (пример) | Аналог в коде |
|---|---:|---|
| $Q_{\max}$ | $1000$ м³/сут | `max_rate = max_rate_factor*base_rate` |
| $\alpha_{1,1}$ | $0.10$ | `x[inj1, step1]` |
| $q^{\text{inj}}_{1,1}$ | $\max(0.10 \cdot 1000, 1)=100$ | `new_rate = max(x_i*max_rate, MIN_INITIAL_WELL_RATE)` |
| $\sum_j \alpha_{j,1}$ | $0.30$ | `lin_eq = (A, b)` где $b_1=0.3$ |
| $q_o(1), q_w(1)$ | $500, 800$ м³/сут | `compute_well_qoi(..., SurfaceOilRateTarget/SurfaceWaterRateTarget)` |
| $q_{\text{inj}}(1)$ | $300$ м³/сут | сумма `new_rate` по инжекторам |
| $C_1$ (в сутки) | $40500$ \$ | `obj += ...; obj -= ...` |
| $J_1$ | $\approx 1\,210\,137.42$ \$ | `return sgn*dt*obj*discount/scale` |
| $J_2$ | $\approx 1\,270\,767.08$ \$ | то же для шага 2 |
| $\mathrm{NPV}$ | $\approx 2\,480\,904.51$ \$ | сумма по шагам |
| $\partial \mathrm{NPV}/\partial q^{\text{inj}}_{1,1}$ | $2000$ | `do_dq = dforces[step][:Facility].control[inj].target.value` |
| $\partial \mathrm{NPV}/\partial \alpha_{1,1}$ | $2000 \cdot 1000 = 2\cdot10^6$ | `df[inj_no, stepno] = do_dq * max_rate` |

---

## Мини-FAQ

**Где именно «заходят числа» дебитов?**  
В `npv_objective`: дебиты продюсеров $q_o, q_w, q_g$ входят в доход/штраф, дебиты инжекторов $q_{\text{inj}}$ — в затраты. Всё это извлекается через `compute_well_qoi(...)` и суммируется в `obj` за шаг, затем умножается на `dt` и дисконтируется.

**Как $\alpha$ превращается в дебит?**  
Через $q^{\text{inj}}_{j,i}=\max(\alpha_{j,i} Q_{\max}, Q_{\min})$ и замену цели скважины: `replace_target(ctrl, TotalRateTarget(new_rate))` на нужном шаге в `case.forces`.

**Откуда берётся градиент по $\alpha$?**  
Adjoint возвращает $\partial \mathrm{NPV}/\partial(\text{target\_rate})$, далее умножаем на $Q_{\max}$ — получаем $\partial \mathrm{NPV}/\partial \alpha$.

---