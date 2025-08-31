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

### Где «начинает работать» оптимизатор в коде

В `rate_optimization.jl` внутри `optimize_rates(steps; use_box_bfgs=true)`:

```julia
setup = JutulDarcy.setup_rate_optimization_objective(coarse_case, base_rate;
    max_rate_factor = 10,
    oil_price = 100.0,
    water_price = -10.0,
    water_cost = 5.0,
    discount_rate = 0.05,
    maximize = use_box_bfgs,
    sim_arg = (rtol = 1e-5, tol_cnv = 1e-5),
    steps = steps
)

if use_box_bfgs
    obj_best, x_best, hist = Jutul.unit_box_bfgs(setup.x0, setup.obj;
        maximize = true, lin_eq = setup.lin_eq)
    H = hist.val
else
    lower = zeros(length(setup.x0)); upper = ones(length(setup.x0))
    results, x_best = lbfgsb(setup.F!, setup.dF!, setup.x0;
        lb = lower, ub = upper, iprint = 1, factr = 1e12, maxfun = 20, maxiter = 20)
    H = results
end
return (setup.case, H, x_best)
```

**Отсюда видно:** оптимизатор стартует после того, как `setup_rate_optimization_objective(...)` вернула интерфейс `setup`:
- `setup.x0` — старт `x` (доли $\alpha$);  
- `setup.obj` / `setup.F!` / `setup.dF!` / `setup.F_and_dF!` — функции цели и её градиента;  
- `setup.lin_eq` — линейные равенства $A x = b$.

Дальше вызывается либо **`Jutul.unit_box_bfgs`** (Box‑BFGS с равенствами), либо **`LBFGSB.lbfgsb`** (классический L‑BFGS‑B по бокс‑ограничениям).

---

### Что подаётся на вход оптимизатору

- **Вектор переменных** $x$:  
  - `steps = :first` → длина $= n_{\text{inj}}$.  
  - `steps = :each` → длина $= n_{\text{inj}}\times n_{\text{step}}$.  
  Компоненты $x$ — это доли $\alpha \in [0,1]$.

- **Бокс‑ограничения:** $0 \le x \le 1$.  
  В `lbfgsb` это `lb=zeros(...)`, `ub=ones(...)`. Для `unit_box_bfgs` — встроено.

- **Линейные равенства:** $A x = b$.  
  В `:first` — $A\in\mathbb{R}^{1\times n_{\text{inj}}}$ (одна строка из единиц), $b=[n_{\text{inj}}/ \texttt{max\_rate\_factor}]$.  
  В `:each` — $A\in\mathbb{R}^{n_{\text{step}}\times (n_{\text{inj}} n_{\text{step}})}$ (по строке на шаг).

- **Целевая функция** $F(x)$ = **NPV** (с дисконтом), и её **градиент** $\nabla F(x)$.  
  Оптимизатор вызывает `setup.F!` и `setup.dF!` (или `setup.F_and_dF!`), а те внутри:
  1) расставляют таргеты скважин по $x$;
  2) прогоняют симуляцию;
  3) считают $F$ и $\nabla F$ через adjoint (см. §6).

- **Флаг maximize/minimize:** здесь `maximize=true` (ищем максимум NPV).  

---

### Что выдаёт оптимизатор

- **`x_best`** — оптимальные доли $\alpha^\star$ (вектор той же длины, что и `x0`).  
- **`obj_best`** — лучшее значение цели $F(x^\star)$ (NPV).  
- **`hist`/`results`** — история итераций (значения цели, нормы проекционного градиента, статусы line search). В твоём логе, например:
  ```
  It.| Objective  | Proj. grad | Linesearch-its
   0 | 2.1311e+08 | 6.7642e+08 | -
   1 | 2.1378e+08 | 2.1397e+07 | 1
   ...
  ```

---

### Что делает BFGS «под капотом» (вкратце и без воды)

Ищем максимум гладкой цели на допустимом множестве:  
$
\max_{x\in[0,1]^n,\ A x=b} F(x).
$

Итерация $k$:

1) **Градиент** $g_k = \nabla F(x_k)$.  
   Для равенств/бокса строится **проекционный градиент** $g^{\text{proj}}_k$ (обнуляются компоненты по активным границам и ортогонализуются по $A x=b$).

2) **Направление** $p_k = - H_k\, g^{\text{proj}}_k$, где $H_k$ — приближение к $(\nabla^2 F)^{-1}$.  
   Вначале $H_0 = I$.

3) **Поиск шага** (line search): выбирается $t_k>0$ (иногда с бэктрекингом / условиями Вольфе).  
   Кандидат: $\tilde x_{k+1} = x_k + t_k\, p_k$.

4) **Проекция в допустимую область:**  
   - **бокс:** обрезка $[0,1]$ по компонентам,  
   - **равенства:** проектирование на гиперплоскость $\{x: A x=b\}$.

   Получаем $x_{k+1} = \Pi(\tilde x_{k+1})$.

5) **Обновление кривизны (BFGS):**  
   $s_k = x_{k+1}-x_k,\quad y_k = g_{k+1}-g_k,\quad \rho_k = 1/(y_k^\top s_k)$.  
$
H_{k+1} = (I - \rho_k s_k y_k^\top)\, H_k\, (I - \rho_k y_k s_k^\top) + \rho_k\, s_k s_k^\top.
$

**L‑BFGS‑B** хранит не полный $H_k$, а последние $m$ пар $(s,y)$ (ограниченная память) + работает с активным множеством по бокс‑ограничениям. Равенства обрабатываются проекцией/штрафом (в твоём логе у `lbfgsb` были предупреждения именно про «constraint handling»).

---

### Мини‑пример (3 инжектора, 1 шаг, равенство $\sum \alpha = 0.3$)

- Старт: $x_0 = (0.1,\,0.1,\,0.1)$ (всё внутри [0,1], сумма = 0.3).  
  $H_0 = I$.

- Пусть градиент (для **максимизации**): $g_0 = \nabla F(x_0) = (-1.59,\ 0.40,\ -0.80)\times 10^6$.  
  (интерпретация: выгодно увеличить 1‑ю и 3‑ю доли, уменьшить 2‑ю — знак «–» у 1‑й и 3‑й означает, что **двигаясь в сторону + по этим координатам** мы увеличим цель, так как шаг $p_k=-H_k g_k$).

- Направление: $p_0 = - H_0 g_0 = (1.59,\ -0.40,\ 0.80)\times 10^6$.

- **Проекция на гиперплоскость $\mathbf{1}^\top p = 0$** (чтобы равенство $\sum \alpha = 0.3$ сохранялось на шаге):  
  Вычтем среднее: $\bar p = \frac{1}{3}\sum p = 0.6633\times 10^6$.  
  $p^{\text{proj}} = p - \bar p \cdot (1,1,1) = (0.9267,\ -1.0633,\ 0.1367)\times 10^6$.  
  (сумма компонент = 0).

- Нормируем шаг по line search: выберем маленький $t=10^{-7}$ (чтобы остаться в [0,1]).  
  $\tilde x_1 = x_0 + t p^{\text{proj}} \approx (0.1927,\ -0.0063,\ 0.1137)$.

- **Бокс‑проекция** на $[0,1]$: вторая компонента станет 0:  
  $x_1'=(0.1927,\,0,\,0.1137)$.

- **Коррекция равенства** $\sum \alpha = 0.3$: сейчас сумма $=0.3064$.  
  Проецируем на $\sum \alpha = 0.3$ (равномерно сдвигом вдоль $(1,1,1)$):  
  $\delta = (0.3064-0.3)/3 \approx 0.0021$.  
  $x_1 = x_1' - \delta(1,1,1) \approx (0.1906,\, -0.0021,\, 0.1116)$ → снова бокс‑проекция $\to (0.1906,\,0,\,0.1094)$ и финальная равенство‑коррекция по двум свободным координатам: $(0.1906,\,0,\,0.1094)$ уже суммируется в 0.3000.

- Симуляция → новый градиент $g_1$ → обновление $H_1$ по формуле BFGS → итерация повторяется.

> В реальном коде эти проекции/коррекции выполняются внутри `unit_box_bfgs`/`lbfgsb`; мы показали **смысл**: шаг строится с учётом **бокса** и **равенства**.
---


## 6) Где **лежат градиенты** (напоминалка, с цифрами и ссылкой на код)

### Коротко (3 “уровня” градиента)
1) **По целевым дебитам (target rates)** — *сырые чувствительности adjoint*:  
   `dforces[step][:Facility].control[well].target.value`  
   Это величина $ \displaystyle \frac{\partial \mathrm{NPV}}{\partial q^{\text{target}}_{well,\,step}} $ — как меняется NPV, если слегка увеличить целевой дебит **этой** скважины **на этом** шаге.

2) **По управлениям $\alpha$** — *матрица `df` (инжектор × шаг)*:  
   $
   \frac{\partial \mathrm{NPV}}{\partial \alpha_{j,i}} \;=\;
   \underbrace{\frac{\partial \mathrm{NPV}}{\partial q^{\text{target}}_{j,i}}}_{\text{из }dforces}\times
   \underbrace{\frac{\partial q^{\text{target}}_{j,i}}{\partial \alpha_{j,i}}}_{=\,Q_{\max}\ \text{или }0\text{ на пороге}}
   \quad\Rightarrow\quad
   \texttt{df[inj\_no, stepno]} = \texttt{do\_dq} \times \texttt{max\_rate}.
   $

3) **Вектор для оптимизатора** — *`vec(df)` → копируется в `dFdx`*:  
   оптимизатор ожидает плоский вектор той же длины, что и `x`, поэтому `df` расплющивается столбцами → записывается в предоставленный буфер `dFdx` внутри функций-обёрток `F_and_dF!`/`dF!`.

### Где в коде (файл и функции)
- **Файл:** `objectives.jl`  
- **Функции / места:**
  - внутри `setup_rate_optimization_objective(...)` определена внутренняя функция `f!(x; grad=true)`, где:
    1) запускается adjoint:  
       `dforces, t_to_f, grad_adj = Jutul.solve_adjoint_forces!(...)`;
    2) читается чувствительность **по таргетам**:  
       `do_dq = dforces[stepno][:Facility].control[inj].target.value`;
    3) собирается **матрица `df`**:  
       `df[inj_no, stepno] = do_dq * max_rate`;
    4) формируется **вектор градиента**: `grad = vec(df)` и возвращается вместе с целью.
  - обёртки, которые видит оптимизатор (там же, в `objectives.jl`):  
    ```julia
    function F_and_dF!(dFdx, x)
        obj, grad = f!(x, grad = true)
        dFdx .= grad             # ← сюда попадает vec(df)
        return obj
    end
    function dF!(dFdx, x)
        _, grad = f!(x, grad = true)
        dFdx .= grad
        return dFdx
    end
    ```

### Числовой мини‑пример (твой кейс, единицы «м³/сут»)
Из лога:  
- `base_rate ≈ 79.5 м³/сут` на скважину, `max_rate_factor = 10` ⇒
  $Q_{\max} = 10 \times 79.5 = \mathbf{795\ \text{м}^3/\text{сут}}$.
- На шаге $i$ adjoint дал по трём инжекторам (пример):  
  $(\partial \mathrm{NPV}/\partial q^{\text{target}}_{1,i},\ \partial \mathrm{NPV}/\partial q^{\text{target}}_{2,i},\ \partial \mathrm{NPV}/\partial q^{\text{target}}_{3,i})
   =(2000,\ -500,\ 1000)\ \,[\$ / (\text{м}^3/\text{сут})]$.

Тогда столбец градиента **по долям** на этом шаге:
$
df[:, i] =
\begin{bmatrix}
2000\\-500\\1000
\end{bmatrix} \times 795 \;=\;
\begin{bmatrix}
1.59\cdot 10^6\\ -3.98\cdot 10^5\\ 7.95\cdot 10^5
\end{bmatrix}\ [\$].
$

Если шагов два и для второго шага получили, скажем,  
$(2100,\ -100,\ 500)$, то
$
df =
\begin{bmatrix}
1.59\cdot 10^6 & 1.67\cdot 10^6\\
-3.98\cdot 10^5 & -7.95\cdot 10^4\\
7.95\cdot 10^5 & 3.98\cdot 10^5
\end{bmatrix},
\qquad
\text{а } \mathrm{vec}(df) =
\begin{bmatrix}
1.59\cdot 10^6\\
-3.98\cdot 10^5\\
7.95\cdot 10^5\\
1.67\cdot 10^6\\
-7.95\cdot 10^4\\
3.98\cdot 10^5
\end{bmatrix}.
$

Именно этот вектор `vec(df)` и копируется в `dFdx` внутри `F_and_dF!`/`dF!` — это **тот самый** градиент, который использует оптимизатор.

> ⚠️ Если на какой‑то скважине сейчас действует минимум `MIN_INITIAL_WELL_RATE`, то $q^{\text{target}}=\max(\alpha Q_{\max}, Q_{\min})$ «упёрся» в порог и $\partial q^{\text{target}}/\partial \alpha = 0$, поэтому соответствующая компонента `df` будет нулевой независимо от `do_dq`.

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