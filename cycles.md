# 0) Старт скрипта (файл `rate_optimization.jl`)

1. `using Jutul, JutulDarcy, GLMakie, GeoEnergyIO, HYPRE, LBFGSB`
2. Загрузка тестового кейса месторождения из `GeoEnergyIO`:

   * `data_dir = GeoEnergyIO.test_input_file_path("EGG")`
   * `data_pth = joinpath(data_dir, "EGG.DATA")`
3. Построение кейса:

   * `fine_case = setup_case_from_data_file(data_pth)`
   * `coarse_case = coarsen_reservoir_case(fine_case, (20,20,3), method=:ijk)`
   * усечение по числу отчётных шагов (например, первые 50).
4. Базовый целевой расход для нагнетателя:
   `ctrl = coarse_case.forces[1][:Facility].control`
   `base_rate = ctrl[:INJECT1].target.value` (примерно «исходно 500 м³/сут на нагнетатель»).

---

# 1) Точка входа в оптимизацию (файл `rate_optimization.jl`)

Функция: `optimize_rates(steps; use_box_bfgs = true)`

1. Вызов сетапа цели:
   `setup = JutulDarcy.setup_rate_optimization_objective(coarse_case, base_rate; steps=steps, ...)`
   Возвращает:

   * `x0` — стартовый вектор управлений,
   * `obj` — замыкание для расчёта цели/градиента,
   * `lin_eq` — линейные равенства (если включено «фиксировать сумму закачки»),
   * `case` — кейс с той же моделью.
2. Запуск оптимизатора:

   * если `use_box_bfgs`: `Jutul.unit_box_bfgs(setup.x0, setup.obj; maximize=true, lin_eq=setup.lin_eq)`
     (внутри — **основной цикл оптимизации**, см. раздел 5),
   * иначе — L-BFGS-B (`LBFGSB.lbfgsb(...)`).
3. Возврат: `(setup.case, hist, x_best)` где `hist` — история NPV по итерациям.

---

# 2) Постановка цели и отображение x → скважинные цели (файл `objectives.jl`)

Функция: `setup_rate_optimization_objective(case, base_rate; steps, constraint, max_rate_factor, ...)`

## 2.1 Формирование списков скважин

* Определяются `injectors` (нагнетательные) и `producers` (добывающие).

## 2.2 Параметризация управления

* `max_rate = max_rate_factor * base_rate`.
* Размерность `x`:

  * `steps = :first` → `x ∈ [0,1]^{n_наг}` (один вектор на весь период),
  * `steps = :each` → `x ∈ [0,1]^{n_наг × n_шагов}` (матрица по времени).
* Преобразование **x → forces(x)**: на каждом отчётном шаге для каждого нагнетателя заменяется `TotalRateTarget`:

  $$
  q^{\text{inj}}_{i,t}(x) = \max\big(x_{i,t}\cdot \text{max\_rate},\ q_{\min}\big).
  $$

### ЦИКЛ 2.A (внутри построения `forces(x)`)

**НАЧАЛО:** `for t in 1:n_steps`
 **вложенный:** `for inj in injectors`
  назначить целевой расход нагнетателя `inj` на шаге `t` как выше
 **КОНЕЦ вложенного**
**КОНЕЦ внешнего**

## 2.3 (Опция) Линейные равенства «фиксировать суммарную закачку»

Если `constraint = :total_sum_injected`, формируется `lin_eq = (A,b)` так, что

$$
\sum_{i} x_{i,t} = b_t = n_{\text{наг}}\cdot \frac{\text{base\_rate}}{\text{max\_rate}}
\quad (\text{обычно } b_t \text{ одинаково для всех } t).
$$

## 2.4 Симуляция для заданного x

* Вызывается `simulate_reservoir(case; output_substates=use_ministeps, info_level=-1, ...)`
  (см. следующий раздел про внутренние циклы).

---

# 3) Экономическая цель (NPV) (файл `objectives.jl`)

Функция: `npv_objective(model, state, dt, step_info, forces; injectors, producers, timesteps, prices/costs/discount, ...)`

Идея: суммируем по времени дисконтированные денежные потоки:

$$
\Delta\mathrm{NPV}_t = \Delta t \cdot \Big[\underbrace{p_o\,q_o(t) + p_g\,q_g(t) + p_w\,q^{\text{prod}}_w(t)}_{\text{добыча}}
\ -\ \underbrace{c_w^{\text{inj}}\,q^{\text{inj}}_w(t)}_{\text{закачка}}\Big] \cdot (1+r)^{-t/u},
\quad \mathrm{NPV}=\sum_t \Delta\mathrm{NPV}_t.
$$

Где:

* $q_o,q_g,q^{\text{prod}}_w$ — дебиты добывающих,
* $q^{\text{inj}}_w$ — суммарная закачка по нагнетательным,
* $p_\cdot$ — цены/штрафы за продукцию, $c_\cdot$ — затраты на закачку,
* $r$ — ставка дисконта, $u$ — единица дисконта, $\Delta t$ — длительность шага.

### ЦИКЛ 3.A (подсчёт NPV по времени)

**НАЧАЛО:** `for т in все отчётные (или мини-)шаги`
 собрать дебиты по добывающим:
 - **ЦИКЛ 3.A.1** `for prod in producers` → аккумулировать $q_o,q_g,q^{\text{prod}}_w$
 собрать закачку по нагнетательным:
 - **ЦИКЛ 3.A.2** `for inj in injectors` → аккумулировать $q^{\text{inj}}_w$
 посчитать $\Delta \mathrm{NPV}_t$ и прибавить в сумму
**КОНЕЦ**

---

# 4) Что происходит внутри симулятора (вызов `simulate_reservoir`)

Это «чёрный ящик» Jutul/JutulDarcy, но последовательность стандартна:

### ЦИКЛ 4.A (по отчётным шагам времени)

**НАЧАЛО:** `for t in 1:T`
 (опционально) деление на мини-шаги для устойчивости/вывода:
 - **ЦИКЛ 4.A.1** `for k in 1:n_ministeps(t)`
  решение нелинейной НСУ/МЛУ по давлению-насыщенности на мини-интервале
 **КОНЕЦ 4.A.1**
 сохранение отчётных величин (дебиты, состояния)
**КОНЕЦ 4.A**

Результат: `states`, `reports`, `step_info`, массив длительностей `dt` (и `timesteps` для мини-шагов).

---

# 5) Градиент по управлению (аджойнт) (файл `objectives.jl`)

Внутри `setup_rate_optimization_objective(...)` после прямого прогона:

1. Готовим хранилище чувствительностей: `Jutul.setup_adjoint_forces_storage(...)`.
2. Запускаем обратный расчёт: `Jutul.solve_adjoint_forces!(...)`.

### ЦИКЛ 5.A (обратный временной проход — аджойнт)

**НАЧАЛО:** `for t = T:-1:1`
 решается присоединённая система ⇒ чувствительность $\partial \mathrm{NPV}/\partial \text{force}_{i,t}$
**КОНЕЦ**

3. Пересчёт чувствительностей к параметрам $x$ (правило цепочки):

$$
\frac{\partial \mathrm{NPV}}{\partial x_{i,t}}
= \frac{\partial \mathrm{NPV}}{\partial q^{\text{inj}}_{i,t}}
\cdot \frac{\partial q^{\text{inj}}_{i,t}}{\partial x_{i,t}}
= \left(\frac{\partial \mathrm{NPV}}{\partial q^{\text{inj}}_{i,t}}\right)\cdot \text{max\_rate},
$$

если «пол» $\max(\cdot,q_{\min})$ не активен. Если активен, производная 0.

Возвращаем из замыкания `obj(x; grad=true)` пару `(J(x), ∇J(x))`.

---

# 6) Основной цикл оптимизации (BFGS/L-BFGS-B)

Когда в `rate_optimization.jl` вызывается `unit_box_bfgs(...)` или `lbfgsb(...)`, внутри идёт итерационный процесс «до сходимости».

### ЦИКЛ 6.A (итерации оптимизатора)

**НАЧАЛО:** `while not converged and iter < max_iter`

1. Вызов цели: `J, g = obj(x; grad=true)`
   → **внутри этого шага ещё раз происходит**:
   **ЦИКЛ 2.A** (сбор `forces(x)`), **ЦИКЛ 4.A** (прямой расчёт по времени), **ЦИКЛ 3.A** (NPV), **ЦИКЛ 5.A** (аджойнт, градиент).
2. Построение квазиньютоновского направления $d_k$ (BFGS/L-BFGS).
3. **Поиск шага (line search):** $\tau = 1, \beta, \beta^2, ...$ до выполнения критериев (Armijo/Wolfe).
4. Обновление: $x \leftarrow \Pi\{x + \tau\,d_k\}$, где $\Pi$ — проекция на $[0,1]$ и (если есть) на гиперплоскость $\sum_i x_{i,t} = b_t$.
5. Логирование: добавляем `J` в `hist`.
   **КОНЕЦ**

**Стоп-критерии:** норма градиента $\|\nabla J\| \le \varepsilon$, малость шага, или достигнут `max_iter`.

---

# 7) Финал и пост-процессинг (файл `rate_optimization.jl`)

1. Получили `x_best` → формируем оптимальные `forces(x_best)` и «оптимизированный кейс».
2. Финальная симуляция: `simulate_reservoir(case_opt, ...)` → `states_opt, reports_opt`.
3. Графики:

   * история сходимости NPV (`hist`),
   * сравнение дебитов нефти/воды по времени,
   * 3D-карты разности водонасыщенности (между базовым, «константным» и «поштучным» вариантами).

---

## Мини-числа по ключевым местам (для закрепления)

* Пусть `n_наг = 3`, `n_шагов = 50`, `base_rate=500`, `max_rate_factor=10` → `max_rate=5000`.
* При `steps=:each` у `x` размер $3\times 50$. Для строки `t=7`:
  `x[:,7] = (0.12, 0.08, 0.10)` (если сумма 0.30 зафиксирована), тогда
  $q^{\text{inj}}_{:,7} = (600, 400, 500)$ м³/сут.
* На шаге `t=7` симулятор даёт по добыче, например:
  $q_o=125$, $q^{\text{prod}}_w=85$, $q^{\text{inj}}_w=1500$.
  При `oil_price=60`, `water_cost=1`, `water_price=-2`, `Δt=30` сут, `discount≈0.984`:
  $\Delta\mathrm{NPV}_7 \approx 30\cdot(60·125 - 1·1500 - 2·85)·0.984 \approx 30·(7500-1500-170)·0.984 \approx 30·5830·0.984 \approx 171{,}9\ \text{тыс.}$

---

## Быстрые ориентиры «где какой цикл»

* **ЦИКЛ 2.A** — построение `forces(x)` (по времени и по нагнетателям).
* **ЦИКЛ 4.A (+4.A.1)** — внутренняя временная схема симулятора (отчётные и мини-шаги).
* **ЦИКЛ 3.A (+3.A.1, 3.A.2)** — суммирование экономики (по времени и по скважинам).
* **ЦИКЛ 5.A** — обратный проход (аджойнт) по времени, от T к 1.
* **ЦИКЛ 6.A** — итерации оптимизатора; **в каждой итерации** полностью выполняются 2.A, 4.A, 3.A, 5.A.
