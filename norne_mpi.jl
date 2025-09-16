#!/usr/bin/env julia
using JutulDarcy
using GeoEnergyIO
using Jutul                    # нужно для Jutul.MPI_PArrayBackend
using MPI
using PartitionedArrays        # активирует расширение под PArrays
using HYPRE
using Printf

# ---------------------------
# helpers for env-based tuning
# ---------------------------
env_symbol(name::String, default::Symbol) = haskey(ENV, name) ? Symbol(ENV[name]) : default
function env_bool(name::String, default::Bool)
    val = get(ENV, name, nothing)
    isnothing(val) && return default
    lower = lowercase(val)
    return lower in ("1", "true", "t", "yes", "y", "on")
end
function env_float(name::String, default::Union{Nothing, Float64})
    val = get(ENV, name, nothing)
    if isnothing(val) || isempty(val)
        return default
    end
    try
        return parse(Float64, val)
    catch
        return default
    end
end
function env_int(name::String, default::Union{Nothing, Int})
    val = get(ENV, name, nothing)
    if isnothing(val) || isempty(val)
        return default
    end
    try
        return parse(Int, val)
    catch
        return default
    end
end

const PRECOND = env_symbol("JUTUL_PRECOND", :cprw)
const LINSOLVER = env_symbol("JUTUL_SOLVER", :bicgstab)
const CPR_UPDATE = env_symbol("JUTUL_CPR_UPDATE", :iteration)
const CPR_UPDATE_PARTIAL = haskey(ENV, "JUTUL_CPR_UPDATE_PARTIAL") ? env_symbol("JUTUL_CPR_UPDATE_PARTIAL", :iteration) : (CPR_UPDATE == :once ? :iteration : CPR_UPDATE)
const CPR_SMOOTHER = env_symbol("JUTUL_CPR_SMOOTHER", :ilu0)
const CPR_PARTIAL = env_bool("JUTUL_CPR_PARTIAL", CPR_UPDATE in (:once, :step))
const LS_RTL = env_float("JUTUL_LSOLVER_RTOL", nothing)
const LS_MAXIT = env_int("JUTUL_LSOLVER_MAXIT", nothing)

# --- MPI init ---
MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const SIZE = MPI.Comm_size(COMM)

# --- utilities ---
elapsed_str(t)::String = t < 1 ? @sprintf("%.3f s", t) :
                         t < 120 ? @sprintf("%.2f s", t) : @sprintf("%.1f min", t/60)
maxall(x) = (y = Ref{Float64}(0.0); MPI.Allreduce!(Ref(x), y, MPI.MAX, COMM); y[])

# --- optional HYPRE init (safe if already initialized) ---
try
    HYPRE.Init(finalize_atexit = false)
catch err
    RANK == 0 && @warn "HYPRE.Init failed" err
end

# --- Stage 1: dataset path ---
t0 = time()
egg_dir = GeoEnergyIO.test_input_file_path("EGG")
egg_pth = joinpath(egg_dir, "EGG.DATA")
t_path = maxall(time() - t0)

# --- Stage 2: build case ---
t1 = time()
case = setup_case_from_data_file(
    egg_pth;
    backend = :csr,
    split_wells = true,
    verbose = (RANK == 0),
)
t_setup = maxall(time() - t1)

# --- Stage 3: MPI simulation on explicit PArray backend ---
par_backend = Jutul.MPI_PArrayBackend(COMM)

ls_args = Dict{Symbol, Any}(
    :update_interval => CPR_UPDATE,
    :update_interval_partial => CPR_UPDATE_PARTIAL,
    :partial_update => CPR_PARTIAL,
    :smoother_type => CPR_SMOOTHER,
)
if !isnothing(LS_MAXIT)
    ls_args[:max_iterations] = LS_MAXIT
end

t2 = time()
ws, states = JutulDarcy.simulate_reservoir_parray(
    case, par_backend;
    info_level = 1,
    output_path = "out_mpi",
    precond = PRECOND,
    linear_solver = LINSOLVER,
    linear_solver_arg = ls_args,
    rtol = LS_RTL,
)
t_sim = maxall(time() - t2)

# --- Report (rank 0) ---
if RANK == 0
    println("\n=== MPI REPORT ===")
    println("Ranks: ", SIZE, " | Threads per rank: ", Threads.nthreads())
    println("Linear solver: ", LINSOLVER, ", precond: ", PRECOND)
    println("CPR update: ", CPR_UPDATE, ", smoother: ", CPR_SMOOTHER, ", partial: ", CPR_PARTIAL)
    if !isnothing(LS_RTL)
        println("Linear rtol override: ", LS_RTL)
    end
    if !isnothing(LS_MAXIT)
        println("Linear max it override: ", LS_MAXIT)
    end
    println("Path resolve (max):   ", elapsed_str(t_path))
    println("Case setup (max):     ", elapsed_str(t_setup))
    println("Simulation (max):     ", elapsed_str(t_sim))
    println("Total wall-time (max):", elapsed_str(t_path + t_setup + t_sim))
    println("Report steps:         ", length(states))
end

# --- clean up before MPI finalize ---
ws = nothing
states = nothing
case = nothing
GC.gc()
GC.gc()

MPI.Barrier(COMM)
MPI.Finalize()
