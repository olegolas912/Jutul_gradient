#!/usr/bin/env julia
using JutulDarcy
using GeoEnergyIO
using Dates
using DelimitedFiles
using Printf
using HYPRE

# helpers -------------------------------------------------------------------
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

function timestamp() DateTime(now()) end
function elapsed_str(t)::String
    t < 1 ? @sprintf("%.3f s", t) :
    t < 120 ? @sprintf("%.2f s", t) :
    @sprintf("%.1f min", t/60)
end

try
    HYPRE.Init(finalize_atexit = false)
catch err
    @warn "HYPRE.Init failed" err
end

# --- Stage 1: dataset path ---
t0 = time()
norne_dir = GeoEnergyIO.test_input_file_path("EGG")
norne_pth = joinpath(norne_dir, "EGG.DATA")
t_path = time() - t0

# --- Stage 2: case assembly ---
t1 = time()
case = setup_case_from_data_file(
    norne_pth;
    backend = :csr,
    verbose = true,
)
t_setup = time() - t1

ls_args = Dict{Symbol, Any}(
    :update_interval => CPR_UPDATE,
    :update_interval_partial => CPR_UPDATE_PARTIAL,
    :partial_update => CPR_PARTIAL,
    :smoother_type => CPR_SMOOTHER,
)
if !isnothing(LS_MAXIT)
    ls_args[:max_iterations] = LS_MAXIT
end

# --- Stage 3: threaded simulation ---
t2 = time()
ws, states = simulate_reservoir(
    case;
    info_level = 1,
    output_path = "out_threads",
    precond = PRECOND,
    linear_solver = LINSOLVER,
    linear_solver_arg = ls_args,
    rtol = LS_RTL,
)
t_sim = time() - t2

println("\n=== THREADS REPORT ===")
println("Threads: ", Threads.nthreads())
println("Linear solver: ", LINSOLVER, ", precond: ", PRECOND)
println("CPR update: ", CPR_UPDATE, ", smoother: ", CPR_SMOOTHER, ", partial: ", CPR_PARTIAL)
println("Path resolve:     ", elapsed_str(t_path))
println("Case setup:       ", elapsed_str(t_setup))
println("Simulation:       ", elapsed_str(t_sim))
println("Total wall-time:  ", elapsed_str(t_path + t_setup + t_sim))
println("Report steps:     ", length(states))

# CSV logging for comparisons
ts = Dates.format(timestamp(), dateformat"yyyy-mm-ddTHH:MM:SS")
data = ["threads" t_path t_setup t_sim (t_path+t_setup+t_sim) Threads.nthreads() ts]
mkpath("timings")
writedlm("timings/timings_threads.csv",
    ["mode" "t_path" "t_setup" "t_sim" "t_total" "nthreads" "timestamp";
     data], ',')
println("Timings saved: timings/timings_threads.csv")

# cleanup
ws = nothing
states = nothing
case = nothing
GC.gc()
GC.gc()
