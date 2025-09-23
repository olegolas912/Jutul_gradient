using DelimitedFiles
using GLMakie

function read_timing_csv(path::AbstractString)
    data, _ = readdlm(path, ',', header=true)
    step = Int64.(data[:, 1])
    forward_mean = Float64.(data[:, 3])
    backward_mean = Float64.(data[:, 4])
    total_mean = Float64.(data[:, 5])
    return step, forward_mean, backward_mean, total_mean
end

function plot_timing_steps(path::AbstractString; save_path::Union{Nothing,String}=nothing)
    step, fwd, bwd, total = read_timing_csv(path)
    fig = Figure(size = (900, 320))
    ax = Axis(fig[1, 1], title = "Adjoint timing by step", xlabel = "report step", ylabel = "seconds")
    lines!(ax, step, fwd, label = "forward mean")
    lines!(ax, step, bwd, label = "backward mean")
    lines!(ax, step, total, label = "total mean")
    axislegend(ax; position = :rt)
    if save_path !== nothing
        save(save_path, fig)
    end
    display(fig)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    csv_path = length(ARGS) ≥ 1 ? ARGS[1] : get(ENV, "JUTUL_TIMING_CSV", "")
    isempty(csv_path) && error("Provide CSV path via first argument or JUTUL_TIMING_CSV")
    png_path = length(ARGS) ≥ 2 ? ARGS[2] : replace(csv_path, ".csv" => ".png")
    plot_timing_steps(csv_path; save_path = png_path)
end
