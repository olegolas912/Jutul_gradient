using Jutul, JutulDarcy, GeoEnergyIO, GLMakie

data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data = parse_data_file(data_pth);
case = setup_case_from_data_file(data);

function F(prm, step_info = missing)
    data_c = deepcopy(data)
    data_c["GRID"]["PORO"] = fill(prm["poro"], size(data_c["GRID"]["PORO"]))
    case = setup_case_from_data_file(data_c)
    return case
end

x_truth = only(unique(data["GRID"]["PORO"]))
prm_truth = Dict("poro" => x_truth)
case_truth = F(prm_truth)
ws, states = simulate_reservoir(case_truth)

function pdiff(p, p0)
    v = 0.0
    for i in eachindex(p)
        v += (p[i] - p0[i])^2
    end
    return v
end

step_times = cumsum(case.dt)
total_time = step_times[end]
function mismatch_objective(m, s, dt, step_info, forces)
    t = step_info[:time] + dt
    step = findmin(x -> abs(x - t), step_times)[2]
    p = s[:Reservoir][:Pressure]
    v = pdiff(p, states[step][:Pressure])
    return (dt/total_time)*(v/(si_unit(:bar)*100)^2)
end

dprm_case = setup_reservoir_dict_optimization(case)
free_optimization_parameters!(dprm_case)
dprm_grad = parameters_gradient_reservoir(dprm_case, mismatch_objective);