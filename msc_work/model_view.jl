using Jutul
using JutulDarcy
using GLMakie


case = setup_case_from_data_file(joinpath("/home/oleg/Github/Jutul_gradient", "SPE1.DATA"))
ws, states = simulate_reservoir(case, output_substates = true)
plot_reservoir(case.model, states)