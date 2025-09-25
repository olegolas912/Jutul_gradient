using JutulDarcy, GLMakie
@time file = "/home/oleg/Upscaled_10m_cells/Upscaled_10m_cells.data"
@time case = setup_case_from_data_file(file; skip_wells = true, skip_forces = true)
@time fig = plot_reservoir(case.model)
@time display(fig)