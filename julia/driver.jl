require("sayama_rep")

r = @time(q_convergence(num_landscapes=1))
dump_experiment("juliatest.csv", r)

"""r = @time(convergence_fixed_paired(num_landscapes=1))
dump_experiment("convergence_short.csv", r)

r = @time(convergence_fixed_paired(num_landscapes=1, max_it=150))
dump_experiment("convergence_long.csv", r)

r = @time(convergence_fixed_paired(num_landscapes=1, max_it=300, step_size=5, max_mem=100))
dump_experiment("convergence_very_long.csv", r)"""