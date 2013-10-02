require("discussion")

function landscape(resolution, dimensions, utility_fn)
	aspects = linspace(0, 1, int(1/resolution))
	points = combinations(aspects, dimensions)
	return pmap(x -> (x, utility_fn(x)), points)
end

"""d = Discussion()
init!(d, None)
do_discussion!(d, d.players)

d2 = Discussion()
d2.num_memory = 5
init!(d2, None)
do_discussion!(d2, d2.players)

d3 = Discussion()
d3.num_memory = 30
init!(d3, None)
do_discussion!(d3, d3.players)"""

function dump_experiment(file_name, results)
	file = open(file_name, "w")
    @printf(file, "%s\n", join(results["fields"], ","))

    for result in results["results"]
        @printf(file, "%s\n", join(result, ","))
    end
    close(file)
end

function run_discussion(discussion)
	run, discussion = discussion
    do_discussion!(discussion, discussion.players)
    return [discussion.num_memory, run, discussion.true_plan_utility,pairwise_convergence(discussion, 1000, discussion.players), discussion.dimension]
end

function run_q_convergence(discussion)
	run, discussion = discussion
    do_discussion!(discussion, discussion.players)

    return [discussion.num_memory, run, (discussion.current_it) / float(discussion.max_it)]
end

function q_convergence(;runs::Int=100, players::Int=3, num_landscapes::Int=100)
	""" Run an experiment recording time to convergence of individual
    plans to the group plan for q 0 ~ 10.
    Returns a results dictionary.
    """
    num_players = players
    max_it = 200.
    consensus_threshold = 0.04
    results = ["fields"=> ["q", "run", "time"], "results" => []]
    # q 0 - 10
    count = 1
    discussions = Any[]
    landscapes = Discussion[]
    for i in 1:num_landscapes
    	d = Discussion(search_radius, dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, noise, num_memory, num_opinions)
        push!(landscapes, d)
    end
    #println("Made ",num_landscapes," landscapes.")
    landscapes = map(x -> init!(x, None), landscapes)
    #println("Solved them.")
    for num_memory in 0:10
        # 100 runs of each
        for i in 1:runs
            for l in 1:num_landscapes
                #println("Making run ",count," of ",runs*num_landscapes*11,"..")
                count += 1
                d = deepcopy(landscapes[l])
                d.num_memory = num_memory
                init_players!(d)
                push!(discussions, (i, d))
            end
        end
    end
    results["results"] = pmap(run_q_convergence, discussions)
    #println("Ran ",count," discussions.")
    #print results["results"]
    return results
end

function convergence_fixed_paired(;runs::Int=100, players::Int=3, num_landscapes::Int=100, max_it::Float64=100,
	step_size::Int=1, max_mem::Int=50)
    """ Run some number of replications of the three discussion types and return a results dictionary.
    """
    num_players = players
    consensus_threshold = -1  # No consensus
    results = ["fields" => ["q", "run", "utility","fidelity","convergence", "protocol","dimensions","landscape"], "results" => []]
    # q 0 - 10
    count = 1
    discussions = Any[]
    landscapes = Discussion[]
    for i in 1:landscapes
        d = Discussion(dimension, num_players, 0, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
        push!(landscapes, d)
    end
    landscapes = pmap(x -> init!(x, None), landscapes)
    for num_memory in 0:50
        # 100 runs of each
        for i in 1:runs
            for l in landscapes
                #println("Making run ",count," of ",runs*num_landscapes*51,"..")
                count += 1
                d = deepcopy(l)
                d.num_memory = num_memory
                init_players!(d)
                push!(discussions, (i, d))
            end
        end
    end

    results["results"] = pmap(run_discussion, discussions)
    #print results["results"]
    return results
end

r = q_convergence(num_landscapes=1)
dump_experiment("juliatest.csv", r)

r = @time(convergence_fixed_paired(num_landscapes=1))
dump_experiment("convergence_short.csv", r)

r = @time(convergence_fixed_paired(num_landscapes=1, max_it=150))
dump_experiment("convergence_long.csv", r)

r = @time(convergence_fixed_paired(num_landscapes=1, max_it=300, step_size=5, max_mem=100))
dump_experiment("convergence_very_long.csv", r)