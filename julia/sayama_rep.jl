require("discussion")

function landscape(resolution, dimensions, utility_fn)
	aspects = linspace(0, 1, int(1/resolution))
	points = combinations(aspects, dimensions)
	return pmap(x -> (x, utility_fn(x)), points)
end

function dump_experiment(file_name, results)
	file = open(file_name, "w")
    print(file, "$(join(results["fields"], ","))\n")
    print(file, map(x -> join(x, ","), results["results"]))
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

function q_convergence(;runs::Int=100, players::Int=3, num_landscapes::Int=1)
	""" Run an experiment recording time to convergence of individual
    plans to the group plan for q 0 ~ 10.
    Returns a results dictionary.
    """
    num_players = players
    max_it = 200
    consensus_threshold = 0.04
    results = ["fields"=> ["q", "run", "time"], "results" => []]
    # q 0 - 10
    count = 1
    discussions = Any[]
    landscapes = Discussion[]
    for i in 1:num_landscapes
    	d = Discussion(search_radius, dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, noise, num_memory, num_opinions)
        init!(d, None)
        push!(landscapes, d)
    end
    #println("Made $num_landscapes landscapes.")
    #println("Solved them.")
    for num_memory in 0:10
        # 100 runs of each
        for i in 1:runs
            for l in landscapes
                #println("Making run $count of $(runs*num_landscapes*11)..")
                count += 1
                d = deepcopy(l)
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

function convergence_fixed_paired(;runs::Int=100, players::Int=3, num_landscapes::Int=100, max_it::Int=100,
	step_size::Int=1, max_mem::Int=50)
    """ Run some number of replications over some number of landscapes and return a results dictionary.
    """
    num_players = players
    consensus_threshold = -1  # No consensus
    results = ["fields" => ["q", "run", "utility","fidelity","convergence", "protocol","dimensions"], "results" => []]
    # q 0 - 10
    count = 1
    discussions = Any[]
    landscapes = Discussion[]
    for i in 1:landscapes
        d = Discussion(dimension, num_players, 0, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
        init!(d, None)
        push!(landscapes, d)
    end
    
    for num_memory in 0:step_size:50
        # 100 runs of each
        for i in 1:runs
            for l in landscapes
                #println("Making run $count of $(runs*num_landscapes*(max_mem+1))..")
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