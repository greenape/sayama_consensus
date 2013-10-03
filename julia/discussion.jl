require("agent")
using PyCall
pyinitialize("/System/Library/Frameworks/Python.framework/Versions/2.7/python")
@pyimport scipy.optimize as opt

type Discussion
	players::Array{Agent, 1}
    theta::Float64
    search_radius::Float64
    frequencies::Array{Float64}
    bounds::NTuple{Array{Float64, 1}, Array{Float64, 1}}
    working_plan::Array{Float64}
    dimension::Int
    num_frequencies::Int
    num_players::Int
    max_it::Int
    alpha::Float64
    consensus_threshold::Float64
    trajectories::Dict{Int, Array{Any, 1}}
    distances::Dict{Int, Array{Any, 1}}
    max_sum::Float64
    min_sum::Float64
    noise::Float64
    num_memory::Int
    num_opinions::Int
    track_convergence::Bool
    convergence::Array{Float64}
    true_plan_utility::Float64
    current_it::Int
end

dimension = 2
num_players = 3
num_memory = 0
num_opinions = 20
num_frequencies = 5
max_it = 100
alpha = 0.05
noise = 0.2
search_radius = 0.005
consensus_threshold = 0.04

#Discussion(None, 0., search_radius, None, None, None, dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, None, None, 0., 0., noise, num_memory, num_opinions, false, 0.)

Discussion() = Discussion(Agent[], 0., search_radius, Float64[], (Float64[], Float64[]), Float64[], dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, Dict{Int, Array{Any, 1}}(), Dict{Int, Array{Any, 1}}(), 0., 0., noise, num_memory, num_opinions, true, Float64[], 0., 0)
Discussion(search_radius, dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, noise, num_memory, num_opinions) = Discussion(Agent[], 0., search_radius, Float64[], (Float64[], Float64[]), Float64[], dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, Dict{Int, Array{Any, 1}}(), Dict{Int, Array{Any, 1}}(), 0., 0., noise, num_memory, num_opinions, true, Float64[], 0., 0)
Discussion(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, recording, frequencies) = Discussion(Agent[], 0., search_radius, Float64[], (Float64[], Float64[]), Float64[], dimension, num_frequencies, num_players, max_it, alpha, consensus_threshold, Dict{Int, Array{Any, 1}}(), Dict{Int, Array{Any, 1}}(), 0., 0., noise, num_memory, num_opinions, false, Float64[], 0., 0)


function init!{Discussion}(self::Discussion, frequencies)
    self.players = Agent[]
    self.bounds = (zeros(self.dimension), ones(self.dimension))
    self.working_plan = random_plan(self.dimension, self.bounds)
    if is(frequencies, None)
        generate_frequencies!(self)
        self.max_sum = find_max_s(self)
        self.min_sum = find_min_s(self)
    else
        self.frequencies = frequencies
    end
    self.trajectories = [-1 => Any[]]
    self.distances = [-1 => Any[]]
    
    init_players!(self)
    return self
    end

function init_players!(discussion::Discussion)
        discussion.players = Agent[]
        # Make players
        for i in 1:discussion.num_players
            player = Agent(discussion.noise, discussion.num_memory, discussion.search_radius, i, discussion.bounds, discussion.dimension)
            function true_util(plan)
                true_utility(discussion, plan)
            end
            init!(player, discussion.num_opinions, true_util)
            push!(discussion.players, player)
        end
        # Inform of fellows
        for player in discussion.players
            set_other_players!(player, discussion.players)
            discussion.trajectories[player.player_id] = Any[]
            discussion.distances[player.player_id] = Any[]
        end
    end

function set_bounds!(discussion::Discussion, bounds)
        discussion.bounds = bounds
        #print bounds
        discussion.working_plan = random_plan(discussion.dimension, bounds)
        for player in discussion.players
            player.bounds = bounds
            player.own_plan = random_plan(discussion.dimension, bounds)
        end
    end

function generate_frequencies!(discussion::Discussion)
        """ Populate frequencies.
        """
        discussion.frequencies = [rand()*50. for x=1:discussion.dimension, y=1:discussion.num_frequencies]
        # print discussion.frequencies
    end

function true_utility(discussion::Discussion, plan)
        """ Return the true utility of a plan.
        """
        summation = s_eq(discussion, plan)
        utility = summation - discussion.min_sum
        utility /= discussion.max_sum - discussion.min_sum
        if utility > 1 || utility < 0
            println("Utility out of bounds: $utility, summation = $summation")
        end
        utility
    end

function find_max_s(discussion::Discussion)
        """ Use an initial brute force search followed by scipy
        optimisation to find the maximum of the s function.
        """
        function f(plan)
            -s_eq(discussion, plan)
        end
        search_bounds = collect(zip(zeros(discussion.dimension), ones(discussion.dimension)))
        grid = ntuple(discussion.dimension, x -> (0, 1, discussion.search_radius))
        minimised = opt.minimize(f, opt.brute(f, grid), bounds=search_bounds, method="L-BFGS-B", tol=1e-16, options={"disp" => false})
        #ops = @options display=None fcountmax=1000 tol=1e-6
        #x, fval, fcount, converged = fminbox(f, [0., 0.], search_bounds[1], search_bounds[2], ops)
        -minimised["fun"]
    end

function find_min_s(discussion::Discussion)
        """ Use an initial brute force search followed by scipy
        optimisation to find the minimum of the s function.
        """
        search_bounds = collect(zip(zeros(discussion.dimension), ones(discussion.dimension)))
        grid = ntuple(discussion.dimension, x -> (0, 1, discussion.search_radius))
        function f(plan)
            return s_eq(discussion, plan)
        end
        minimised = opt.minimize(f, opt.brute(f, grid), bounds=search_bounds, method="L-BFGS-B", tol=1e-16, options={"disp" => false})
        #return minimised['fun']
        #ops = @options display=None fcountmax=1000 tol=1e-6
        #x, fval, fcount, converged = fminbox(f, [0., 0.], search_bounds[1], search_bounds[2], ops)
        minimised["fun"]
    end

function s_eq(discussion::Discussion,  plan)
        """ Compute the sum of sinusoids at some set
        of points.
        """
        sum(sin(discussion.frequencies .* plan))
    end

function print_s_eq(discussion::Discussion)
        """ Return the equation that gives the s(v) for the utility
        function.
        """
        eq = "s(v)="
        for i in 1:discussion.dimension
            for j in 1:discussion.num_frequencies
                eq += "sin(%fx_%d) + " % (discussion.frequencies[i][j], i)
            end
        end
        eq
    end

function choose_speaker(discussion::Discussion, players::Array{Agent})
        """ Pick somebody at random to make a suggestion.
        """
        players[rand(1:length(players))]
    end

function update_plan!(discussion::Discussion, opinion, proposer, players::Array{Agent})
        """ Incorporate an opinion into the global plan
        if enough yes votes are taken.
        """
        if motion_carried(discussion, vote(discussion, proposer, opinion, players), players)
            discussion.working_plan, util = opinion
            println("Carried $opinion")
            #print "Actual util", discussion.true_utility(discussion.working_plan)
        end
    end

function motion_carried(discussion::Discussion, yes_votes, players::Array{Agent})
        """ Return true if this plan has enough votes
        to replace the group plan.
        """
        result = 1. + yes_votes
        result /= length(players)
        result > discussion.theta
    end

function vote(discussion::Discussion, proposer, opinion, players::Array{Agent})
        """ Take a vote on a proposed change.
        """
        yes_votes = 0
        for player in players
            if !is(player, proposer)
                if consider_plan(player, proposer, opinion, discussion.working_plan)
                    yes_votes += 1
                end
            end
        end
        #print "%d votes for." % yes_votes
        yes_votes
    end

function update_temperature!(discussion::Discussion, players::Array{Agent})
        """ Update cognition temperature of agents.
        """
        temp = discussion.alpha * (discussion.max_it / discussion.current_it)
        for agent in players
            agent.temp = temp
        end
    end

function do_turn!(discussion::Discussion, players::Array{Agent})
        """ Iterate the discussion.
        """
        update_temperature!(discussion, players)
        update_plans!(players)
        discussion.theta = (discussion.current_it / discussion.max_it)
        speaker = choose_speaker(discussion, players)
        opinion = choose_opinion(speaker, discussion.working_plan)
        #print speaker, "proposed", opinion
        update_plan!(discussion, opinion, speaker, players)
    end

function update_plans!(players::Array{Agent})
        """ Have all the agents search for a better plan.
        """
        for agent in players
            update!(agent)
        end
    end

function consensus_reached(discussion::Discussion, players::Array{Agent})
        """ Return true if the sum of distances between individual
        plans and the group plan is less than threshold.
        """
        distance_sum = 0.
        for player in players
            distance_sum += get_distance(player, discussion.working_plan)
        end
        println("Distance sum is $distance_sum")
        distance_sum < discussion.consensus_threshold
    end

function get_distance(discussion::Discussion, players::Array{Agent})
        """ Return the sum of distances from the working plan.
        """
        distance_sum = 0.
        for player in players
            distance_sum += get_distance(player, discussion.working_plan)
        end
        distance_sum
    end

function do_discussion!(discussion::Discussion, players::Array{Agent})
        """ Run a discussion.
        """
        
        discussion.working_plan = random_plan(discussion.dimension, discussion.bounds)
        if is(players, None)
            players = discussion.players
        end
        for player in players
            player.own_plan = random_plan(discussion.dimension, discussion.bounds)
            player.own_util = get_utility(player, player.own_plan)
        end
        for i in 1:discussion.max_it
            discussion.current_it = i
            store_trajectories!(discussion, players)
            store_plan_distance!(discussion, players)
            if discussion.track_convergence
                store_convergence!(discussion, players)
            end
            do_turn!(discussion, players)
            if consensus_reached(discussion, players)
                store_trajectories!(discussion, players)
                store_plan_distance!(discussion, players)
                discussion.true_plan_utility = true_utility(discussion, discussion.working_plan)
                return None
            end
            println("Played turn $i")
        end
        discussion.true_plan_utility = true_utility(discussion, discussion.working_plan)
    end

function store_trajectories!(discussion::Discussion, players::Array{Agent})
        """ Store the group plan, and each agent's plan.
        """
        push!(discussion.trajectories[-1], discussion.working_plan)
        for player in players
            push!(discussion.trajectories[player.player_id], player.own_plan)
        end
    end

function store_plan_distance!(discussion::Discussion, players::Array{Agent})
        """ Record the sum of distance to the working plan, and the distances of
        individual agents.
        """
        push!(discussion.distances[-1], get_distance(discussion, players))
        for player in players
            push!(discussion.distances[player.player_id], get_distance(player, discussion.working_plan))
        end
    end

function store_convergence!(discussion::Discussion, players::Array{Agent})
        push!(discussion.convergence, pairwise_convergence(discussion, 1000, players))
    end

function t_diff(discussion::Discussion, n::Int, players)
        util_a, util_b, pair_avg, pair_sum, diff_sum
        
        if !is(players, None)
            players = discussion.players
        end
        pairs = Set(combinations(players, 2))
        plans = random.random((n, discussion.dimension))
        vals = dict(zip(players, [zeros(n)]*length(players)))
        t_sum = 0.
        for player in players
            for i in 1:n
                vals[player][i] = get_utility(player, plans[i])
            end
        end
        for pair in pairs
            a, b = pair
            diff = abs(vals[a] - vals[b])
            t = sum(diff)
            t /= sqrt((n*sum(diff^2) - sum(diff)^2) / (n - 1))
            println(string(a, b, t))
            t_sum += t
        end
        t_sum /= length(pairs)
        println("T_avg = $t_sum")
    end


function pairwise_convergence(discussion::Discussion, n::Int, players::Array{Agent})
        """ Compute the average percentage difference in utility functions
        of all agents across n random points in the problem space.
        """        
        if is(players, None)
            players = discussion.players
        end
        pairs = collect(combinations(players, 2))
        diff_sum = 0.
        plans = [random_plan(discussion.dimension, discussion.bounds) for x=1:n]
        vals = Dict{Int, Array{Float64}}()
        for player in players
            vals[player.player_id] = map(x -> get_utility(player, x), plans)
        end
        for pair in pairs
            a, b = pair
            diff = abs(vals[a.player_id] - vals[b.player_id])
            pair_avg = vals[a.player_id] + vals[b.player_id]
            pair_avg /= 2.
            diff_sum += mean(diff ./ pair_avg)
            #diff_sum += mean(diff)
        end
        return diff_sum / length(pairs)
    end