import Base.push!

type BoundedQueue
    queue::Vector{Any}
    size::Int64
end

type Agent
	own_plan::Array{Float64, 1}
    opinions::Dict{Array{Float64, 1}, Float64}
    others_plan::Dict{Agent, BoundedQueue}
    own_util::Float64
    noise_amp::Float64
    num_memory::Int
    search_radius::Float64
    temp::Float64
    player_id::Int
    bounds::NTuple{Array{Float64, 1}, Array{Float64, 1}}
    dimension::Int
    max_evals::Int
end

Agent(noise, num_memory, search_radius, player_num, bounds, dimension) = Agent(Float64[], Dict{Array{Float64, 1}, Float64}(), Dict{Agent, BoundedQueue}(), 0., noise, num_memory, search_radius, 0., player_num, bounds, dimension, 100)

function push!(queue::BoundedQueue, item)
    push!(queue.queue, item)
    if length(queue.queue) >= queue.size
        shift!(queue.queue)
    end
end

function init!(self::Agent, num_opinions, global_util)
        # Randomly initialize own plan
        self.own_plan = random_plan(self.dimension, self.bounds)
        sample!(self, global_util, num_opinions, self.noise_amp, self.dimension)
        self.own_util = get_utility(self, self.own_plan)
    end

function random_plan(dimensions, bounds)
    """ Generate a random plan within the given bounds of
    the specified number of dimensions.
    """
    return (bounds[2] - bounds[1]) .* rand(dimensions) + bounds[1]
end


function get_utility(self::Agent, plan::Array{Float64})
    """ Return the internal utility of a plan.
        """
        # Build list of all known plans (all v_i,j,k)
        plans = collect(self.opinions)
        utility = 0.
        #if self.own_util != None:
        #   plans += [(tuple(self.own_plan), self.own_util)]
        for other_plan in self.others_plan
            agent, plan_mem = other_plan
            plans = vcat(plans, plan_mem.queue)
        end


        plan_list_tmp, utils_tmp = zip(plans...)
        plan_list = collect(plan_list_tmp)
        utils = collect(utils_tmp)

        #println(utils)
        #println(plan_list)
        #println(plan)

        #print plan
        #util_ = np.sum(utils[np.unique(np.where(plan_list == plan)[0])])
        #matches = np.where(plan_list == plan)[0]
        #println("Evaluating ",plan," against ",plan_list)
        matches = convert(Array{Bool}, map(x -> x == plan, plan_list))


        # If the plan is identical with one we know, weight goes to infinity,
        # so return sum of utils of identical plans?
        #assert(not matches.any())
        if any(matches)
            #print "Matched", matches
            return mean(utils[matches])
        end

        weights = Float64[get_weight(x, plan) for x in plan_list]
        #results = np.array(map(lambda x: self.get_weight(x[0], plan), plans))
        #denom = np.sum(results)
        sum((weights ./ sum(weights)) .* utils)
end

function p_accept(agent::Agent, plan)
    """ Returns the probability of accepting a
    plan based on current temp && distance from
    own plan.
    """
    c = plan .- agent.own_plan
    norm = dot(c, c)
    exp(-norm / agent.temp)
end

function get_weight(plan_a, plan_b)
    """
    Get the weight of plan_b as the normalised
    inverse square distance from plan_a.
    """
    c = plan_a .- plan_b
    norm = sqrt(dot(c, c))
    if norm == 0
        return 0
    end
    norm^-2.
end


function set_other_players!(self::Agent, players)
        """ Add all the other players to mental model.
        """

        for player in players
            if !is(player, self)
                self.others_plan[player] = BoundedQueue(Any[], self.num_memory)
            end
        end
    end

function sample!(self::Agent, utility_fn, num_opinions, noise, dimensions)
        """ Sample some number of opinions from the true utility with noise.
        """
        while length(self.opinions) < num_opinions
            plan = rand(dimensions)
            fuzz = (noise + noise) * rand() - noise
            self.opinions[plan] = utility_fn(plan) + fuzz
        end
    end

    function consider_plan(self::Agent, agent::Agent, opinion, working_plan)
        """ Consider a proposed change to the group plan
        && either incorporate it, or incorporate && support
        it.
        """
        push!(self.others_plan[agent], opinion)
        self.own_util = get_utility(self, self.own_plan)
        incorp = incorporate(self, opinion, working_plan)
        back = false
        if !incorp
            back = support(self, opinion)
        end
        # Remember the opinion
        #prself, "considered plan by", agent, incorp, support, "Response to plan", opinion
        back || incorp
    end

    function where(index::BitArray, x::Array, y::Array)
        """
        Return an array with values chosen from x or y if the entry
        in index is true or false.
        """
        map(i -> index[i] ? x[i] : y[i], 1:length(index))
    end

    function incorporate(self::Agent, opinion, working_plan)
        """ Choose whether to incorporate a new opinion into
        this agent's plan.
        """
        #propinion, "opinion"
        plan, utility = opinion
        index = plan .!= working_plan
        diff = where(index, plan, self.own_plan)
        #prplan,"differs from", working_plan,"at",index,"ours is",diff
        expect_util = get_utility(self, diff)
        #prself.player_id,"considering",opinion,expect_util,"vs",self.own_plan,",",self.own_util
        if expect_util > self.own_util
            self.own_plan = diff
            self.own_util = expect_util
            return true
        end
        return false
    end

    function support(self::Agent, opinion)
        """ Return true to vote the opinion into the global
        plan.
        """
        #propinion, "opinion"
        plan, utility = opinion
        rand() < p_accept(self, plan)
    end

    function inverse_util(self::Agent, plan)
        """ Negative of utility function for
        minimising.
        """
        return -get_utility(self, plan)
    end

    function constraint_1(self::Agent, x)
        """ Constrakeeps solutions within self.radius
        of current plan.
        """
        if self.search_radius > get_distance(self, x)
            return true
        end
        return false
    end

    function constraint_2(self::Agent, x)
        """ Constrain answers within max & min bound.
        """
        if all(self.bounds[2] .>= x) && all(self.bounds[1] .<= x)
            return true
        end
        return false
    end

    function update!(self::Agent)
        """ Search for a new, more optimal plan within search
        radius.
        
        function f(plan)
            inverse_util(self, plan)
        end

        function radius(x)
            constraint_1(self, x)
        end

        function bound(x)
            constraint_2(self, x)
        end

        #min_bounds = [max(self.own_plan[x] - self.search_radius,self.bounds[0][x]) for x in xrange(self.dimension)]
        #max_bounds = [min(self.own_plan[x] + self.search_radius,self.bounds[1][x]) for x in xrange(self.dimension)]
        #search_bounds = zip(min_bounds, max_bounds)
        new_plan = opt.fmin_cobyla(f, self.own_plan, [radius, bound], disp=1)
        self.own_util = get_utility(self, new_plan)
        #prsearch_bounds
        #prself.own_plan, "New Plan", new_plan
        self.own_plan = new_plan
        """
        self.own_plan, self.own_util = hillclimb(self)
    end

    function hillclimb(self::Agent)
        plan = self.own_plan
        util = self.own_util
        mutation_prob = 1 / float(self.dimension)
        step_size = self.search_radius / float(self.max_evals)

        function mutate(x)
            rand() > mutation_prob ? x : ((rand() *2*step_size) - step_size) + x
        end

        for i in 1:self.max_evals
            candidate = map(mutate, plan)
            if constraint_2(self, candidate) && constraint_1(self, candidate)
                candidate_util = get_utility(self, candidate)
                if candidate_util > util
                    plan = candidate
                    util = candidate_util
                end
            end
        end
        plan, util
    end



    function choose_opinion(self::Agent, working_plan)
        """ Make a suggestion to modify an aspect
        of the group plan.
        """
        # Work out possible new plans
        possible_plans = Dict{Array{Float64}, Float64}()
        util_working = get_utility(self, working_plan)
        for i in 1:length(working_plan)
            tmp_plan = working_plan
            tmp_plan[i] = self.own_plan[i]
            #prtmp_plan, "Tmp plan"
            possible_plans[tmp_plan] =  get_utility(self, tmp_plan) - util_working
        end

        return weighted_choice(self, possible_plans)
    end


    function get_distance(self::Agent, plan)
        """ Get the distance between this plan && our
        internal plan.
        """

        a = self.own_plan
        b = plan
        c = a - b
        return sqrt(dot(c, c))
    end

    function diff_util(self::Agent, utility_fn)
        """ Returns a utility function that is the difference
        between own && the one given.
        """
        function f(plan)
            return get_utility(self, plan) - utility_fn(plan)
        end
        return f
    end

    function abs_diff_util(self::Agent, utility_fn)
        """ Returns a utility function that is the absolutw
        difference between own && one given.
        """
        function f(plan)
            return abs(get_utility(self, plan) - utility_fn(plan))
        end
        return f
    end

    function weighted_choice(self::Agent, plans)
        # Normalize util gains
        plan_list, util_list = zip(collect(plans)...)
        utils = collect(util_list)
        plan_list = collect(plan_list)
        max_util = max(utils)
        min_util = min(utils)
        diff = max_util - min_util

        if diff == 0
            utils = ones(length(utils)) ./ float(length(utils))
        else
            utils = (utils - min_util) ./ diff
        end
        total = sum(utils)
        threshold = rand()*total
        bracket = 0.
        for i in 1:length(plans)
            if bracket + utils[i] > threshold
                return (plan_list[i], get_utility(self, plan_list[i]))
            end
            bracket += utils[i]
        end
        return (plan_list[i], get_utility(self, plan_list[i]))
    end
