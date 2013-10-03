import numpy as np
import random
import collections
import scipy.optimize as opt
from itertools import combinations
from time import time
from multiprocessing import Pool

# Default settings
dimension = 2
num_players = 6
num_memory = 0
num_opinions = 20
num_frequencies = 5
max_it = 100
alpha = 0.05
noise = 0.2
search_radius = 0.005
consensus_threshold = 0.04

DTYPE = np.float


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def landscape(resolution, dimensions, utility_fn):
        """ Returns the fitness landscape constructed from some
        utility function, discretized at the specified resolution.
        """
        aspects = [np.arange(0, 1, resolution)] * dimensions
        points = cartesian(aspects)
        return [(x, utility_fn(x)) for x in points]


def dump_landscape(file_name, dimensions, landscape):
    """ Dump a landscape to a file.
    """
    file = open(file_name, 'w')
    line = "X%d," * dimensions + "utility\n"
    file.write(line % tuple(xrange(0, dimensions)))
    file.close()
    line = "%f," * dimensions + "%f\n"
    for record in landscape:
        points, utility = record
        # prline % tuple(list(points) + [utility])
        file = open(file_name, 'a')
        file.write(line % tuple(list(points) + [utility]))
        file.close()


def dump_trajectories(file_name, dimensions, trajectories):
    file = open(file_name, 'w')
    line = "X%d," * dimensions + "Agent\n"
    file.write(line % tuple(xrange(0, dimensions)))
    file.close()
    line = "%.50g," * dimensions + "%f\n"
    for key, item in trajectories.items():
        for points in item:
            file = open(file_name, 'a')
            file.write(line % tuple(list(points) + [key]))
            file.close()

def dump_experiment(file_name, results):
    file = open(file_name, 'w')
    line = ",".join(results['fields']) + "\n"
    file.write(line)
    file.close()
    for result in results['results']:
        file = open(file_name, 'a')
        line = ",".join(str(x) for x in result) + "\n"
        file.write(line)
        file.close()

def hillclimb(self):
        plan = self.own_plan
        util = self.own_util
        mutation_prob = 1 / float(self.dimension)
        step_size = self.search_radius / float(self.max_evals)

        def mutate(x):
            result = x
            if np.random.random() > mutation_prob:
                result += (np.random.random() * 2 * step_size) - step_size

            return result

        for i in xrange(self.max_evals):
            candidate = np.array(map(mutate, plan))
            if self.constraint_2(candidate) and self.constraint_1(candidate):
                candidate_util = get_utility(self, candidate)
                if candidate_util > util:
                    plan = candidate
                    util = candidate_util
        self.own_plan = plan
        self.own_util = util


def random_plan(dimensions, bounds):
    """ Generate a random plan within the given bounds of
    the specified number of dimensions.
    """
    return (bounds[1] - bounds[0]) * np.random.random(dimensions) + bounds[0]

def get_utility(self, plan):
        """ Return the internal utility of a plan.
        """
        # Build list of all known plans (all v_i,j,k)
        plans = []
        plans += self.opinions
        utility = 0.
        #if self.own_util != None:
        #   plans += [(tuple(self.own_plan), self.own_util)]
        for agent, plan_mem in self.others_plan.items():
            plans += plan_mem

        plan_list_tmp, utils_tmp = zip(*plans)
        plan_list = np.array(plan_list_tmp, dtype=DTYPE)
        utils = np.array(utils_tmp, dtype=DTYPE)

        #print plan
        #util_ = np.sum(utils[np.unique(np.where(plan_list == plan)[0])])
        #matches = np.where(plan_list == plan)[0]

        matches = (plan_list == plan).all(axis=1)


        # If the plan is identical with one we know, weight goes to infinity,
        # so return sum of utils of identical plans?
        #assert(not matches.any())
        if matches.any():
            #print "Matched", matches
            return np.average(utils[matches])

        weights = np.array([get_weight(x, np.array(plan)) for x in plan_list], dtype=DTYPE)
        #results = np.array(map(lambda x: self.get_weight(x[0], plan), plans))
        #denom = np.sum(results)
        util = np.sum(weights / np.sum(weights) * utils)

        #utility = np.sum(np.array(map(lambda x: (self.get_weight(x[0], plan) / denom) * x[1], plans)))
        #print "New util.",denom, utility, plan, results
        #assert(util == utility)
        return util

def p_accept(self, plan):
    """ Returns the probability of accepting a
    plan based on current temp and distance from
    own plan.
    """
    a = np.array(plan)
    b = np.array(self.own_plan)
    #print a, b, "plans"
    c = a - b
    norm = c.dot(c)
    return np.exp(-norm / self.temp)

def get_weight(plan_a, plan_b):
    """ Get the weight of plan_b as the normalised
    inverse square distance from plan_a.
    """

    #a = np.array(plan_a)
    #b = np.array(plan_b)
    c = plan_a - plan_b
    norm = np.sqrt(c.dot(c))
    if norm == 0:
        return 0.
    #print a, b, norm, "weight", pow(norm, -2)
    return np.power(norm, -2)

def run_discussion(args):
    protocol = args[0]
    protocol, run, num_memory, constructor,dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, recording, dimensions, frequencies, max_s, min_s = args
    discussion = constructor(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, recording, frequencies, min_s, max_s)
    
    discussion.do_discussion()
    return [num_memory, run, discussion.true_plan_utility,discussion.fidelity(1000),discussion.pairwise_convergence(1000),protocol, dimension]

def run_q_convergence(args):
    protocol = args[0]
    protocol, run, num_memory, constructor,dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, recording, dimensions, frequencies, max_s, min_s = args
    discussion = constructor(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, recording, frequencies, min_s, max_s)

    discussion.do_discussion()

    return [num_memory, run, (discussion.current_it + 1) / float(max_it)]


class Agent(object):
    """ A simple planning agent.
    
    # Number of opinions to remember about other agents
    num_memory = 0
    # Shared opinions - queue of tuples of a list of coords & an opinion
    others_plan = {}
    # Original opinions
    opinions = set([])
    # Own 'best' plan
    own_plan = None
    # Utility of own plan
    own_util = None
    # Cognition temperature
    temp = 0.
    # Search radius
    search_radius = 0.
    # Noise amplitude
    noise_amp = 0.
    """

    def __init__(self, dimensions, noise, temp, num_memory, num_opinions, search_radius, global_util,
        player_id, bounds):
        # Randomly initialize own plan
        self.own_plan = random_plan(dimensions, bounds)
        self.opinions = set([])
        self.others_plan = {}
        self.sample(global_util, num_opinions, noise, dimensions)
        self.own_util = None
        self.own_util = get_utility(self, self.own_plan)
        self.noise_amp = noise
        self.num_memory = num_memory
        self.search_radius = search_radius
        self.temp = temp
        self.player_id = player_id
        self.bounds = bounds
        self.dimension = dimensions
        self.max_evals = 100

    def set_other_players(self, players):
        """ Add all the other players to mental model.
        """

        for player in players:
            if player is not self:
                self.others_plan[player] = collections.deque([], self.num_memory)

    def sample(self, utility_fn, num_opinions, noise, dimensions):
        """ Sample some number of opinions from the true utility with noise.
        """
        while len(self.opinions) <= num_opinions:
            plan = np.random.random(dimensions)
            fuzz = (noise + noise) * np.random.rand() - noise
            self.opinions.add((tuple(plan), utility_fn(plan) + fuzz))

    def consider_plan(self, agent, opinion, working_plan):
        """ Consider a proposed change to the group plan
        and either incorporate it, or incorporate and support
        it.
        """
        self.others_plan[agent].appendleft(opinion)
        self.own_util = get_utility(self, self.own_plan)
        incorp = self.incorporate(opinion, working_plan)
        support = False
        if not incorp:
            support = self.support(opinion)
        # Remember the opinion
        #print self, "considered plan by", agent, incorp, support, "Response to plan", opinion
        return support or incorp

    def incorporate(self, opinion, working_plan):
        """ Choose whether to incorporate a new opinion into
        this agent's plan.
        """
        #print opinion, "opinion"
        plan, utility = opinion
        index = np.array(plan) != np.array(working_plan)
        diff = np.where(index, np.array(plan), self.own_plan)
        #print plan,"differs from", working_plan,"at",index,"ours is",diff
        expect_util = get_utility(self, diff)
        #print self.player_id,"considering",opinion,expect_util,"vs",self.own_plan,",",self.own_util
        if expect_util > self.own_util:
            self.own_plan = diff
            self.own_util = expect_util
            return True
        return False

    def support(self, opinion):
        """ Return true to vote the opinion into the global
        plan.
        """
        #print opinion, "opinion"
        plan, utility = opinion
        return np.random.rand() < p_accept(self, plan)

    def inverse_util(self, plan):
        """ Negative of utility function for
        minimising.
        """
        return -get_utility(self, plan)

    def constraint_1(self, x):
        """ Constrakeeps solutions within self.radius
        of current plan.
        """
        if self.search_radius > self.get_distance(x):
            return 1.
        return -1.

    def constraint_2(self, x):
        """ Constrain answers within max & min bound.
        """
        if (self.bounds[1] >= x).all() and (self.bounds[0] <= x).all():
            return 1.
        return -1.

    def update(self):
        """ Search for a new, more optimal plan within search
        radius.
        """

        #min_bounds = [max(self.own_plan[x] - self.search_radius,self.bounds[0][x]) for x in xrange(self.dimension)]
        #max_bounds = [min(self.own_plan[x] + self.search_radius,self.bounds[1][x]) for x in xrange(self.dimension)]
        #search_bounds = zip(min_bounds, max_bounds)
        #new_plan = opt.fmin_cobyla(self.inverse_util, np.array(self.own_plan), [self.constraint_1, self.constraint_2], disp=0)
        #self.own_util = get_utility(self, new_plan)
        #print search_bounds
        #print self.own_plan, "New Plan", new_plan
        #self.own_plan = new_plan
        hillclimb(self)

    def choose_opinion(self, working_plan):
        """ Make a suggestion to modify an aspect
        of the group plan.
        """
        # Work out possible new plans
        possible_plans = []
        util_working = get_utility(self, working_plan)
        
        for i in xrange(0, len(working_plan)):
            tmp_plan = list(working_plan)
            tmp_plan[i] = self.own_plan[i]
            #print tmp_plan, "Tmp plan"
            possible_plans += [(tuple(tmp_plan), get_utility(self, tmp_plan) - util_working)]

        return self.weighted_choice(possible_plans)


    def get_distance(self, plan):
        """ Get the distance between this plan and our
        internal plan.
        """

        a = np.array(self.own_plan)
        b = np.array(plan)
        c = a - b
        return np.sqrt(c.dot(c))

    def diff_util(self, utility_fn):
        """ Returns a utility function that is the difference
        between own and the one given.
        """
        def f(plan):
            return self.get_utility(plan) - utility_fn(plan)
        return f

    def abs_diff_util(self, utility_fn):
        """ Returns a utility function that is the absolutw
        difference between own and one given.
        """
        def f(plan):
            return np.abs(self.get_utility(plan) - utility_fn(plan))
        return f

    def weighted_choice(self, plans):
        # Normalize util gains
        plan_list, util_list = zip(*plans)
        utils = np.array(util_list)
        max_util = np.max(utils)
        min_util = np.min(utils)
        diff = max_util - min_util
        
        if diff == 0:
            utils = np.ones(utils.size) / float(utils.size)
        else:
            utils = (utils - min_util) / diff
        total = np.sum(utils)
        threshold = random.uniform(0, total)
        bracket = 0.
        for i in xrange(len(plans)):
            if bracket + utils[i] > threshold:
                return (plan_list[i], get_utility(self, plan_list[i]))
            bracket += utils[i]
        return (plan_list[i], get_utility(self, plan_list[i]))

    def __str__(self):
        return "Agent id: %d" % self.player_id

    def __unicode__(self):
        return self.__str__()


class Discussion(object):
    """
    # Current working aproximation, tuple of vector of choices + a utility
    working_plan = []
    # Number of participants
    num_players = 0
    # Discussion participants
    players = []
    # Incorporation threshold
    theta = 0.
    # Problem space dimension (m)
    dimension = 0
    # Number of frequencies (l)
    num_frequencies = 0
    # Frequencies lxm matrix of random values between 0 & 50
    frequencies = [[]]
    # Number of iterations to discuss for
    max_it = 0.
    # Current iteration
    current_it = 0.
    # Cognition temperature scaler
    alpha = 0.
    """

    def __init__(self, dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, 
        search_radius, consensus_threshold, track_convergence=False, frequencies=None, min_s=None, max_s=None):
        self.players = []
        self.theta = 0.
        self.search_radius = search_radius
        self.frequencies = [[]]
        self.bounds = np.array([np.zeros(dimension), np.ones(dimension)])
        self.working_plan = random_plan(dimension, self.bounds)
        self.dimension = dimension
        self.num_frequencies = num_frequencies
        self.num_players = num_players
        self.max_it = float(max_it)
        self.alpha = alpha
        if frequencies is None:
            self.generate_frequencies()
        else:
            self.frequencies = frequencies
        self.consensus_threshold = consensus_threshold
        self.trajectories = {-1: []}
        self.distances = {-1: []}
        if max_s is None:
            self.max_sum = self.find_max_s()
        else:
            self.max_sum=max_s
        if min_s is None:
            self.min_sum = self.find_min_s()
        else:
            self.min_sum=min_s
        self.noise = noise
        self.num_memory = num_memory
        self.num_opinions = num_opinions
        self.init_players()
        self.track_convergence = track_convergence
        self.convergence = []
        self.true_plan_utility = 0.

    def init_players(self):
        
        # Make players
        self.players = []
        for i in xrange(0, self.num_players):
            self.players += [Agent(self.dimension, self.noise, 0, self.num_memory, self.num_opinions, self.search_radius, self.true_utility, i+1, self.bounds)]
        # Inform of fellows
        for player in self.players:
            player.set_other_players(self.players)
            self.trajectories[player.player_id] = []
            self.distances[player.player_id] = []

    def set_bounds(self, bounds):
        self.bounds = bounds
        #print bounds
        self.working_plan = random_plan(self.dimension, bounds)
        for player in self.players:
            player.bounds = bounds
            player.own_plan = random_plan(self.dimension, bounds)

    def generate_frequencies(self):
        """ Populate frequencies.
        """
        self.frequencies = np.array([[random.uniform(0, 50.) for y in xrange(0, self.num_frequencies)] for x in xrange(0, self.dimension)], dtype=DTYPE)
        # prself.frequencies

    def true_utility(self, plan):
        """ Return the true utility of a plan.
        """
        summation = self.s_eq(np.array(plan))
        utility = summation - self.min_sum
        utility /= self.max_sum - self.min_sum
        if utility > 1 or utility < 0:
            print "Utility out of bounds:", utility, "summation =", summation
        return utility

    def find_max_s(self):
        """ Use an initial brute force search followed by scipy
        optimisation to find the maximum of the s function.
        """
        def f(plan):
            return -self.s_eq(plan)
        search_bounds = [(0, 1)] * self.dimension
        grid = tuple([(0, 1, self.search_radius)] * self.dimension)
        minimised = opt.minimize(f, opt.brute(f, grid), bounds=search_bounds, method='L-BFGS-B', tol=1e-16, options={'disp': False})
        return -minimised['fun']

    def find_min_s(self):
        """ Use an initial brute force search followed by scipy
        optimisation to find the minimum of the s function.
        """
        search_bounds = [(0, 1)] * self.dimension
        grid = tuple([(0, 1, self.search_radius)] * self.dimension)
        f = self.s_eq
        minimised = opt.minimize(f, opt.brute(f, grid), bounds=search_bounds, method='L-BFGS-B', tol=1e-16, options={'disp': False})
        return minimised['fun']

    
    def s_eq(self, plan):
        """ Compute the sum of sinusoids at some set
        of points.
        """
        return np.sum(np.sin(self.frequencies * np.reshape(plan, (-1, 1))))

    def print_s_eq(self):
        """ Return the equation that gives the s(v) for the utility
        function.
        """
        eq = "s(v)="
        for i in xrange(0, self.dimension):
            for j in xrange(0, self.num_frequencies):
                eq += "sin(%fx_%d) + " % (self.frequencies[i][j], i)
        return eq

    def choose_speaker(self, players):
        """ Pick somebody at random to make a suggestion.
        """
        return random.choice(players)

    def update_plan(self, opinion, proposer, players):
        """ Incorporate an opinion into the global plan
        if enough yes votes are taken.
        """
        if self.motion_carried(self.vote(proposer, opinion, players), players):
            self.working_plan, util = opinion
            #print "Carried", opinion
            #print "Actual util", self.true_utility(self.working_plan)

    def motion_carried(self, yes_votes, players):
        """ Return true if this plan has enough votes
        to replace the group plan.
        """
        result = 1. + yes_votes
        result /= len(players)
        return result > self.theta

    def vote(self, proposer, opinion, players):
        """ Take a vote on a proposed change.
        """
        yes_votes = 0
        for player in players:
            if player is not proposer:
                if player.consider_plan(proposer, opinion, self.working_plan):
                    yes_votes += 1
        #print "%d votes for." % yes_votes
        return yes_votes

    def update_theta(self):
        """ Update theta as turns progress.
        """
        self.theta = self.current_it / self.max_it

    def update_temperature(self, players):
        """ Update cognition temperature of agents.
        """
        temp = self.alpha * (self.max_it / self.current_it)
        for agent in players:
            agent.temp = temp

    def do_turn(self, players):
        """ Iterate the discussion.
        """
        self.update_temperature(players)
        self.update_plans(players)
        self.update_theta()
        speaker = self.choose_speaker(players)
        opinion = speaker.choose_opinion(self.working_plan)
        #print speaker, "proposed", opinion
        self.update_plan(opinion, speaker, players)

    def update_plans(self, players):
        """ Have all the agents search for a better plan.
        """
        for agent in players:
            agent.update()

    def consensus_reached(self, threshold, players):
        """ Return true if the sum of distances between individual
        plans and the group plan is less than threshold.
        """
        distance_sum = 0.
        for player in players:
            distance_sum += player.get_distance(self.working_plan)
        #print distance_sum
        return distance_sum < threshold

    def get_distance(self, players):
        """ Return the sum of distances from the working plan.
        """
        distance_sum = 0.
        for player in players:
            distance_sum += player.get_distance(self.working_plan)
        return distance_sum

    
    def do_discussion(self, players=None):
        """ Run a discussion.
        """
        
        self.working_plan = random_plan(self.dimension, self.bounds)
        if players is None:
            players = self.players
        for player in players:
            player.own_plan = random_plan(self.dimension, self.bounds)
            player.own_util = get_utility(player, player.own_plan)
        for i in xrange(1, int(self.max_it)):
            self.current_it = float(i)
            self.store_trajectories(players)
            self.store_plan_distance(players)
            if self.track_convergence:
                self.store_convergence()
            self.do_turn(players)
            if self.consensus_reached(self.consensus_threshold, players):
                self.store_trajectories(players)
                self.store_plan_distance(players)
                self.true_plan_utility = self.true_utility(self.working_plan)
                return None
        self.true_plan_utility = self.true_utility(self.working_plan)

    def store_trajectories(self, players):
        """ Store the group plan, and each agent's plan.
        """
        self.trajectories[-1] += [tuple(self.working_plan)]
        for player in players:
            self.trajectories[player.player_id] += [tuple(player.own_plan)]

    def store_plan_distance(self, players):
        """ Record the sum of distance to the working plan, and the distances of
        individual agents.
        """
        self.distances[-1] += [self.get_distance(players)]
        for player in players:
            self.distances[player.player_id] += [player.get_distance(self.working_plan)]

    def store_convergence(self):
        self.convergence += [self.pairwise_convergence(100)]

    def t_diff(self, n, players=None):
        util_a, util_b, pair_avg, pair_sum, diff_sum
        
        if players is None:
            players = self.players
        pairs = set(combinations(players, 2))
        plans = np.random.random((n, self.dimension))
        vals = dict(zip(players, [np.zeros(n)]*len(players)))
        t_sum = 0.
        for player in players:
            for i in xrange(n):
                vals[player][i] = player.get_utility(plans[i])
        for a, b in pairs:
            diff = np.abs(vals[a] - vals[b])
            t = np.sum(diff)
            t /= np.sqrt((n*np.sum(np.power(diff, 2)) - np.power(np.sum(diff), 2)) / (n - 1))
            pra, b, t
            t_sum += t
        t_sum /= len(pairs)
        print "T_avg", t_sum


    def pairwise_convergence(self, n, players=None):
        """ Compute the average percentage difference in utility functions
        of all agents across n random points in the problem space.
        """
        diff_sum
        
        if players is None:
            players = self.players
        pairs = set(combinations(players, 2))
        diff_sum = 0.
        plans = np.random.random((n, self.dimension))
        vals = np.zeros((len(players), n))
        for player in players:
            for i in xrange(n):
                vals[player.player_id-1, i] = get_utility(player, plans[i])
        for a, b in pairs:
            diff = np.abs(vals[a.player_id-1] - vals[b.player_id-1])
            pair_avg = vals[a.player_id-1] + vals[b.player_id-1]
            pair_avg /= 2.
            diff_sum += np.mean(diff / pair_avg)
        return diff_sum / len(pairs)

    def fidelity(self, n, players=None):
        """ Compute the average percentage difference between
        the constructed utility functions of agents and the
        true landscape. """
        diff_sum
        
        if players is None:
            players = self.players
        diff_sum = 0.
        plans = np.random.random((n, self.dimension))
        vals = np.zeros((len(players), n))
        true_vals = np.zeros(n)
        for player in players:
            for i in xrange(n):
                vals[player.player_id-1, i] = get_utility(player, plans[i])
        for i in xrange(n):
            true_vals[i] = self.true_utility(plans[i])
        for player in players:
            diff = np.abs(vals[player.player_id-1] - true_vals)
            pair_avg = vals[player.player_id-1] + true_vals
            pair_avg /= 2.
            diff_sum += np.mean(diff / pair_avg)
        return diff_sum / len(players)


def convergence_fixed_paired(runs=100, players=3, landscapes=1,max_it=100,
    step_size=1, max_mem=50):
    """ Run some number of replications of the three discussion types and return a results dictionary.
    """
    num_players = players
    consensus_threshold = -1  # No consensus
    results = {'fields': ['q', 'run', 'utility','fidelity','convergence', 'protocol','dimensions','landscape'], 'results': []}
    # q 0 - 10
    count = 1

    discussions = []
    frequencies = []
    for i in xrange(landscapes):
        d = Discussion(dimension, num_players, 0, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
        frequencies += [(d.frequencies, d.max_sum, d.min_sum)]
    for num_memory in xrange(0, max_mem+step_size, step_size):
        # 100 runs of each
        for i in xrange(runs):
            for l in xrange(landscapes):
                print "Making run %d of %d.." % (count, runs*landscapes*51)
                count += 1
                discussions += [('Standard', i, num_memory, Discussion,dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, False, 2, d[0], d[1], d[2])]
                discussions += [('Standard_100', i, num_memory, Discussion,dimension, num_players, num_memory, num_opinions, num_frequencies, 100, alpha, noise, search_radius, consensus_threshold, False, 2, d[0], d[1], d[2])]
    random.shuffle(discussions)
    pool = Pool()
    results['results'] = pool.map(run_discussion, discussions)
    #print results['results']
    return results

def q_convergence(runs=100, players=3, landscapes=1):
    """ Run an experiment recording time to convergence of individual
    plans to the group plan for q 0 ~ 10.
    Returns a results dictionary.
    """
    num_players = players
    max_it = 200
    consensus_threshold = 0.04
    results = {'fields': ['q', 'run', 'time'], 'results': []}
    # q 0 - 10
    count = 1
    
    discussions = []
    frequencies = []
    for i in xrange(landscapes):
        d = Discussion(dimension, num_players, 0, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
        frequencies += [(d.frequencies, d.max_sum, d.min_sum)]
    for num_memory in xrange(11):
        # 100 runs of each
        for i in xrange(runs):
            for d in frequencies:
                print "Making run %d of %d.." % (count, runs*landscapes*11)
                count += 1
                discussions += [('Standard', i, num_memory, Discussion,dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, False, 2, d[0], d[1], d[2])]
    random.shuffle(discussions)
    pool = Pool(2)
    results['results'] = pool.map(run_q_convergence, discussions)
    print results['results']
    return results


def run():
    #dump_experiment("convergence_fixed.csv", convergence_fixed_paired(players=3, runs=1, landscapes=100))
    t = time()
    dump_experiment("q_convergence.csv", q_convergence(players=3, runs=10, landscapes=1))
    print time() -  t

if  __name__ =='__main__':
    run()