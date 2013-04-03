# cython: profile=True
import numpy as np
import random
import collections
import scipy.optimize as opt
from itertools import combinations
import quadtree
import cython
from time import time
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# Default settings
cdef int dimension = 2
cdef int num_players = 6
cdef int num_memory = 0
cdef int num_opinions = 20
cdef int num_frequencies = 5
cdef int max_it = 100
cdef float alpha = 0.05
cdef float noise = 0.2
cdef float search_radius = 0.005
cdef float consensus_threshold = 0.04


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
        # print line % tuple(list(points) + [utility])
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

def random_plan(dimensions, bounds):
    """ Generate a random plan within the given bounds of
    the specified number of dimensions.
    """
    return (bounds[1] - bounds[0]) * np.random.random(dimensions) + bounds[0]


class Agent:
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

    def __init__(self, int dimensions, float noise, temp, int num_memory, int num_opinions, float search_radius, global_util,
        int player_id, np.ndarray[DTYPE_t, ndim=2] bounds):
        # Randomly initialize own plan
        self.own_plan = random_plan(dimensions, bounds)
        self.opinions = set([])
        self.others_plan = {}
        self.sample(global_util, num_opinions, noise, dimensions)
        self.own_util = None
        self.own_util = self.get_utility(self.own_plan)
        self.noise_amp = noise
        self.num_memory = num_memory
        self.search_radius = search_radius
        self.temp = temp
        self.player_id = player_id
        self.bounds = bounds
        self.dimension = dimensions

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

    @cython.boundscheck(False)
    def get_utility(self, plan):
        """ Return the internal utility of a plan.
        """
        # Build list of all known plans (all v_i,j,k)
        plans = []
        plans += self.opinions
        matched = False
        cdef float utility = 0.
        #if self.own_util != None:
        #   plans += [(tuple(self.own_plan), self.own_util)]
        for agent, plan_mem in self.others_plan.items():
            plans += plan_mem

        plan_list_tmp, utils_tmp = zip(*plans)
        cdef np.ndarray[DTYPE_t, ndim=2] plan_list = np.array(plan_list_tmp, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t] utils = np.array(utils_tmp, dtype=DTYPE)

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

        cdef np.ndarray[DTYPE_t] weights = np.array([self.get_weight(x, np.array(plan)) for x in plan_list], dtype=DTYPE)
        #results = np.array(map(lambda x: self.get_weight(x[0], plan), plans))
        #denom = np.sum(results)
        cdef float util = np.sum(weights / np.sum(weights) * utils)

        #utility = np.sum(np.array(map(lambda x: (self.get_weight(x[0], plan) / denom) * x[1], plans)))
        #print "New util.",denom, utility, plan, results
        #assert(util == utility)
        return util

    def consider_plan(self, agent, opinion, working_plan):
        """ Consider a proposed change to the group plan
        and either incorporate it, or incorporate and support
        it.
        """
        self.own_util = self.get_utility(self.own_plan)
        incorp = self.incorporate(opinion, working_plan)
        support = False
        if not incorp:
            support = self.support(opinion)
        self.others_plan[agent].appendleft(opinion)
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
        cdef float expect_util = self.get_utility(diff)
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

        def p_accept(plan):
            """ Returns the probability of accepting a
            plan based on current temp and distance from
            own plan.
            """
            cdef np.ndarray[DTYPE_t] a = np.array(plan)
            cdef np.ndarray[DTYPE_t] b = np.array(self.own_plan)
            #print a, b, "plans"
            cdef np.ndarray[DTYPE_t] c = a - b
            cdef float norm = c.dot(c)
            return np.exp(-norm / self.temp)

        return np.random.rand() < p_accept(plan)

    def update(self):
        """ Search for a new, more optimal plan within search
        radius.
        """

        def f(plan):
            """ Negative of utility function for
            minimising.
            """
            return -self.get_utility(plan)

        def constraint_1(x):
            """ Constraint keeps solutions within self.radius
            of current plan.
            """
            if self.search_radius > self.get_distance(x):
                return 1
            return -1

        def constraint_2(x):
            """ Constrain answers within max & min bound.
            """
            if (self.bounds[1] >= x).all() and (self.bounds[0] <= x).all():
                return 1
            return -1

        #min_bounds = [max(self.own_plan[x] - self.search_radius,self.bounds[0][x]) for x in xrange(self.dimension)]
        #max_bounds = [min(self.own_plan[x] + self.search_radius,self.bounds[1][x]) for x in xrange(self.dimension)]
        #search_bounds = zip(min_bounds, max_bounds)
        new_plan = opt.fmin_cobyla(f, np.array(self.own_plan), [constraint_1, constraint_2], disp=0)
        self.own_util = self.get_utility(new_plan)
        #print search_bounds
        #print self.own_plan, "New Plan", new_plan
        self.own_plan = new_plan

    @cython.boundscheck(False)
    def choose_opinion(self, working_plan):
        """ Make a suggestion to modify an aspect
        of the group plan.
        """
        # Work out possible new plans
        possible_plans = []
        cdef float util_working = self.get_utility(working_plan)
        for i in xrange(0, len(working_plan)):
            tmp_plan = list(working_plan)
            tmp_plan[i] = self.own_plan[i]
            #print tmp_plan, "Tmp plan"
            possible_plans += [(tuple(tmp_plan), self.get_utility(tmp_plan) - util_working)]

        return self.weighted_choice(possible_plans)

    def get_weight(self, np.ndarray[DTYPE_t] plan_a, np.ndarray[DTYPE_t] plan_b):
        """ Get the weight of plan_b as the normalised
        inverse square distance from plan_a.
        """

        #a = np.array(plan_a)
        #b = np.array(plan_b)
        cdef np.ndarray[DTYPE_t] c = plan_a - plan_b
        cdef float norm = np.sqrt(c.dot(c))
        if norm == 0:
            return 0.
        #print a, b, norm, "weight", pow(norm, -2)
        return pow(norm, -2)

    def get_distance(self, plan):
        """ Get the distance between this plan and our
        internal plan.
        """

        cdef np.ndarray[DTYPE_t] a = np.array(self.own_plan)
        cdef np.ndarray[DTYPE_t] b = np.array(plan)
        cdef np.ndarray[DTYPE_t] c = a - b
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
            return abs(self.get_utility(plan) - utility_fn(plan))
        return f

    def weighted_choice(self, plans):
        # Normalize util gains
        plan_list, util_list = zip(*plans)
        cdef np.ndarray[DTYPE_t] utils = np.array(util_list)
        cdef float max_util = np.max(utils)
        cdef float min_util = np.min(utils)
        cdef float diff = max_util - min_util
        if diff == 0:
            utils = np.ones(utils.size) / float(utils.size)
        else:
            utils = (utils - min_util) / diff
        cdef float total = np.sum(utils)
        cdef float threshold = random.uniform(0, total)
        cdef float bracket = 0.
        for i in xrange(len(plans)):
            if bracket + utils[i] > threshold:
                return (plan_list[i], self.get_utility(plan_list[i]))
            bracket += utils[i]
        return (plan_list[i], self.get_utility(plan_list[i]))

    def __str__(self):
        return "Agent id: %d" % self.player_id

    def __unicode__(self):
        return self.__str__()


class HeadlessChicken(Agent):
    def choose_opinion(self, working_plan):
        """ Make a suggestion to modify an aspect
        of the group plan, at random.
        """
        # Work out possible new plans
        possible_plans = []
        for i in xrange(0, len(working_plan)):
            tmp_plan = list(working_plan)
            tmp_plan[i] = (self.bounds[i][1] - self.bounds[i][0]) * np.random.rand() + self.bounds[i][0]
            #print tmp_plan, "Tmp plan"
            possible_plans += [(tuple(tmp_plan), self.get_utility(tmp_plan))]
        return self.weighted_choice(possible_plans)

    def __str__(self):
        return "Headless Chicken id: %d" % self.player_id


class Discussion:
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

    def __init__(self, dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold):
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
        self.generate_frequencies()
        self.consensus_threshold = consensus_threshold
        self.trajectories = {-1: []}
        self.distances = {-1: []}
        self.max_sum = self.find_max_s()
        self.min_sum = self.find_min_s()
        self.noise = noise
        self.num_memory = num_memory
        self.num_opinions = num_opinions
        self.init_players()

    def init_players(self):
        # Make players
        for i in xrange(0, self.num_players):
            self.players += [Agent(self.dimension, self.noise, 0, self.num_memory, self.num_opinions, self.search_radius, self.true_utility, i+1, self.bounds)]
        # Inform of fellows
        for player in self.players:
            player.set_other_players(self.players)
            self.trajectories[player.player_id] = []
            self.distances[player.player_id] = []

    def set_bounds(self, bounds):
        self.bounds = bounds
        self.working_plan = random_plan(self.dimension, bounds)
        for player in self.players:
            player.bounds = bounds
            player.own_plan = random_plan(self.dimension, bounds)

    def generate_frequencies(self):
        """ Populate frequencies.
        """
        self.frequencies = np.array([[random.uniform(0, 50.) for y in xrange(0, self.num_frequencies)] for x in xrange(0, self.dimension)], dtype=DTYPE)
        # print self.frequencies

    def true_utility(self, plan):
        """ Return the true utility of a plan.
        """
        cdef float summation = self.s_eq(plan)
        cdef float utility = summation - self.min_sum
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

    @cython.boundscheck(False)
    def s_eq(self, np.ndarray[dtype=DTYPE_t] plan):
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
        cdef float result = 1. + yes_votes
        result /= len(players)
        return result > self.theta

    def vote(self, proposer, opinion, players):
        """ Take a vote on a proposed change.
        """
        cdef float yes_votes = 0
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
        cdef float temp = self.alpha * (self.max_it / self.current_it)
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
        cdef float distance_sum = 0.
        for player in players:
            distance_sum += player.get_distance(self.working_plan)
        #print distance_sum
        return distance_sum < threshold

    def get_distance(self, players):
        """ Return the sum of distances from the working plan.
        """
        cdef float distance_sum = 0.
        for player in players:
            distance_sum += player.get_distance(self.working_plan)
        return distance_sum

    def do_discussion(self, players=None):
        """ Run a discussion.
        """
        self.working_plan = random_plan(self.dimension, self.bounds)
        if players is None:
            players = self.players
        for i in xrange(1, int(self.max_it)):
            self.current_it = float(i)
            self.store_trajectories(players)
            self.store_plan_distance(players)
            self.do_turn(players)
            if self.consensus_reached(self.consensus_threshold, players):
                self.store_trajectories(players)
                self.store_plan_distance(players)
                return None

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

    @cython.boundscheck(False)
    def pairwise_convergence(self, int n, players=None):
        """ Compute the average percentage difference in utility functions
        of all agents across n random points in the problem space.
        """
        cdef float util_a, util_b, pair_avg, pair_sum, diff_sum
        cdef np.ndarray[dtype=DTYPE_t] plan
        cdef int i
        if players is None:
            players = self.players
        pairs = set(combinations(players, 2))
        diff_sum = 0.
        cdef np.ndarray[dtype=DTYPE_t, ndim=2] plans = np.random.random((n, self.dimension))
        for a, b in pairs:
            fn = a.abs_diff_util(b.get_utility)
            pair_sum = 0.
            for i in xrange(n):
                plan = plans[i]
                util_a = abs(a.get_utility(plan))
                util_b = abs(b.get_utility(plan))
                pair_avg = util_a + util_b
                pair_avg /= 2.
                pair_sum += fn(plan) / pair_avg
            diff_sum += pair_sum / n
        return diff_sum / len(pairs)

class ChickenDiscussion(Discussion):
    """ A discussion with some number of headless Chicken
    participants. """

    def __init__(self, dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, chickens):
        self.chickens = chickens
        super(Discussion, self).__init__(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)

    def init_players(self):
        """ Make players, with some number of them chickens.
        """
        for i in xrange(0, self.num_chickens):
            self.players += [HeadlessChicken(self.dimension, self.noise, 0, self.num_memory, self.num_opinions, self.search_radius, self.true_utility, i+1, self.bounds)]
        for i in xrange(self.num_chickens, self.num_players):
            self.players += [Agent(self.dimension, self.noise, 0, self.num_memory, self.num_opinions, self.search_radius, self.true_utility, i+1, self.bounds)]
        # Inform of fellows
        for player in self.players:
            player.set_other_players(self.players)
            self.trajectories[player.player_id] = []
            self.distances[player.player_id] = []

class StructuredDiscussion(Discussion):
    """ A phased discussion which frames discussions in progressively
    larger areas of the problem space.
    """

    def __init__(self, dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold, depth):
        self.depth = depth
        super(Discussion, self).__init__(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)

    def make_sections(self):
        """ Return a set of 2d discussion spaces in an inverted tree.
        """
        tree = quadtree.Tree(np.array([np.array([0,0]), np.array([0,1]), np.array([1,1]), np.array([1, 0])]), self.depth)
        tree.generate()
        return tree.get_bounds().reverse()

    def do_structured_discussion(self, players=None):
        """ Run a discussion as a backwards breadth first
        traversal of the problem space tree.
        Make a random working, and individual plan within
        the space. Discuss, then move on.
        N.b. time in each space needs to relate to q?
        """
        for space in self.make_sections():
            self.set_bounds(space)
            self.do_discussion()



class PairDiscussion(Discussion):
    """ A discussion where there is an initial phase
    of pairwise discussions on the topic, followed by a
    group discussion phase.
    """

    def make_pairs(self, players):
        """ Return a set of all pairs of agents.
        """
        return set(combinations(players, 2))

    def do_pair_discussion(self, players=None):
        """ Run a discussion in pairs of each agent, then
        a group discussion. N.b. pass on plan to next pair, or new
        working plan each time?
        """

        if players is None:
            players = self.players

        for pair in self.make_pairs(players):
            self.do_discussion(pair)



def q_plan_convergence(runs=100):
    """ Run an experiment recording time to convergence of individual
    plans to the group plan for q 0 ~ 10.
    Returns a dictionary mapping q to convergence times.
    """
    num_players = 3
    max_it = 200
    results = {'fields':['q','run','time'], 'length':runs*10, 'results':[]}
    # q 0 - 10
    count = 1
    for num_memory in xrange(11):
        # 100 runs of each
        for i in xrange(runs):
            print "Run %d of %d" % (count, (runs + 1) * 11)
            time_to_converge = 0
            count += 1
            took = time()
            discussion = Discussion(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
            took = time() - took
            print "Made discussion in %fs" % took
            #print discussion.players
            took = time()
            discussion.do_discussion()
            took = time() - took
            print "Had discussion in %fs" % took
            time_to_converge = (discussion.current_it + 1) / float(max_it)
            results['results'] += [[num_memory, i, time_to_converge]]
            trajectory_file = "trajectories_mem_%d_run_%d.csv" % (num_memory, i)
            dump_trajectories(trajectory_file, dimension, discussion.trajectories)
    return results


@cython.boundscheck(False)
def individual_convergence(runs=100, sample_size=1000):
    num_players = 3
    max_it = 100
    consensus_threshold = -1  # No consensus
    results = {'fields': ['q', 'run', 'convergence'], 'length': runs*50, 'results': []}
    # q 0 - 10
    count = 1
    for num_memory in xrange(51):
        # 100 runs of each
        for i in xrange(runs):
            print "Run %d of %d" % (count, (runs + 1) * 51)
            count += 1
            took = time()
            discussion = Discussion(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
            took = time() - took
            print "Made discussion in %fs" % took
            #print discussion.players
            took = time()
            discussion.do_discussion()
            took = time() - took
            print "Had discussion in %fs" % took
            took = time()
            results['results'] += [[num_memory, i, discussion.pairwise_convergence(sample_size)]]
            took = time() - took
            print "Dumped results in %fs" % took
    return results


def run():
    discussion = Discussion(dimension, num_players, num_memory, num_opinions, num_frequencies, max_it, alpha, noise, search_radius, consensus_threshold)
    #print landscape(search_radius, dimension, discussion.true_utility)
    landscape_file = "landscape.csv"
    dump_landscape(landscape_file, dimension, landscape(0.04, dimension, discussion.true_utility))
    agent_1 = discussion.players[0]
    agent_2 = discussion.players[1]
    dump_landscape("start_l.csv", dimension, landscape(0.04, dimension, agent_1.abs_diff_util(agent_2.get_utility)))
    discussion.do_discussion()
    dump_landscape("end_l.csv", dimension, landscape(0.04, dimension, agent_1.abs_diff_util(agent_2.get_utility)))
    trajectory_file = "trajectories.csv"
    dump_trajectories(trajectory_file, dimension, discussion.trajectories)
    dump_experiment("individual_convergence_rep.csv",individual_convergence(100, sample_size=100))
    dump_experiment("q_convergence_rep.csv", q_plan_convergence(100))
