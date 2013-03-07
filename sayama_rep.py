import numpy as np

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


class Agent:
	""" A simple planning agent.
	"""
	# Number of opinions to remember about other agents
	num_memory = 0
	# Shared opinions - queue of tuples of a list of coords & an opinion
	others_plan = {}
	# Original opinions
	opinions = set([])
	# Own 'best' plan
	own_plan = []
	# Utility of own plan
	own_util = 0
	# Cognition temperature
	temp = 0.
	# Search radius
	search_radius = 0.
	# Noise amplitude
	noise_amp = 0.


	def __init__(self, dimensions, noise, temp, num_memory, num_opinions, search_radius, global_util):
		# Randomly initialize own plan
		self.own_plan = np.random.random(dimensions)
		self.own_util = self.get_utility(own_plan)
		self.sample(global_util, num_opinions, noise, dimensions)
		self.noise_amp = noise
		self.num_memory = num_memory
		self.search_radius = search_radius
		self.temp = temp


	def set_other_players(players):
		""" Add all the other players to mental model.
		"""

		for player in players:
			if player != self:
				self.others_plan[player] = collections.deque([], num_memory)


	def sample(utility_fn, num_opinions, noise, dimensions):
		""" Sample some number of opinions from the true utility with noise.
		"""
		while len(self.opinions) <= num_opinions:
			plan = np.random(dimensions)
			fuzz = (noise + noise) * np.random.rand() - noise
			self.opinions.add((plan, utility_fn(plan) + fuzz))


	def get_utility(plan):
		""" Return the internal utility of a plan.
		"""
		utility = 0

		return utility


	def consider_plan(agent, opinion):
		""" Consider a proposed change to the group plan
		and either incorporate it, or incorporate and support
		it.
		"""
		# Remember the opinion
		self.others_plan[agent].appendleft(opinion)
		return self.incorporate(opinion) or self.support(opinion)


	def incorporate(opinion):
		""" Choose whether to incorporate a new opinion into
		this agent's plan.
		"""
		plan, utility = opinion
		expect_util = self.get_utility(plan)
		if expect_util > self.own_util:
			self.own_plan = plan
			self.own_util = expect_util
			return True
		return False


	def support(opinion):
		""" Return true to vote the opinion into the global
		plan.
		"""
		plan, utility = opinion


	def update():
		""" Search for a new, more optimal plan.
		"""


	def choose_opinion(working_plan):
		""" Make a suggestion to modify an aspect
		of the group plan.
		"""
		util_working = self.get_utility(working_plan)
		tmp_plan = working_plan
		best_plan = working_plan
		util_best = util_working
		for i in range(0, len(working_plan)):
			tmp_plan = working_plan
			tmp_plan[i] = self.own_plan[i]
			if self.get_utility(tmp_plan) > util_best:
				best_plan = tmp_plan
		return (best_plan, util_best)
			


	def update_temperature(new_temp):
		self.temp = new_temp


class Discussion:
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
	max_it = 0
	# Current iteration
	current_it = 0
	# Cognition temperature scaler
	alpha = 0.

	def __init__(self, dimension, num_players, num_memory, num_opinions, num_frequencies, max_it,
		alpha, noise, search_radius):
		self.working_plan = np.random.random(dimension)
		self.dimension = dimension
		self.num_frequencies = num_frequencies
		self.num_players = num_players
		self.max_it = max_it
		self.alpha = alpha
		self.generate_frequencies()
		for i in range(0, num_players):
			players += [Agent(dimension, noise, 0, num_memory, num_opinions, search_radius, self.true_utility)]


	def generate_frequencies():
		""" Populate frequencies.
		"""
		self.frequencies = [[random(0, 50) for x in range(0, self.num_frequencies)] for x in range(0, self.dimension)]


	def true_utility(plan):
		""" Return the true utility of a plan.
		"""
		summation = 0

		for i in range(0, len(plan)):
			for j in range(0, num_frequencies):
				summation += np.sin(frequencies[i, j] * plan[i])
		utility = summation - floor(summation)
		utility /= ceil(summation) - floor(summation)
		return utility


	def choose_speaker():
		""" Pick somebody at random to make a suggestion.
		"""
		return random.choice(self.players)


	def update_plan(opinion, proposer):
		""" Incorporate an opinion into the global plan
		if enough yes votes are taken.
		"""
		if self.motion_carried(vote(proposer, opinion)):
			self.working_plan = opinion


	def motion_carried(yes_votes):
		""" Return true if this plan has enough votes
		to replace the group plan.
		"""
		result = 1. + yes_votes
		result /= len(players)
		return result > self.theta


	def vote(proposer, opinion):
		""" Take a vote on a proposed change.
		"""
		yes_votes = 0
		for player in self.players:
			if player.consider_plan(proposer, opinion):
				yes_votes += 1
		return yes_votes


	def update_theta():
		""" Update theta as turns progress.
		"""
		self.theta = self.current_it / self.max_it


	def update_temperature():
		""" Update cognition temperature of agents.
		"""
		temp = self.alpha * (self.max_it / self.current_it)
		for agent in self.players:
			agent.set_temperature(temp)


	def do_turn():
		""" Iterate the discussion.
		"""
		speaker = self.choose_speaker()
		opinion = speaker.choose_opinion(working_plan)
		self.update_plan(opinion, speaker)
		self.update_theta()
		self.update_temperature()
		self.update_plans()


	def update_plans():
		""" Have all the agents search for a better plan.
		"""
		for agent in players:
			agent.update_plan()


	def do_discussion():
		""" Run a discussion.
		"""
		for i in range(0, self.max_it):
			self.do_turn()
			self.current_it = i


