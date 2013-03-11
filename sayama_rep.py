import numpy as np
import random
import collections

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
		aspects = [np.arange(resolution, 1, resolution)] * dimensions
		points = cartesian(aspects)
		return [(x, utility_fn(x)) for x in points]


def dump_landscape(file, dimensions, landscape):
	""" Dump a landscape to a file.
	"""

	line = "%f," * dimensions + "%f\n"
	for record in landscape:
		points, utility = record
		file.write(line % tuple(list(points) + [utility]))


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


	def __init__(self, dimensions, noise, temp, num_memory, num_opinions, search_radius, global_util):
		# Randomly initialize own plan
		own_plan = np.random.random(dimensions)
		self.opinions = set([])
		self.others_plan = {}
		self.sample(global_util, num_opinions, noise, dimensions)
		self.own_util = self.get_utility(own_plan)
		self.own_plan = own_plan
		self.noise_amp = noise
		self.num_memory = num_memory
		self.search_radius = search_radius
		self.temp = temp


	def set_other_players(self, players):
		""" Add all the other players to mental model.
		"""

		print "startingplayers",self.others_plan
		for player in players:
			if player is not self:
				self.others_plan[player] = collections.deque([], num_memory)
			else:
				print "Self:",self,"Other:",player
		print "players",self.others_plan


	def sample(self, utility_fn, num_opinions, noise, dimensions):
		""" Sample some number of opinions from the true utility with noise.
		"""
		while len(self.opinions) <= num_opinions:
			plan = np.random.random(dimensions)
			fuzz = (noise + noise) * np.random.rand() - noise
			self.opinions.add((tuple(plan), utility_fn(plan) + fuzz))


	def get_utility(self, plan):
		""" Return the internal utility of a plan.
		"""
		# Build list of all known plans
		plans = []
		plans += self.opinions
		#if self.own_plan != None:
		#	plans += [(tuple(self.own_plan), self.own_util)]
		for agent, plan_mem in self.others_plan.items():
			plans += plan_mem


		#print plans
		#print map(lambda x: self.get_weight(x[0], plan), plans)

		results = np.array(map(lambda x: self.get_weight(x[0], plan), plans))

		denom = np.sum(results)

		utility = np.sum(np.array(map(lambda x: (self.get_weight(x[0], plan) / denom) * x[1], plans)))
		#print denom, utility, plan, results

		return utility


	def consider_plan(self, agent, opinion):
		""" Consider a proposed change to the group plan
		and either incorporate it, or incorporate and support
		it.
		"""
		incorp = self.incorporate(opinion)
		support = False
		if not incorp:
			support = self.support(opinion)
		# Remember the opinion
		self.others_plan[agent].appendleft(opinion)
		print self, "considered plan by", agent, incorp, support, "Response to plan", opinion
		return support or incorp


	def incorporate(self, opinion):
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
			a = np.array(plan)
			b = np.array(self.own_plan)
			#print a, b, "plans"
			norm = np.linalg.norm((a - b), ord=1)
			return np.exp(-pow(norm, 2) / self.temp)

		return np.random.rand() < p_accept(plan)


	def update(self):
		""" Search for a new, more optimal plan within search
		radius.
		"""
		own_plan = self.own_plan + np.random.random(len(self.own_plan)) * self.search_radius
		self.own_util = self.get_utility(own_plan)
		self.own_plan = own_plan


	def choose_opinion(self, working_plan):
		""" Make a suggestion to modify an aspect
		of the group plan.
		"""
		util_working = self.get_utility(working_plan)
		tmp_plan = list(working_plan)
		best_plan = working_plan
		util_best = util_working
		for i in range(0, len(working_plan)):
			tmp_plan = list(working_plan)
			tmp_plan[i] = self.own_plan[i]
			if self.get_utility(tmp_plan) > util_best:
				best_plan = tuple(tmp_plan)
		return (best_plan, util_best)
			

	def set_temperature(self, new_temp):
		self.temp = new_temp


	def get_weight(self, plan_a, plan_b):
		""" Get the weight of plan_b as the normalised
		inverse square distance from plan_a.
		"""

		a = np.array(plan_a)
		b = np.array(plan_b)
		norm = np.linalg.norm((a - b), ord=1)
		#print a, b, norm, "weight", pow(norm, -2)
		if norm == 0.0:
			norm = 1.0
		return pow(norm, -2)


	def get_distance(self, plan):
		""" Get the distance between this plan and our
		internal plan.
		"""

		a = np.array(self.own_plan)
		b = np.array(plan)
		return np.linalg.norm((a - b), ord=1)


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

	def __init__(self, dimension, num_players, num_memory, num_opinions, num_frequencies, 
		max_it, alpha, noise, search_radius, consensus_threshold):
		self.players = []
		self.theta = 0.
		self.frequencies = [[]]
		self.working_plan = np.random.random(dimension)
		self.dimension = dimension
		self.num_frequencies = num_frequencies
		self.num_players = num_players
		self.max_it = float(max_it)
		self.alpha = alpha
		self.generate_frequencies()
		self.consensus_threshold = consensus_threshold
		# Make players
		for i in range(0, num_players):
			self.players += [Agent(dimension, noise, 0, num_memory, num_opinions, search_radius, self.true_utility)]
		# Inform of fellows
		for player in self.players:
			player.set_other_players(self.players)


	def generate_frequencies(self):
		""" Populate frequencies.
		"""
		self.frequencies = [[random.uniform(0, 50.) for y in range(0, self.num_frequencies)] for x in range(0, self.dimension)]
		print self.frequencies


	def true_utility(self, plan):
		""" Return the true utility of a plan.
		"""
		summation = 0
		max_sum = self.num_frequencies*self.dimension
		for i in range(0, len(plan)):
			for j in range(0, self.num_frequencies):
				summation += np.sin(self.frequencies[i][j] * plan[i])
		utility = summation + max_sum
		utility /= max_sum + max_sum
		return utility


	def choose_speaker(self):
		""" Pick somebody at random to make a suggestion.
		"""
		return random.choice(self.players)


	def update_plan(self, opinion, proposer):
		""" Incorporate an opinion into the global plan
		if enough yes votes are taken.
		"""
		if self.motion_carried(self.vote(proposer, opinion)):
			self.working_plan, util = opinion
			print "Carried", opinion
			print "Actual util", self.true_utility(self.working_plan)


	def motion_carried(self, yes_votes):
		""" Return true if this plan has enough votes
		to replace the group plan.
		"""
		result = 1. + yes_votes
		result /= len(self.players)
		return result > self.theta


	def vote(self, proposer, opinion):
		""" Take a vote on a proposed change.
		"""
		yes_votes = 0
		for player in self.players:
			if player is not proposer:
				if player.consider_plan(proposer, opinion):
					yes_votes += 1
		return yes_votes


	def update_theta(self):
		""" Update theta as turns progress.
		"""
		self.theta = self.current_it / self.max_it


	def update_temperature(self):
		""" Update cognition temperature of agents.
		"""
		temp = self.alpha * (self.max_it / self.current_it)
		for agent in self.players:
			agent.set_temperature(temp)


	def do_turn(self):
		""" Iterate the discussion.
		"""
		self.update_temperature()
		self.update_plans()
		self.update_theta()
		speaker = self.choose_speaker()
		opinion = speaker.choose_opinion(self.working_plan)
		print speaker, "proposed", opinion
		self.update_plan(opinion, speaker)


	def update_plans(self):
		""" Have all the agents search for a better plan.
		"""
		for agent in self.players:
			agent.update()


	def consensus_reached(self, threshold):
		""" Return true if the sum of distances between individual
		plans and the group plan is less than threshold.
		"""
		distance_sum = 0.
		for player in self.players:
			distance_sum += player.get_distance(self.working_plan)
		return distance_sum < threshold



	def do_discussion(self):
		""" Run a discussion.
		"""
		for i in range(1, int(self.max_it)):
			self.current_it = float(i)
			self.do_turn()
			if self.consensus_reached(self.consensus_threshold):
				return None


if __name__=="__main__":
	dimension = 2
	num_players = 6
	num_memory = 0
	num_opinions = 20
	num_frequencies = 5
	max_it = 200
	alpha = 0.05
	noise = 0.2
	search_radius = 0.005
	consensus_threshold = 0.04
	discussion = Discussion(dimension, num_players, num_memory, num_opinions, num_frequencies, 
		max_it, alpha, noise, search_radius, consensus_threshold)
	#print landscape(search_radius, dimension, discussion.true_utility)
	landscape_file = open("landscape.tsv", 'w')
	dump_landscape(landscape_file, dimension, landscape(0.02, dimension, discussion.true_utility))
	landscape_file.close()
	#discussion.do_discussion()

