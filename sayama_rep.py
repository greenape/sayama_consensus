import numpy as np

class Agent:
	""" A simple planning agent.
	"""
	# Number of opinions to remember about other agents
	num_memory = 0
	# Shared opinions
	# Original opinions
	# Mental models of other agents collections.deque
	# Cognition temperature
	# Search radius
	# Noise amplitude

	def sample(plan, opinions, noise):
		""" Sample some number of opinions from a plan with noise.
		"""

	def get_utility(opinion):
		""" Return the internal utility of an opinion.
		"""

	def incorporate(opinion):
		""" Choose whether to incorporate a new opinion into
		this agent's plan.
		"""

	def support(opinion):
		""" Return true to vote the opinion into the global
		plan.
		"""

	def update():
		""" Search for a new, more optimal plan.
		"""

	def choose_opinion():
		""" Select an opinion of mine to suggest to the
		group plan.
		"""

class Plan:
	""" A plan, m-dimensional collection of choices
	in a problem space.
	"""



class Discussion:
	# The true utility function
	true_plan
	# Current working aproximation
	working_plan
	# Discussion participants
	players
	# Incorporation threshold
	theta
	# Problem space dimension


	def true_utility(plan):
		""" Return the true utility of a plan.
		"""

	def choose_speaker():
		""" Pick somebody to make a suggestion.
		"""

	def update_plan(opinion, yes_votes):
		""" Incorporate an opinion into the global plan
		if enough yes votes are taken.
		"""
