from sage.all import *
import networkx as nx
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import random
import collections

PUSH = 0
PULL = 1
EXCHANGE = 2 
BROADCAST = 3

FIELD_SIZE = 2 # MUST BE A PRIME OR PRIME POWER!
NUM_NODES = 1000
NUM_MESSAGES = 10
MESSAGE_LENGTH = 100

N_TRIALS = 10

COEFF_MATRIX_SPACE = MatrixSpace(GF(FIELD_SIZE), NUM_MESSAGES, NUM_MESSAGES)
MESSAGE_MATRIX_SPACE = MatrixSpace(GF(FIELD_SIZE), NUM_MESSAGES, MESSAGE_LENGTH)

class RLNCStats:
	def __init__(self):
		self.stopping_rounds = []
	def summarize(self):
		print sp.stats.describe(np.array(self.stopping_rounds))
	def update(self, rounds):
		self.stopping_rounds.append(rounds)

class RLNCNode:
	def __init__(self, identifier):
		self.messages = np.array([]) # A matrix where every row is a message
		self.coefficients = np.array([]) # A matrix where every row is coefficients for a message
		self.identifier = identifier

	def receive(self, message):
		if np.all(message.coefficients==0):
			return # Message is just the 0 vector and hence adds no value
		elif self.can_decode():
			return # No need to receive anything

		if self.messages.size == 0: # I know nothing, so this has to add value
			self.coefficients = np.array([message.coefficients])
			self.messages = np.array([message.message])
		else:
			# Add a message if the rank increases (implies new messsage is linearly independent of old ones)
			old_rank = self.rank()
			new_coefficients = np.concatenate((self.coefficients, np.array([message.coefficients])), axis=0)
			new_messages = np.concatenate((self.messages, np.array([message.message])), axis=0)
			new_rank = compute_rank(new_coefficients)
			if new_rank > old_rank:
				self.coefficients = new_coefficients
				self.messages = new_messages

	def rank(self):
		return compute_rank(self.coefficients)

	def can_decode(self):
		return self.rank() == NUM_MESSAGES

	def get_random_message(self):
		if self.messages.size == 0: # You have nothing to broadcast
			return

		rand_coefficients = np.random.randint(0, FIELD_SIZE, self.messages.shape[0]) # One random coefficient for each message I know about
		message = mod(np.dot(rand_coefficients, self.messages)).astype(np.int)
		coefficients = mod(np.dot(rand_coefficients, self.coefficients)).astype(np.int)
		return RLNCMessage(coefficients, message)

	def decode(self):
		a = self.coefficients
		b = self.messages
		assert a.shape == (NUM_MESSAGES, NUM_MESSAGES)
		assert b.shape == (NUM_MESSAGES, MESSAGE_LENGTH)
		ma = COEFF_MATRIX_SPACE(list(a.flatten()))
		inv = ma.inverse()
		mb = MESSAGE_MATRIX_SPACE(list(b.flatten()))
		return inv * mb

class RLNCMessage:
	def __init__(self, coefficients, message):
		self.coefficients = coefficients
		self.message = message

	def __str__(self):
		return "coefficients:" + str(self.coefficients) + "message:" + str(self.message)

def compute_rank(matrix):
	if matrix.size == 0:
		return 0
	MS = MatrixSpace(GF(FIELD_SIZE), matrix.shape[0], matrix.shape[1])
	M = MS(list(matrix.astype(np.int).flatten()))
	return M.rank()

def summarize(graph):
	for node in graph.nodes():
		rank = graph.node[node]['rlnc'].rank()
		print "Node %d has rank %d / %d" % (node, rank, NUM_MESSAGES)

def mod(array):
	return np.mod(array, FIELD_SIZE)

def can_stop_sending(graph):
	return all([graph.node[node]['rlnc'].can_decode() for node in graph.nodes()])

def gossip_round(graph, exchange_type=PUSH):
	for node in graph.nodes():	
		if exchange_type == BROADCAST:
			for neighbour in graph.neighbors(node):
				send_message(node, neighbor)
		else:
			random_neighbor = random.choice(graph.neighbors(node))
			if exchange_type == PUSH:
				# Pick a node uniformly at random from the sender's neighbors to receive a message
				send_message(node, random_neighbor)
			elif exchange_type == PULL:
				send_message(random_neighbor, node)
			elif exchange_type == EXCHANGE:
				send_message(random_neighbor, node)
				send_message(node, random_neighbor)

def send_message(sender, receiver):
	message = graph.node[sender]['rlnc'].get_random_message()
	if message is not None:
		graph.node[receiver]['rlnc'].receive(message)

def verify_solution(solution, messages):
	for message in messages:
		assert message in solution

# TODO: count number of messages sent?
# TODO: loop for variance (it is a randomized algorithm, afterall)


"""
Graph generators
"""
def generate_complete_graph(n):
	return nx.complete_graph(n)

def generate_path_graph(n):
	return nx.path_graph(n)

def generate_wheel_graph(n):
	return nx.wheel_graph(n)

def generate_cycle_graph(n):
	return nx.cycle_graph(n)

# TODO: random graphs, dynamic graphs
# TODO: No RLNC, just straight messages (pick a random one)
# TODO: hot rumours (no stopping correctness guarnetees)


			# Edge fault randomly in each round
if __name__ == '__main__':
	stats = RLNCStats()
	for i in range(N_TRIALS):
		# 1) Generate a connected graph
		# TODO: determine graph type
		print "Generating graph"
		graph = generate_complete_graph(NUM_NODES)
		for node in graph.nodes():
			graph.node[node]['rlnc'] = RLNCNode(node)

		# 2) Create the messages
		print "Generating %d messages of length %d in a field of size %d" % (NUM_MESSAGES, MESSAGE_LENGTH, FIELD_SIZE)
		messages = MESSAGE_MATRIX_SPACE.random_element()

		# 3) Distribute the messages
		standard_basis_vectors = matrix.identity(NUM_MESSAGES)
		for i, (standard_basis_vector, message) in enumerate(zip(standard_basis_vectors, messages.rows())):
			# TODO: the starting configuration should determine how things are spread?
			graph.node[0]['rlnc'].receive(RLNCMessage(standard_basis_vector, message))
		
		# 4) Run gossip until can stop
		print "Starting protocol"
		rounds = 0
		while not can_stop_sending(graph):
			gossip_round(graph)
			rounds += 1
			if (rounds % 50 == 0):
				summarize(graph)
			print "It's now round %d" % (rounds)
		print "It took %d rounds" % (rounds)

		# 5) Verify answer
		for node in graph.nodes():
			solution = graph.node[node]['rlnc'].decode()
			verify_solution(solution, messages)
		print "Solution has been verified!"

		# Update stats
		stats.update(rounds)
	stats.summarize()