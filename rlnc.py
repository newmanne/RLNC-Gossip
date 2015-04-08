import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import binascii

PUSH = 0
PULL = 1
EXCHANGE = 2 
BROADCAST = 3

FIELD_SIZE = 10
NUM_NODES = 2

class RLNCNode:
	def __init__(self, identifier):
		self.messages = np.array([]) # A matrix where every row is a message
		self.coefficients = np.array([]) # A matrix where every row is coefficients for a message
		self.identifier = identifier

	def receive(self, message):
		if np.all(message.coefficients==0):
			return # Message is just the 0 vector
		elif self.can_decode():
			return # No need to receive anything

		if self.messages.size == 0:
			print message
			self.coefficients = np.array([message.coefficients])
			self.messages = np.array([message.message])
		else:
			old_rank = self.rank()
			new_coefficients = np.concatenate((self.coefficients, np.array([message.coefficients])), axis=0)
			new_messages = np.concatenate((self.messages, np.array([message.message])), axis=0)
			new_rank = compute_rank(new_coefficients)
			if new_rank > old_rank:
				print message
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
		message = mod(np.dot(rand_coefficients, self.messages))
		coefficients = mod(np.dot(rand_coefficients, self.coefficients))
		return RLNCMessage(coefficients, message)

	def decode(self):
		a = self.coefficients
		b = self.messages
		assert a.shape == (NUM_MESSAGES, NUM_MESSAGES)
		assert b.shape == (NUM_MESSAGES, MESSAGE_LENGTH)
		solution = np.linalg.solve(a, b) #Not allowed to do this because of the whole finite field thing
		solution = mod(solution)
		return solution.astype(np.int)

class RLNCMessage:
	def __init__(self, coefficients, message):
		assert coefficients.shape == (NUM_MESSAGES,)
		assert message.shape == (MESSAGE_LENGTH, )
		self.coefficients = coefficients
		self.message = message

	def __str__(self):
		return "coefficients:" + str(self.coefficients) + "message:" + str(self.message)

def compute_rank(matrix):
	return np.linalg.matrix_rank(matrix)

def summarize(graph):
	for node in graph.nodes():
		rank = graph.node[node]['rlnc'].rank()
		print "Node %d has rank %d / %d" % (node, rank, NUM_MESSAGES)

def mod(array):
	return np.mod(array, FIELD_SIZE)

def can_stop_sending(graph):
	return all([graph.node[node]['rlnc'].can_decode() for node in graph.nodes()])

def generate_complete_graph(n):
	return nx.complete_graph(n)

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

# Send a message from sender to reciever
def send_message(sender, receiver):
	message = graph.node[sender]['rlnc'].get_random_message()
	if message is not None:
		graph.node[receiver]['rlnc'].receive(message)

def verify_solution(solution, messages):
	for message in messages:
		assert message.tolist() in solution.tolist()

# TODO: count number of messages sent?
# TODO: loop for variance

if __name__ == '__main__':
	
	# 1) Generate a connected graph
	print "Generating graph"
	graph = generate_complete_graph(NUM_NODES)
	for node in graph.nodes():
		graph.node[node]['rlnc'] = RLNCNode(node)

	# 2) Create the messages
	messages = [
		np.array([1, 0,] * 5),
		np.array([1] * 10),
		np.array([1] * 9 + [0])
	]
	NUM_MESSAGES = len(messages)
	MESSAGE_LENGTH = len(messages[0])
	for message in messages:
		assert len(message) == MESSAGE_LENGTH

	# 3) Distribute the messages
	standard_basis_vectors = np.identity(NUM_MESSAGES)
	for i, message in enumerate(messages):
		graph.node[0]['rlnc'].receive(RLNCMessage(standard_basis_vectors[i], message))
	
	# 4) Run gossip until can stop
	print "Starting protocol"
	rounds = 0
	while not can_stop_sending(graph):
		gossip_round(graph)
		rounds += 1
		if (rounds % 10 == 0):
			summarize(graph)
		print "It's now round %d" % (rounds)
	print "It took %d rounds" % (rounds)

	# 5) Verify answer
	for node in graph.nodes():
		solution = graph.node[node]['rlnc'].decode()
		print solution
		verify_solution(solution, messages)
