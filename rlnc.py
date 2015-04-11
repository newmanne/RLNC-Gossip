from sage.all import *
import networkx as nx
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import random
import collections
from collections import defaultdict
import json
import threading

PUSH = "PUSH"
PULL = "PULL"
EXCHANGE = "EXCHANGE" 
BROADCAST = "BROADCAST"

FIELD_SIZE = 2 # MUST BE A PRIME OR PRIME POWER!
MESSAGE_LENGTH = 100
N_TRIALS = 5

class RLNCNode:
	def __init__(self, identifier, gossipper):
		self.messages = np.array([]) # A matrix where every row is a message
		self.coefficients = np.array([]) # A matrix where every row is coefficients for a message
		self.identifier = identifier
		self.gossipper = gossipper
		self.stopping_round = 999999
		self.extra_messages = 0

	def get_num_messages(self):
		return self.gossipper.num_messages

	def get_current_round(self):
		return self.gossipper.round

	def get_matrix_space(self):
		return self.gossipper.matrix_space

	def receive(self, message):
		if np.all(message.coefficients==0) or self.can_decode():
			self.extra_messages += 1
			return # Message is just the 0 vector and hence adds no value or I can already decode

		if self.messages.size == 0: # I know nothing, so this has to add value
			self.coefficients = np.array([message.coefficients])
			self.messages = np.array([message.message])
		else:
			# Add a message if the rank increases (implies new messsage is linearly independent of old ones)
			old_rank = self.rank()
			new_coefficients = np.concatenate((self.coefficients, np.array([message.coefficients])), axis=0)
			new_messages = np.concatenate((self.messages, np.array([message.message])), axis=0)
			new_rank = self.compute_rank(new_coefficients)
			if new_rank > old_rank:
				self.coefficients = new_coefficients
				self.messages = new_messages
			else:
				self.extra_messages += 1

	def mod(self, array):
		return np.mod(array, FIELD_SIZE)

	def rank(self):
		return self.compute_rank(self.coefficients)

	def compute_rank(self, matrix):
		if matrix.size == 0:
			return 0
		MS = MatrixSpace(GF(FIELD_SIZE), matrix.shape[0], matrix.shape[1])
		M = MS(list(matrix.astype(np.int).flatten()))
		return M.rank()

	def can_decode(self):
		can = self.rank() == self.get_num_messages()
		if can:
			self.stopping_round = min(self.stopping_round, self.get_current_round())
		return can

	def get_random_message(self):
		if self.messages.size == 0: # You have nothing to broadcast
			return

		rand_coefficients = np.random.randint(0, FIELD_SIZE, self.messages.shape[0]) # One random coefficient for each message I know about
		message = self.mod(np.dot(rand_coefficients, self.messages)).astype(np.int)
		coefficients = self.mod(np.dot(rand_coefficients, self.coefficients)).astype(np.int)
		return RLNCMessage(coefficients, message)

	def decode(self):
		a = self.coefficients
		b = self.messages
		assert a.shape == (self.get_num_messages(), self.get_num_messages())
		assert b.shape == (self.get_num_messages(), MESSAGE_LENGTH)
		coefficient_matrix_space = MatrixSpace(GF(FIELD_SIZE), self.get_num_messages(), self.get_num_messages())
		ma = coefficient_matrix_space(list(a.flatten()))
		inv = ma.inverse()
		mb = self.get_matrix_space()(list(b.flatten()))
		return inv * mb

class RLNCMessage:
	def __init__(self, coefficients, message):
		self.coefficients = coefficients
		self.message = message

	def __str__(self):
		return "coefficients:" + str(self.coefficients) + "message:" + str(self.message)

class BaseGossipper():
	def __init__(self, graph_generator, num_nodes, num_messages, verify=False):
		self.num_nodes = num_nodes
		self.num_messages = num_messages
		self.round = 0
		self.graph = graph_generator(num_nodes)
		self.verify = verify

		# Generate msesages
		self.matrix_space = MatrixSpace(GF(FIELD_SIZE), num_messages, MESSAGE_LENGTH)
		self.messages = self.matrix_space.random_element()

	def gossip(self, exchange_type=PUSH):
		print "Starting protocol"
		while not self.can_stop_sending():
			gossip_round(self.graph, exchange_type)
			self.round += 1
			if (self.round % 50 == 0):
				self.summarize()
				print "It's now round %d" % (self.round)
		print "It took %d rounds" % (self.round)

		if self.verify:
			for node in self.get_nodes():
				solution = node.decode()
				verify_solution(solution, self.messages)
			print "Solution has been verified!"

	def get_nodes(self):
		for node in self.graph.nodes():
			yield self.graph.node[node]['rlnc']

	def round_info(self):
		summary_info = {
			"stopping_round": self.round
		}
		node_info = {}
		for node in self.get_nodes():
			node_info[node.identifier] = {
				"stopping_round": node.stopping_round,
				"extra_messages": node.extra_messages
			}
		return {
			"summary": summary_info,
			"nodes": node_info
		}

	def can_stop_sending(self):
		return all([node.can_decode() for node in self.get_nodes()])

class RLNCGossipper(BaseGossipper):
	def __init__(self, graph_generator, num_nodes, num_messages, verify=True):
		BaseGossipper.__init__(self, graph_generator, num_nodes, num_messages, verify)

		for node in self.graph.nodes():
			self.graph.node[node]['rlnc'] = RLNCNode(node, self)

		# Assign messages
		standard_basis_vectors = matrix.identity(self.num_messages)
		for i, (standard_basis_vector, message) in enumerate(zip(standard_basis_vectors, self.messages.rows())):
			self.graph.node[0]['rlnc'].receive(RLNCMessage(standard_basis_vector, message))

	def summarize(self):
		for node in self.get_nodes():
			rank = node.rank()
			print "Node %d has rank %d / %d" % (node.identifier, rank, self.num_messages)

class SimpleGossipper(BaseGossipper):
	def __init__(self, graph_generator, num_nodes, num_messages):
		BaseGossipper.__init__(self, graph_generator, num_nodes, num_messages, False)
		for node in self.graph.nodes():
			self.graph.node[node]['rlnc'] = SimpleGossipperNode(node, self)

		self.messsages = map(tuple, self.messages)

		# Assign messages
		for message in self.messages:
			self.graph.node[0]['rlnc'].receive(message)

	def summarize(self):
		pass

class SimpleGossipperNode():
	def __init__(self, identifier, gossipper):
		self.messages = []
		self.identifier = identifier
		self.gossipper = gossipper
		self.stopping_round = 999999
		self.extra_messages = 0

	def receive(self, message):
		if message in self.messages:
			self.extra_messages += 1
		else:
			self.messages.append(message)
			if self.can_decode():
				self.stopping_round = self.get_current_round()

	def get_random_message(self):
		return random.choice(self.messages) if len(self.messages) > 0 else None

	def can_decode(self):
		return len(self.messages) == self.get_num_messages()

	def get_num_messages(self):
		return self.gossipper.num_messages

	def get_current_round(self):
		return self.gossipper.round

def gossip_round(graph, exchange_type):
	for node in graph.nodes():	
		if exchange_type == BROADCAST:
			for neighbour in graph.neighbors(node):
				send_message(graph, node, neighbor)
		else:
			random_neighbor = random.choice(graph.neighbors(node))
			if exchange_type == PUSH:
				# Pick a node uniformly at random from the sender's neighbors to receive a message
				send_message(graph, node, random_neighbor)
			elif exchange_type == PULL:
				send_message(graph, random_neighbor, node)
			elif exchange_type == EXCHANGE:
				send_message(graph, random_neighbor, node)
				send_message(graph, node, random_neighbor)

def send_message(graph, sender, receiver):
	message = graph.node[sender]['rlnc'].get_random_message()
	if message is not None:
		graph.node[receiver]['rlnc'].receive(message)

def verify_solution(solution, messages):
	for message in messages:
		assert message in solution

"""
Graph generators
"""
def generate_complete_graph(n):
	return nx.complete_graph(n)

def generate_path_graph(n):
	return nx.path_graph(n)

# TODO: random graphs, dynamic graphs
# TODO: No RLNC, just straight messages (pick a random one)
# TODO: hot rumours (no stopping correctness guarnetees)
# Edge fault randomly in each round

class WorkerThread(threading.Thread):
	def __init__(self, gossipper, config, lock):
		threading.Thread.__init__(self)
		self.gossipper = gossipper
		self.config = config
		self.lock = lock

	def run(self):
		self.gossipper.gossip()	
		info = self.gossipper.round_info()
		self.lock.acquire()
		self.config["results"].append(info)
		self.lock.release()

if __name__ == '__main__':
	data = []
	for num_messages in [1, 10, 20, 30, 40, 50]:
		for num_nodes in [500]:
			config = {
				"configuration": {
					"num_messages": num_messages,
					"num_nodes": num_nodes,
					"message_length": MESSAGE_LENGTH,
					"exchange_type": "PUSH",
				},
				"results": []
			}
			trials = []
			lock = threading.Lock()
			for _ in range(N_TRIALS):
				gossipper = SimpleGossipper(generate_complete_graph, num_nodes, num_messages)
				# gossipper = RLNCGossipper(generate_complete_graph, num_nodes, num_messages)
				t = WorkerThread(gossipper, config, lock)
				trials.append(t)
				t.start()
			print "jobs launched"
			for t in trials:
				t.join()
			data.append(config)
	with open('results.txt', 'w') as results:
		results.write(json.dumps(data))
	print "goodbye"

# How to do random edge faults? Graph.tick() -> 