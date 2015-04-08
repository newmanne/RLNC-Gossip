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

FIELD_SIZE = 2
NUM_MESSAGES = 5
MESSAGE_LENGTH = 10
NUM_NODES = 10

class RLNCNode:
	def __init__(self):
		self.messages = np.array([0, MESSAGE_LENGTH]) # A matrix where every row is a message
		self.coefficients = np.array([[0, NUM_MESSAGES]]) # A matrix where every row is coefficients for a message

	def receive(self, message):
		np.append(self.coefficients, message.coefficients, axis=0)
		np.append(self.messages, message.message, axis=0)

	def can_decode():
		numpy.linalg.matrix_rank(self.coefficients) == NUM_MESSAGES

	def get_random_message():
		rand_coefficients = np.random.randint(0, FIELD_SIZE, (self.messages.shape[0], 1)) # One random coefficient for each message I know about
		messsage = np.dot(rand_coefficients, self.messages)

		for i, message in enumerate(self.messages):
			m_coefficients[i] = numpy.random.randint(0, FIELD_SIZE)
			m = m + m_coefficients[i] * self.messages[i].message
		return RLNCMessage(coefficients, message)
	def decode():
		pass

class RLNCMessage:
	def __init__(self, coefficients, message):
		self.coefficients = coefficients
		self.message = message

"""To show that your S1 and S2 span the same space, you need to show that every linear combination of vectors in S1 is also a linear combination of vectors in S2 and vice versa.
	Therefore, I know that F^k_q has k standard basis vectors. Show that you can write each one as a combo of the vectors in S1
"""

def can_stop_sending():
	return all([can_decode(node['rlnc']) for node in graph.nodes()])

def generate_complete_graph(n):
	return nx.complete_graph(n)

def gossip_round(graph, exchange_type=PUSH):
	for node in graph.nodes():	
		if exchange_type == BROADCAST:
			for neighbour in graph.neighbours(node):
				send_message(node, neighbour)
		else:
			random_neighbour = random.choice(graph.neighbours(node))
			if exchange_type == PUSH:
				# Pick a node uniformly at random from the sender's neighbours to receive a message
				send_message(node, random_neighbour)
			elif exchange_type == PULL:
				send_message(random_neighbour, node)
			elif exchange_type == EXCHANGE:
				send_message(random_neighbour, node)
				send_message(node, random_neighbour)

# Send a message from sender to reciever
def send_message(sender, receiver):
	message = choose_message_to_send(sender)
	# send it, whatever that means
	receiver.receive(message)

# Pick something uniformly from the messages that you span: i.e. take random coefficients and lin comb your messages?
def choose_message_to_send(sender):
	pass

if __name__ == '__main__':
	# 1) Generate a connected graph
	graph = generate_complete_graph(NUM_NODES)
	for node in graph.nodes():
		node['rlnc'] = RLNCNode()
	# 2) Create the messages
	messages = [
		np.array([1, 0, 1, 0, 1, 0])
	]
	# 3) Distribute the messages
	# 4) Run gossip until can stop