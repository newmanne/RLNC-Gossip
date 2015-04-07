import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy.linalg import lu

PUSH = 0
PULL = 1
EXCHANGE = 2 
BROADCAST = 3

FIELD_SIZE = 2
NUM_MESSAGES = 5

# Graph
# Communication protocol (RLNC)
# Gossip algorithm

class RLNCNode:
	def __init__(self):
		self.coefficients = np.array() 
		self.messages = []

	def receive(self, message):
		# Add the coeffs to your matrix I guess? If you ever have k linearly independent equations...
		pass

"""To show that your S1 and S2 span the same space, you need to show that every linear combination of vectors in S1 is also a linear combination of vectors in S2 and vice versa.
	Therefore, I know that F^k_q has k standard basis vectors. Show that you can write each one as a combo of the vectors in S1
"""

def can_stop_sending():
	return all([can_decode(node) for node in graph.nodes()])

def can_decode(matrix):
	numpy.linalg.matrix_rank(matrix) == NUM_MESSAGES

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
	# 2) Create the messages
	# 3) Distribute the messages
	# 4) Run gossip until can stop