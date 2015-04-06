import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

PUSH = 0
PULL = 1
EXCHANGE = 2 

FIELD_SIZE = 2

class RLNCNode:
	def __init__(self):
		self.coefficients 



def generate_complete_graph(n):
	return nx.complete_graph(n)

def gossip(graph, exchange_type=PUSH):
	# Pick a node uniformly at random to send a message
	sender = random.choice(G.nodes())
	# Pick a node uniformly at random from the sender's neighbours to receive a message
	receiver = random.choice(G.neighbours(sender))
	if exchange_type == PUSH:
		# send a random message

def send_to():
	pass

if __name__ == '__main__':
	# nx.draw(generate_complete_graph(5))
	# plt.show()