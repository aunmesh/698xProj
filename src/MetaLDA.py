import numpy as np
from utils import *

class MetaLDA:

	'''
	Constructor for MetaLDA:
	mu = Mean of Topic-Metadata correlation matrix
	sigma_sq = Variance of the matrix
	lamda = Topic-Metadata correlation matrix
	beta = (hyperparameter) prior on pi
	num_topics = number of topics in the model
	pi = multinoulli over vocabulary
	vocab = Dictionary containing vocabulary words
	'''
	def __init__(self, mu, sigma_sq, beta, num_topics, vocab, lamda = None, pi = None):

		#Store state variables in a dict.
		self.mu = mu
		self.sigma_sq = sigma_sq
		self.beta = beta
		self.num_topics = num_topics
		self.vocab = vocab
		self.vocab_size = len(vocab)

		if lamda == None:
			self.lamda =  self.Samplelamda(mu, sigma)
		else:
			self.lamda = lamda

		if pi == None:
			self.pi =  self.SamplePi(self.beta, self.vocab_size)
		else:
			self.pi = pi

		self.state = {}
		self.current_iteration = 0
		self.current_state = {mu:}


	'''
	performs inference over an minibatch and updates posterior sample and state for self.lamda and self.pi.
	self.lamda is updated using SGLD
	self.pi is updated using SGRLD
	Args:
		minibatch : minibatch of documents and metadata [Dictionary of arrays]

	'''
	def inference(self, minibatch):

		#Sample z using Gibbs Sampling - needs Alpha, current z allocations for each word(to calculate ndk.), Current Theta
		#Infer lamda using SGLD - To infer lamda need current Learning Rate , grad of log prior of lamda, grad of log likelihood of data, and a gaussian noise
		#Infer theta using SGRLD - Needs A sample of large enough size to calculate expectation wrt to z. Needs beta , previous value of theta.
		#Store the inferences in class variable

		# As a minibatch comes in we need to calculate Alpha and initialize z for that mini-batch. A good initialisation will be important.
		#For that sampling from Alpha will be required using mixing proportion.

		#Once we have an initial z, we can calculate ndk./i and sample from posterior z.

		#

		z_sample = Gibbs_sampler(minibatch, )


