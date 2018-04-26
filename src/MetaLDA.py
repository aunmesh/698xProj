import numpy as np


class MetaLDA:

	'''
	Constructor for MetaLDA:
	mu = Mean of Topic-Metadata correlation matrix
	sigma_sq = Variance of the matrix
	lambda = Topic-Metadata correlation matrix
	beta = (hyperparameter) prior on pi
	num_topics = number of topics in the model
	pi = multinoulli over vocabulary
	vocab = Dictionary containing vocabulary words
	'''
	def __init__(self, mu, sigma_sq, beta, num_topics, vocab, lambda = None, pi = None):

		self.mu = mu
		self.sigma_sq = sigma_sq
		self.beta = beta
		self.num_topics = num_topics
		self.vocab = vocab
		self.vocab_size = len(vocab)

		if lambda == None:
			self.lambda =  self.SampleLambda(mu, sigma)
		else:
			self.lambda = lambda

		if pi == None:
			self.pi =  self.SamplePi(self.beta, self.vocab_size)
		else:
			self.pi = pi


	'''
	performs inference over an minibatch and updates posterior sample and state for self.lambda and self.pi.
	self.lamda is updated using SGLD
	self.pi is updated using SGRLD
	Args:
		minibatch : minibatch of documents and meta data

	'''
	def inference(self, minibatch):

		#Sample z using Gibbs Sampling
		#Infer Lambda
		#Infer theta
		#Store the inferences in class variable