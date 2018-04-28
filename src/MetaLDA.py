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
	def __init__(self, mu, sigma_sq, beta, num_topics, vocab, dataset_size ,init_lr = 1.0, lamda = None, pi = None):

		#Store state variables in a dict.
		self.mu = mu
		self.sigma_sq = sigma_sq
		self.beta = beta
		self.num_topics = num_topics
		self.vocab = vocab
		self.vocab_size = len(vocab)
		self.d = dataset_size
		self.lr = init_lr
		self.inference_step = 0

		if lamda == None:
			self.lamda =  self.Samplelamda(mu, sigma)
		else:
			self.lamda = lamda

		if pi == None:
			self.pi =  self.SamplePi(self.beta, self.vocab_size)
		else:
			self.pi = pi

		
		self.current_iteration = 0

		self.current_state = dict(lamda = self.lamda, pi = self.pi)

		self.state = []
		self.state.append(self.current_state)

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

		#Update Learning Rate for SGLD and SGRLD optimizers
		
		z_sample = Gibbs_sampler(minibatch, self.current_state["lamda"], self.current_state["theta"])

		theta_new = self.update_theta(z_sample)
		lamda_new = self.update_lamda(z_sample)
		pi_new = self.convert_theta2pi(theta_new)


		#Updates Learning Rate as well as inference_step number
		self.update_lr()

		self.current_state["lamda"] = lamda_new
		self.current_state["pi"] = pi_new
		self.state.append(self.current_state)


	'''
	args:
	z is a numpy aray of size nd
	output- scalar ndk.
	'''
	def calc_ndk_dot(self,z,topic_id):
		res = np.sum(z == topic_id)
		return res

	'''
	res outputs ndk vector , of size V(vocab_size). Num_of times each word in vocab has appeared in the doc from topic_id
	'''
	def calc_ndk(self, z, doc, topic_id):
		res = np.zeros(self.vocab_size)
		
		for pos,word in enumerate(doc):
			if(z[pos] == topic_id):
				res[int(word)]+=1

		return res



	'''
	z_doc : All samples of z for a document. Size: nd
	Output Size: V dimensional vector
	'''
	def calc_expectation_theta_doc( self, doc, z_doc , topic_id):
		res = np.zeros(self.vocab_size)
		for z in z_doc:
			ndk = calc_ndk(z, doc, topic_id)
			ndk_dot = calc_ndk_dot(z , topic_id)
			temp = ndk - ndk_dot * self.current_state['pi']


	'''
	Output size: V dimensional vector 
	'''
	def calc_expectation_sum_theta(self,z_sample, minibatch, topic_id):
		data = minibatch['data']

		for ind,doc in enumerate(data):
			calc_expectation_theta_doc( data[ind], z_sample[ind] , topic_id)


	'''
	minibatch - 
	Output: - Updated Theta . Dimension - T * V
	z-sample - size : list of Batch_size. Each element of list is an array of size : nd, nd is number of words in that document
	'''
	def update_theta(self, minibatch, z_sample):
		#Calculate expectation term first
		expectation





	def update_lamda(self, z_sample):
