#PreProcess Data in ../Data.
#Convert each document to a sequence of numbers of type numpy array
#Build The Vocabulary
#extract the metadata into a numpy array

import numpy as np

def PreProcess(corpus,meta_corpus):
	vocab_temp = []

	docs_temp = file(corpus,'r').readlines()
	docs_temp = [d.strip().split(" ") for d in docs_temp]
	
	#Building Vocabulary
	for doc in docs_temp:
		for word in doc:
			vocab_temp.append(word)

	vocab_temp = set(vocab_temp)
	vocab = {}

	for i,v in enumerate(vocab_temp):
		vocab[v] = i

	docs = []
	for doc in docs_temp:
		temp = []
		for word in doc:
			temp.append(vocab[word])
		docs.append(temp)

	#Building MetaData
	rows = len(docs)

	meta_docs = file(meta_corpus,'r').readlines()
	meta_docs = [d.strip().split(" ") for d in meta_docs]

	cols = len(meta_docs[0])

	meta_docs_array = np.zeros((rows,cols),dtype = float)

	for r in range(rows):
		for c in range(cols):
			meta_docs_array[r,c] = float(meta_docs[r][c])

	return vocab, docs, meta_docs_array

#v,d,m = PreProcess("../data/doc.txt","../data/meta.txt")