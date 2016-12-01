# This code implements the the classical Latent Dirichlet Allocation (LDA) for the yelp data. It is different from the version written by David Blei et al. in
# HoffmanBleiBach2010b.pdf as it is not an online algorithm (the version here is sometimes referred to as the 'batch' version). The code presented here uses 
# collapsed Gibbs sampling just as the R code version of the code that we have used before; this python version of the algorithm is a lot faster though and 
# more convenient to work with since the rest of the code for the review data is also written in python (so that now everything is written in the same language)

# the code requires a .json file of reviews as well as a dictionary in .txt format. It then processes the review data and brings it into a 'bag of words' format
# (each review now is a nx1 dimensional array with n being the size of the dictionary, its ith entry is the number of times the ith word in the dictionary occurs
# in the review); it then uses the formatted data to compute document and topic loadings;

def setting():
	""" computes 'bag of words' representation of Madison data using the dictionary we specified in dictionary.txt; its 
	output is used for the LDA algorithm """

	# we load required packages
	import os, json, string, re, numpy, lda
	import matplotlib.pyplot as plt

	# change working directory
	directory_default = os.getcwd()
	user = os.environ['HOME']
	directory = user  + '/Dropbox/REVIEWS_CODES/Final_Code/Yelp_Data_Set/Data_Sets_Madison'
	os.chdir(directory)

	# load the review data and dictionary
	with open('reviews_Madison.json') as fp: 
		Madison = json.load(fp)   

	with open('dictionary.txt') as f:
		vocab = [line.rstrip('\n') for line in f]

	# we load the review texts
	docs = []
	for d in Madison:
		docs.append(d['text'])

	# we write a dictionary that specifies which word in the dictionary corresponds to which entry in the review arrays
	vocab_map = {}
	for i in range(0,len(vocab)):
		vocab_map[vocab[i]] = i

	D = len(docs)

	# wordcts is a list of lists, each of which corresponds to a review array; running this for all reviews takes a bit of time (a few minutes); these outputs can
	# be saved to be used when implementing the LDA again at a later point in time.
	wordcts = list()
	for d in range(0, D):
		# we format the review texts to ignore puctuation and capital letters etc.
		docs[d] = docs[d].lower()
		docs[d] = re.sub(r'\n\n', ' ',docs[d])
		docs[d] = re.sub(r'[^a-z ]', ' ', docs[d])
		words = string.split(docs[d])
		cts = []
		for w in vocab:
			n = words.count(w)
			cts.append(n)
		wordcts.append(cts)

	# we save the formatted review data
	Y = numpy.array(wordcts,numpy.int32)
	numpy.save('reviews_cts.npy',Y)

	# change back to original directory
	os.chdir(directory_default)

	return Y

# we can now run the (batch) LDA algorithm on the formatted review data Y; we choose K different topics and set our algorithm to run N_iter passes
# Note: the random_state input specifies the topic prior parameters; we choose the default value of 1; There will be a warning message 
# saying that some zero rows are found; this can be circumvented by allowing a larger dictionary (1000 words might be a bit too little)

def fit_lda(K,N_iter,Y):
	""" runs the batch LDA algorithm on Y as above for K topics and N_ter iterations; it outputs topic loadings, topic_word, and 
	document loadings, doc_topic, as well as merges these loadings with our original Madison review data, and saves them in .json format """

	# we load required packages
	import os, json, string, re, numpy, lda
	import matplotlib.pyplot as plt

	# change working directory
	directory_default = os.getcwd()
	user = os.environ['HOME']
	directory = user  + '/Dropbox/REVIEWS_CODES/Final_Code/Yelp_Data_Set/Data_Sets_Madison'
	os.chdir(directory)

		# load the review data and dictionary
	with open('reviews_Madison.json') as fp: 
		Madison = json.load(fp)  

	model = lda.LDA(n_topics=K,n_iter=N_iter,random_state=1)
	model.fit(Y)

	# we save the LDA topic and document loadings
	topic_word = model.topic_word_
	doc_topic = model.doc_topic_
	numpy.save('topic_word.npy',topic_word)
	numpy.save('doc_topic.npy',doc_topic)

	# we attach the LDA results to the review data and save the output
	for i in range(0,len(Madison)):
		Madison[i]['loading'] = list(doc_topic[i])

	with open('reviews_Madison_extended.json', 'w') as outfile:
		json.dump(Madison, outfile)

	# change back to original directory
	os.chdir(directory_default)


