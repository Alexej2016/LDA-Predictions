# This file implements a few auxillary functions that are handy when inspecting the LDA outputs; these functions are also used in some of the scripts for other task (such as
# TrainingTestSet.py). 

# we load required packages
import os, json, string, re
import matplotlib.pyplot as plt
import numpy as np

# change working directory
directory_default = os.getcwd()
user = os.environ['HOME']
directory = user  + '/Dropbox/REVIEWS_CODES/Final_Code/Yelp_Data_Set/Data_Sets_Madison'
os.chdir(directory)

# load the review data and dictionary
with open('reviews_Madison_extended.json') as fp: 
	Madison = json.load(fp)   

with open('dictionary.txt') as f:
	vocab = [line.rstrip('\n') for line in f]

# load topic and document distributions from LDA analysis
topic_word = np.load("topic_word.npy")
doc_topic = np.load("doc_topic.npy")

 # we implement a few functions to visualize the results of the LDA
def display_topic(n,m):
	""" displays the top m words (by probability) of topic n """
	topic_words = np.array(vocab)[np.argsort(topic_word[n])][:-(m+1):-1]
	topic_words_prob = topic_word[n][np.argsort(topic_word[n])[:-(m+1):-1]]
	topic_words_prob = [round(x,3) for x in topic_words_prob]
	print('TOPIC {}: {} '.format(n, zip(topic_words, topic_words_prob)))

def display_topic_dist(n):
	""" selects the nth costumer and displays the topic probabilities for each of his reviews (five reviews at max) """
	id= Madison[n]['user_id']
	set = [i for i in range(0,len(Madison)) if Madison[i]['user_id']==id]
	set = set[0:min(len(set),5)]
	f, ax= plt.subplots(len(set), 1, figsize=(8, 6), sharex=True)
	for i, k in enumerate(set):
		ax[i].stem(doc_topic[k,:], linefmt='r-',
					markerfmt='ro', basefmt='w-')
		ax[i].set_xlim(-1, (len(doc_topic[0])+1))
		ax[i].set_ylim(0, 1)
		ax[i].set_ylabel("Prob")
		ax[i].set_title("Document {}".format(k))
	ax[4].set_xlabel("Topic")
	plt.tight_layout()
	plt.show()

def display_profile(id,display=None):
	""" creates the topic distribution for a given costumer id (i.e. sum of all document distributions for each review the costumer has written, normalized);
		if display is set to True, a bar plot will be returned which displays the distribution """
	index = [i for i in range(0,len(Madison)) if Madison[i]['user_id']==id]
	profile = Madison[index[0]]['loading']
	for i in range(1,len(index)):
		profile = [sum(x) for x in zip(profile, Madison[index[i]]['loading'])]
		profile = [x / float(len(index)) for x in profile]
	if display is True:
		plt.bar(range(0,len(profile)),profile)
		plt.xlabel("Topic")
		plt.ylabel("Frequency")
		plt.show()
	else: 
		return profile