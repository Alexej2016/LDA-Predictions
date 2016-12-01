# This file computes a dictionary of words for the set of reviews of a given yelp area as it is used in the LDA analysis. The dictionary is specified to contain the x 
# most common words in the reviews of a given area (x is currently set to 1,000); the code also computes the list of all words that occur in negative reviews
# (negative defined as a star rating of less than 3); the words in these lists are ordered by frequency of occurence. Between lines 111 and 123 a method to decide 
# exploratorily how many words to include is outlined. The final dictionary is saved as dictionary.txt

### In addition, the code computes the Monroe score (see http://pan.oxfordjournals.org/content/16/4/372 by B. Monroe, at al. (2008)) for all words which are associated 
# with negative reviews, as outlined in http://firstmonday.org/ojs/index.php/fm/article/view/4944/3863 by D. Jurafsky et al. (2014); these words as well as well as their
# Monroe scores are saved as monroe.csv

# We load the yelp data and store the reviews

# we load the required packages
import os, json, nltk, csv
from nltk.tokenize import RegexpTokenizer
from nltk.probability import *
from collections import Counter
from nltk.corpus import stopwords
import math
import matplotlib.pyplot as plt

# change working directory
directory_default = os.getcwd()
user = os.environ['HOME']
directory = user  + '/Dropbox/REVIEWS_CODES/Final_Code/Yelp_Data_Set/Data_Sets_Madison'
os.chdir(directory)

# we define the tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# load the review data
with open('reviews_Madison.json') as fp: 
	Madison = json.load(fp)   

# we will create one string containing every work in all the reviews (text_full), and one that contains all the words
# which show up in bad reviews (text_bad) with bad defined as having a rating of 2 or lower

# define arrays to hold results
review_list = [[],[]]
text_full = ''
text_bad = ''

# we loop over the entire data set to get all the (bad) reviews in one list, as well as strings containing them
for i in range(len(Madison)):
	review_list[0].append(Madison[i]['stars'])
	review_list[1].append(Madison[i]['text'])
	text_full = text_full + ' ' + Madison[i]['text']
	if Madison[i]['stars'] <= 2:
		text_bad = text_bad + ' ' + Madison[i]['text']

# we list all bad reviews (review_list[1] is the list of all reviews)
review_bad = [review_list[1][i] for i in range(len(review_list[1])) if review_list[0][i] <= 2]

# we now edit the strings to make words for comparable

# we set each word in the review strings to be lower case
text_full = text_full.lower()
text_bad = text_bad.lower()

# we count words for both the set of all reviews and the set of all bad reviews; the results are stored in list format 
# (note: the elements of these lists are NOT strings, but rather pairs of strings (words) and integers (their frequencies))
text_word_full = tokenizer.tokenize(text_full)
text_word_bad = tokenizer.tokenize(text_bad)

# we remove stopwords from our lists
text_word_full = [i for i in text_word_full if not i in stopwords.words('english')]
text_word_bad = [i for i in text_word_bad if not i in stopwords.words('english')]

# we tokenize the list of words that should up in every review and all bad reviews, respectively; this will add a feature to each word that indicates
# whether it is e.g. a verb or a noun, or plural etc.
text_word_full_token = nltk.pos_tag(text_word_full)
text_word_bad_token = nltk.pos_tag(text_word_bad)

# we reduce words in plural to their singular by deleting an 's' at the end of the word, if it is in plural
for i in range(len(text_word_full)):
	if (text_word_full_token[i][1] == 'NNS' and text_word_full[i][len(text_word_full[i])-1]=='s'):
		text_word_full[i] = text_word_full[i][:-1]

for i in range(len(text_word_bad)):
	if (text_word_bad_token[i][1] == 'NNS' and text_word_bad[i][len(text_word_bad[i])-1]=='s'):
		text_word_bad[i] = text_word_bad[i][:-1]

# We now create lists of the most frequent words in all our reviews and all bad reviews; these lists will serve as the dictionaries
# for the LDA analysis; we choose the 10000 most frequent words; alternatively, we could restrain our dictionary to only contain words
# which show up at least a certain number of times

# we compute each word's frequency in bad reviews and the total of all reviews
fd_full = nltk.FreqDist(text_word_full)
fd_full.count = Counter(fd_full)

fd_bad = nltk.FreqDist(text_word_bad)
fd_bad.count = Counter(fd_bad)

# We store the frequencies and the words in all reviews; full_count is a list of the frequencies of each word that shows up in the reviews;
# full_word is the corresponding list of words; full_word1 is a list of all words in the reviews that show up at least twice
full_count = [fd_full.count.most_common()[i][1] for i in range(len(fd_full.count.most_common()))]
full_word1 = [fd_full.count.most_common()[i][0] for i in range(len(fd_full.count.most_common())) if fd_full.count.most_common()[i][1]>1]
full_word = [fd_full.count.most_common()[i][0] for i in range(len(fd_full.count.most_common()))]

# We store the frequencies and the words in bad reviews; bad_count is a list of the frequencies of each word that shows up in bad reviews;
# bad_word is the corresponding list of words; bad_word1 is a list of all words in bad reviews that show up at least twice
bad_count = [fd_bad.count.most_common()[i][1] for i in range(len(fd_bad.count.most_common()))]
bad_word1 = [fd_bad.count.most_common()[i][0] for i in range(len(fd_bad.count.most_common())) if fd_bad.count.most_common()[i][1]>1]
bad_word = [fd_bad.count.most_common()[i][0] for i in range(len(fd_bad.count.most_common()))]

# we compute the restricted dictionary (to the 1,000 most frequent ones)
restricted_dict = [fd_full.count.most_common()[i][0] for i in range(1000)]
text_word_full_restricted = [i for i in text_word_full if i in restricted_dict]

# we list all the words among the 10,000 most frequent words that occur in bad reviews
text_word_bad_restricted = [i for i in text_word_bad if i in restricted_dict]

# alternatively, we can decide how many words to include in the dictionary by computing the cumulative proportion of the most frequent
# words relative to the total size of the reviews. Plotting the cumulative proportion against the index (in full_count) will give a
# strictly increasing plot. If we then pick the smallest index, say n, for which the slope of this plot is close to zero, then then only using 
# words 1 to n from full_count will give us a dictionary that contains the most relevant words (also, using words that appear in most reviews will
# improve the the LDA analysis)
#cumulative_full = []
#for i in range(len(full_count)):
#	cumulative_full.append(sum(full_count[0:i]))
#
#plt.plot(cumulative_full)
#plt.ylabel('total number of occurences')
#plt.xlabel('index')
#plt.show()

# We will now compute the Monroe score (log odds ratio ratio) on the restricted dictionary; see D. Jurafsky et al. (2014), Section 2

# we update our frequency counts:
text_word_full_restricted = [i for i in text_word_full if i in restricted_dict]
text_word_bad_restricted = [i for i in text_word_bad if i in restricted_dict]

# we update our frequency counts:
fd_full_restricted = nltk.FreqDist(text_word_full_restricted)
fd_bad_restricted = nltk.FreqDist(text_word_bad_restricted)

fd_bad_restricted.count = Counter(fd_bad_restricted)
bad_word_restricted= [fd_bad_restricted.count.most_common()[i][0] for i in range(len(fd_bad_restricted.count.most_common()))]

a_0 = len(text_word_full_restricted)
n_b = len(text_word_bad_restricted)

# we define a function to compute the Monroe score for a word i; here
# a_0 is the size of the corpus of all reviews, n_b the size of the corpus of bad reviews,
# y_i is the frequency of word i in the corpus of all reviews, yb_i the frequency in the corpus of bad reviews;
def monroe(i):
	"""
	Monroe statistic for word i
	"""
	y_i = fd_full_restricted[i]
	yb_i = fd_bad_restricted[i]
	a_i = y_i
	#
	A_10 = float(yb_i + a_i) 
	A_11 = float(n_b + a_0  - A_10)
	A_20 = float(y_i + a_i) 
	A_21 = float(a_0 + a_0  - A_20)
	#
	m = math.log(A_10/A_11) - math.log(A_20/A_21)
	#
	return(m)

# we compute the Monroe score for each word that occurs in the corpus of bad reviews and store them in a dictionary
monroe_scores = {}

for i in bad_word_restricted:
	c = monroe(i)
	monroe_scores[i] = c

# we also compute a dictionary of all words which have a positive Monroe score; these are the words that are more associated with bad reviews
monroe_bad = {}

for i in bad_word_restricted:
	c = monroe(i)
	if c>0:
		monroe_bad[i] = c

# we save our new dictonary with the dictionary of words associated with bad reviews:
os.chdir(directory)
with open('dictionary.txt','w') as f:
	for s in restricted_dict:
		f.write(s.encode('utf8') + '\n')

writer = csv.writer(open('monroe.csv','wb'))
for key, value in monroe_bad.items():
	s = [key.encode('utf8'),value]
	writer.writerow(s)
