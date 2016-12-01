# This file uses the LDA loadings to predict future costumer ratings; each costumer is associated with a probability distribution over all topics
# based on his/her review history these distributions (and thereby costumers) are then compared using the Jensen-Shannon distance/metric. The
# reviews for a particular business are then reweighted for a given costumer, according to this metric (reviews by people with closer, more similar topic distributions
# receive more weight);

# NOTE: At this point, the improvement of using this method over just using the average rating of a restaurant is marginal at best; depending on what time interval is used
# as training, the improvement is by +/- 0.005 stars on average (sometimes the average rating is actually more precise); however, this method can probably be improved
# substaintially by optimizing over a range of input parameters; most prominently the power coefficient used in reweighting the reviews (currently set to 1) as well 
# as the number of topics used in or LDA could be varied and lead to improvements (experimentation shows that this is likely to make a difference)

# import required packages
import os, json, math

# change working directory
directory_default = os.getcwd()
user = os.environ['HOME']
directory = user  + '/Dropbox/REVIEWS_CODES/Final_Code/Yelp_Data_Set/Data_Sets_Madison'
os.chdir(directory)

# we load the training data set as well as the business and costumer profiles we obtained for that set as well as the set of test costumers (piglets)
with open('reviews_Madison_training_2005-03-03_2013-07-16.json') as fp:
	data_training = json.load(fp)

with open('costumer_profiles_training_2005-03-03_2013-07-16.json') as fp:
	costumer_profiles_training = json.load(fp)

with open('costumer_ids_training_2005-03-03_2013-07-16.json') as fp:
	costumer_ids_training = json.load(fp)

with open('business_profiles_training_2005-03-03_2013-07-16.json') as fp:
	business_profiles_training = json.load(fp)

with open('business_ids_training_2005-03-03_2013-07-16.json') as fp:
	business_ids_training = json.load(fp)

with open('piglets_2013-07-16_2014-07-16.json') as fp:
	piglets = json.load(fp)

# we define classes for businesses and costumers based on their profiles
class Business:
	def __init__(self,i):
		self.id = business_profiles_training[i]['id']
		self.costumers = business_profiles_training[i]['costumer_ids']
		self.rating = business_profiles_training[i]['rating']
		self.no_reviews = business_profiles_training[i]['no_reviews']
		self.review_ids = business_profiles_training[i]['review_ids']

class Costumer:
	def __init__(self,i):
		self.id = costumer_profiles_training[i]['id']
		self.no_reviews = costumer_profiles_training[i]['no.reviews']
		self.dist = costumer_profiles_training[i]['dist']
		self.review_ids = costumer_profiles_training[i]['review_ids']
		self.businesses = costumer_profiles_training[i]['businesses']
		self.index = i

# we define functions that allow us to work with classes either using their ids or indices in the training set
def costumer_map(id):
	""" takes a costumer id and returns the index of the corresponding profile in the costumer training set """
	for i in range(0,len(costumer_profiles_training)):
		if costumer_profiles_training[i]['id'] == id:
			return i

def costumer_map_inverse(n):
	""" inverse of costumer_map """
	return costumer_profiles_training[n]['id']

def business_map(id):
	""" takes a business id and returns the index of the corresponding profile in the business training set """
	for i in range(0,len(business_profiles_training)):
		if business_profiles_training[i]['id'] == id:
			return i

def business_map_inverse(n):
	""" inverse of business_map """
	return business_profiles_training[n]['id']

# we now define the functions required for the prediction
def rating(user,business):
	""" computes the average rating that costumer "user" has given a restaurant "business" """
	which = [ob for ob in data_training if (ob['user_id']==user.id and ob['business_id'] == business.id)]
	if len(which) >0:
		stars = [w['stars'] for w in which]
		return (sum(stars)/float(len(stars)))
	else:
		return('NA')

def KL(P,Q):
	""" Kullbach-Leibler divergence between distributions P and Q """
	S = 0
	for i in range(len(P)):
		if (Q[i]==0 and P[i]==0):
			S = S + 0
		elif (Q[i]==0 and P[i]>0):
			return('Not Defined')
		elif (Q[i]>0 and P[i]==0):
			S = S + 0
		else:
			s = float(P[i])/Q[i]
			S = S + P[i] * math.log(s)
	return(S)

def JS(x1,x2):
	""" Jensen-Shannon distance between two costumer classes x1 and x2 """
	M = [x1.dist[i] + x2.dist[i] for i in range(len(x1.dist))]
	M = [x/float(2) for x in M]
	S = KL(x1.dist,M) + KL(x2.dist,M)
	return(float(S)/2)

def pred_rating(user,business,p):
	""" computes the JS metric between a user and all costumer that reviewed business and returns the reweighted
	 average rating of their reviews as the prediction for this user; the weightings can be varied using power p """
	weights = [Costumer(costumer_map(x)) for x in business.costumers]
	weights = [pow((1-JS(user,x)),p) for x in weights]
	ww = sum(weights)
	weights = [x/float(ww) for x in weights]
	ratings = [rating(Costumer(costumer_map(x)),business) for x in business.costumers]
	pred = [ratings[i] * weights[i] for i in range(len(weights))]
	return(sum(pred))

# we implement functions that quickly give predictions for each of our piglets
# NOTE: so far, every business a piglet reviewed in the testing interval (and which occurs in the training interval) is considered; 
# some of these have very few reviews, so that it seems that, if only businesses with a 'large' number of reviews are considered, the predictions could be improved

def piglet_prediction(n,p=1):
	""" computes tuples containing the predicted, average rating and true rating (that is what the piglet's actual score 
		for a restaurant was) for each business he/she reviewed; power in the JS metric can be specified """
		pig = Costumer(costumer_map(piglets[n]['id']))
		businesses = [Business(business_map(x)) for x in piglets[n]['test_restaurants'] if x in business_ids_training]
		average_rating = [business.rating for business in businesses]
		true_ratings = [piglets[n]['test_restaurants_rating'][i] for i in range(0,len(piglets[n]['test_restaurants'])) if piglets[n]['test_restaurants'][i] in business_ids_training]
		predictions = [pred_rating(pig,business,p) for business in businesses]
		return zip(predictions,average_rating,true_ratings)

def piglet_performance(j,p=1):
	""" returns the total differences in rating between the predicted rating and the actual rating and the average rating and the actual rating for piglet j
	over all businesses he reviewed, as well as the number of restaurants he reviewed, n; JS power p can be specified """
	measures = piglet_prediction(j,p)
	n = len(measures)
	s_prediction = sum([abs(measures[i][2]-measures[i][0]) for i in range(0,n)])
	s_average = sum([abs(measures[i][2]-measures[i][1]) for i in range(0,n)])
	return [s_prediction,s_average,n]

def piglet_performance_total(p):
	""" evaluates the average peformance of our prediction method over all piglets and compares them with the average performance of simply taking the average rating """
	N = 0
	S_prediction = 0
	S_average = 0
	for i in range(0,len(piglets)):
		[s_prediction,s_average,n] = piglet_performance(i,p)
		N = N + n
		S_prediction = S_prediction + s_prediction
		S_average = S_average + s_average
		print i
	return [S_prediction/float(N),S_average/float(N),N]