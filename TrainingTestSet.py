# This file creates the data set used for training the LDA model; One specifies from which time interval one wants to use the yelp data from; the 
# code also computes lists of all costumer and business ids that occur in the specified time interval. The default start and end dates are the dates
# of the first and last review posted (i.e. the whole data set is used for training), respectively. If only one date is specified, this will be assumed 
# to be the end date; the input format of the dates is '2011-01-01' (including parentheses); also, each output file is labeled according to the specified 
# dates and stored in the Data_Sets_Madison folder

# In addition, the code will output business and costumer profiles as they will be used later on for training the prediction model; descriptions of 
# these profiles are below; also, the code selects users that have many (>15) reviews in the test interval (choosen to be the interval between the last review
# in our data set and the last review in the time interval specified here) and who will be used to assess the predictive power of our method(s); the .json file
# which contains the profiles of these costumers is called 'piglets.json'

# we load required packages
import os, json, string, re, sys
import matplotlib.pyplot as plt
import numpy as np

def main():
	"""
	Computes the training data set for the Madison cohort as specified by up to two dates and stores
	the outputs as .json files in the Data_Sets_Madison folder
	"""
	# change working directory
	directory_default = os.getcwd()
	user = os.environ['HOME']
	directory = user  + '/Dropbox/REVIEWS_CODES/Final_Code/Yelp_Data_Set/Data_Sets_Madison'
	os.chdir(directory)

	# we load the relevant functions from the auxillary.py file
	from auxillary import display_profile

	# load the review data and dictionary
	with open('reviews_Madison_extended.json') as fp: 
		Madison = json.load(fp)   

	# load topic and document distributions from LDA analysis
	topic_word = np.load("topic_word.npy")
	doc_topic = np.load("doc_topic.npy")

	# we compute a list of all review dates	
	dates = [Madison[i]['date'] for i in range(0,len(Madison))]

	# we specify start and end dates
	if (len(sys.argv) == 1):
		time_start =  min(dates)
		time_end = max(dates)
	elif (len(sys.argv) == 2):
		time_start = min(dates)
		time_end = sys.argv[1]
	elif (len(sys.argv) == 3):
			time_start = sys.argv[1]
			time_end = sys.argv[2]

	# we compute the indicies of the reviews that fall between our start and end times and the indices of reviews after the interval
	index_training = [i for i in range(0,len(Madison)) if Madison[i]['date'] > time_start and Madison[i]['date'] < time_end]
	index_test = [i for i in range(0,len(Madison)) if Madison[i]['date'] > time_end]

	# we compute training and test reviews and store them in the Data_Sets_Madison folder
	Madison_training = [Madison[i] for i in index_training]
	with open('reviews_Madison_training' + '_' + time_start + '_' + time_end + '.json', 'w') as outfile:
		json.dump(Madison_training, outfile)
	Madison_test = [Madison[i] for i in index_test]
	with open('reviews_Madison_test' + '_' + time_end + '_' + max(dates) + '.json', 'w') as outfile:
		json.dump(Madison_test, outfile)

	# we compute the costumer ids for all reviews in the training set and store then in the same folder
	costumer_ids = [Madison_training[i]['user_id'] for i in range(0,len(Madison_training))]
	costumer_ids = list(set(costumer_ids))
	with open('costumer_ids_training' + '_' + time_start + '_' + time_end + '.json', 'w') as outfile:
		json.dump(costumer_ids, outfile)

	# we compute costumer profiles for each costumer that makes an occurences in our time interval; a costumer profile consists of
	# its (yelp) user id, a list of all ids of reviews from this particular costumer, the ids of all businesses he/she has reviewed
	# and the topic distribution and the number of reviews
	costumer_profiles = []
	for i in range(0,len(costumer_ids)):
		costumer_profile = {}
		costumer_profile['id'] = costumer_ids[i]
		review_ids = [Madison_training[i]['review_id'] for i in range(0,len(Madison_training)) if Madison_training[i]['user_id'] == costumer_profile['id']]
		business_ids = [Madison_training[i]['business_id'] for i in range(0,len(Madison_training)) if Madison_training[i]['user_id'] == costumer_profile['id']]
		costumer_profile['businesses'] = list(set(business_ids))
		costumer_profile['no.reviews'] = len(review_ids)
		costumer_profile['review_ids'] = review_ids
		costumer_profile['dist'] = display_profile(costumer_profile['id'],display=None)
		costumer_profiles.append(costumer_profile) 

	with open('costumer_profiles_training' + '_' + time_start + '_' + time_end + '.json', 'w') as outfile:
		json.dump(costumer_profiles, outfile)

	# we compute business ids for all reviews in the training set and store them in the same folder; 
	business_ids = [Madison_training[i]['business_id'] for i in range(0,len(Madison_training))]
	business_ids = list(set(business_ids))
	with open('business_ids_training' + '_' + time_start + '_' + time_end + '.json', 'w') as outfile:
		json.dump(business_ids, outfile)

	# we compute business profiles for each business that is reviewed in our time interval; a business profile consists of 
	# its (yelp) business id, a list of all ids and indices of costumers that reviewed it in our time frame, the number and ids of such reviews
	# as well as its average rating
	business_profiles = []
	for i in range(0,len(business_ids)):
		business_profile = {}
		business_profile['id'] = business_ids[i]
		review_ids = [Madison_training[i]['review_id'] for i in range(0,len(Madison_training)) if Madison_training[i]['business_id'] == business_profile['id']]
		review_ratings = [Madison_training[i]['stars'] for i in range(0,len(Madison_training)) if Madison_training[i]['business_id'] == business_profile['id']]
		business_profile['no_reviews'] = len(review_ids)
		business_profile['review_ids'] = review_ids
		business_profile['rating'] = sum(review_ratings) / float(len(review_ids))
		costumers = [Madison_training[i]['user_id'] for i in range(0,len(Madison_training)) if Madison_training[i]['business_id'] == business_profile['id']]
		# business_profile['costumer_indices'] = [i for i in range(0,len(costumer_profiles)) if business_profile['id'] in costumer_profiles[i]['businesses']]
		costumers = list(set(costumers))
		business_profile['costumer_ids'] = costumers
		business_profiles.append(business_profile)

	with open('business_profiles_training' + '_' + time_start + '_' + time_end + '.json', 'w') as outfile:
		json.dump(business_profiles, outfile)

	# we now compute a list of high-activity costumers during the testing period; these will be referred to as piglets and include all indivuals who posted at least
	# 15 reviews in the training period;  a piglet is classified by its costumer id, the number of its total reviews in the training and test intervals, a list of 
	# ids from all business he/she reviewed during testing, as well as the rating he gave in these reviews
	costumer_ids_test = [Madison_test[i]['user_id'] for i in range(0,len(Madison_test))]
	costumer_ids_test = list(set(costumer_ids_test))
	piglets = []
	for i in range(0,len(costumer_ids_test)):
		piglet = {}
		piglet['id'] = costumer_ids_test[i]
		piglet['total_test_reviews'] = len([Madison_test[i]['review_id'] for i in range(0,len(Madison_test)) if Madison_test[i]['user_id'] == piglet['id']])
		piglet['total_training_reviews'] = len([Madison_training[i]['review_id'] for i in range(0,len(Madison_training)) if Madison_training[i]['user_id'] == piglet['id']])
		piglet['test_restaurants'] = [Madison_test[i]['business_id'] for i in range(0,len(Madison_test)) if Madison_test[i]['user_id'] == piglet['id']]
		piglet['test_restaurants_rating'] = [Madison_test[i]['stars'] for i in range(0,len(Madison_test)) if Madison_test[i]['user_id'] == piglet['id']]
		if piglet['total_training_reviews'] > 15:
			piglets.append(piglet)

	with open('piglets' + '_' + time_end + '_' + max(dates) + '.json', 'w') as outfile:
		json.dump(piglets, outfile)

	# change back to default directory
	os.chdir(directory_default)

if __name__ == '__main__':	main()