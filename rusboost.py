"""Rusboost; RUSBoost: A Hybrid Approach to
Alleviating Class Imbalance"""

import numpy as np
#********************************************Previously*******************************************************
# def RUSBOOST(classifier, imbalanced_X_train, imbalanced_Y_train, iterations, percentage_minority):
# 	#1. Initialize D1(i) = 1/m for all i, where i is a given example from the train set.
# 	m = shape(imbalanced_X_train)[0]
# 	D = mat(ones((m,1))/m)
# 	#2. For each iteration:
# 		#2a Create temporary training dataset S with distribution D't using RUS
# 		#***2b Call weak learner providing it with examples S't and their weights D't***
# 		#2c Get back a hypothesis ht
# 		#2d Calculate the pseudo-loss (error)
# 		#2e Calculate the weidght update parameter; ln(error/(1-error))
# 		#2f Update Dt
# 		#2g Normalize Dt+1
# 	#3 Output final hypothesis/classification formula



# def adaBoostTrainDS(dataArr,classLabels,numIt=40):
# 	weakClassArr = []
# 	m = shape(dataArr)[0]
# 	D = mat(ones((m,1))/m)
# 	aggClassEst = mat(zeros((m,1)))
# 	for i in range(numIt):
# 		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
# 		print("D:",D.T)
# 		alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #avoid div0 error with max func.
# 		bestStump['alpha'] = alpha
# 		weakClassArr.append(bestStump)
# 		print("classEst: ",classEst.T)
# 		expon = multiply(-1*alpha*mat(classLabels).T,classEst)
# 		D = multiply(D,exp(expon))
# 		D = D/D.sum()
# 		aggClassEst += alpha*classEst
# 		print("aggClassEst: ",aggClassEst.T)
# 		aggErrors = multiply(sign(aggClassEst) !=
# 		mat(classLabels).T,ones((m,1)))
# 		errorRate = aggErrors.sum()/m
# 		print("total error: ",errorRate,"\n")
# 		if errorRate == 0.0: break
# 	return weakClassArr
#****************************************************************************************************************************
def trainNB0(trainMatrix,trainCategory):
	"""Applied on top of a word2vec implementation and transforms base frequencies
	into word probabilities for each of the 2 classes.

    Returns p0Vect,p1Vect,pAbusive
    p0Vect - probability of class 0 assosciated with each feature (NLP; token word)

    trainCategory should be a list of output values (y_train)

	"""
	
	#Count Number of Documents (should this be -1?....Is 1st array list of words
	#or 1st doc and word labels themselves have been removed?)
	numTrainDocs = len(trainMatrix)  

	#First vector of trainMatrix is the words themselves that form the dictionary.
	numWords = len(trainMatrix[0])
	pAbusive = np.sum(trainCategory)/float(numTrainDocs)
	#Set all probabilities to zero.
	p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
	p0Denom = 0.0; p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = np.log(p1Num/p1Denom) #change to log(); avoid underflow.
	p0Vect = np.log(p0Num/p0Denom) #change to log(); avoid underflow. 

	#Avoid assigning zero probability words to the classifier....even if they do not show up in training sample 
	#they are not impossible for the class...correct this by adding a small probability to each.

	p0Vect = [i if i!=-np.inf else np.log(10**(-4)) for i in p0Vect ]
	p1Vect = [i if i!=-np.inf else np.log(10**(-4)) for i in p1Vect ]


	#Probabilities can later be returned via exponetiation of log(p(x)).
	y_train_pred = []
	counts = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
	for i in range(numTrainDocs):
		document = trainMatrix[i]
		pred_class = classifyNB(document, p0Vect, p1Vect, pAbusive)
		y_train_pred.append(pred_class)

		#tp
		if pred_class ==1 and trainCategory[i] ==1:
			counts['tp'] = counts.get('tp')+1
		#fp
		elif pred_class ==1 and trainCategory[i] ==0:
			counts['fp'] = counts.get('fp')+1

		#tn
		elif pred_class ==0 and trainCategory[i] ==0:
			counts['tn'] = counts.get('tn')+1

		#fn
		elif pred_class ==0 and trainCategory[i] ==1:
			counts['fn'] = counts.get('fn')+1


	precision = counts['tp']/(counts['tp']+counts['fp']) #precision = tp/(tp+fp); What percent of the ones the algorithm classified as positive were correct?
	
	
	recall = counts['tp']/(counts['tp']+counts['fn']) #recall = tp/(tp+fn); What percent of the [actual] positive labels did you correctly identify [recover]?

	accuracy = (counts['tp']+counts['tn'])/numTrainDocs

	#add: confusion matrix

	return p0Vect,p1Vect,pAbusive, precision, recall, accuracy, counts['tp'], counts['fp'], counts['tn'], counts['fn'] #add: confusion matrix


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	#Calculate probability that document came from one class vs the other.
	p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def pred(y_test, p0Vec, p1Vec, pClass1):
    predictions = []
    for document in y_test:
        prediction = classifyNB(document, p0Vec, p1Vec, pClass1)
        predictions.append(prediction)
    return predictions


def RUSBoost_NaiveBayes(imbalanced_X_train, imbalanced_Y_train, iterations, percentage_minority):
	""" Inputs: 
	dataArr - matrix of training examples (X only)
	classLabels - array of labels regarding which class each example belongs to. 
		Should be -1 and 1, 1 being the minority class of interest.
	numIt- the number of iterations (or weak classifiers) adaboost will train (unless errorRate hits 0 1st).
    
    Output:
    Returns a list of NaiveBayes classifiers.
	"""
	weakClassArr = []

	#Number of training examples
	m = len(imbalanced_Y_train)

	#Weight each example should contribute to the overall classifier.
	D = np.mat(ones((m,1))/m)

	#Aggregate class estimate from ensemble for each training example.
	#Arbitrarily, we are setting all initial estimates to the majority class; class 0.
	aggClassEst = np.mat(zeros((m,1)))
	
	
	#Split training data based on class labels in order to later create artificial sets of desired proportion
	train_zip = list(zip(imbalanced_X_train, imbalanced_Y_train))
	pos = [(x,y) for x, y in train_zip  if y==1]
	neg = [(x,y) for x, y in train_zip  if y!=1]


	#User specifies what % of train set should belong to minority class. Here we change that to a ratio of the two classes themselves.
	#For example if the user specifies .5 our ratio would be .5/.5=1. This will later be used for random under sampling RUS.

	maj2min_ratio = percentage_minority/(1-percentage_minority)

	for i in range(numIt):

		#Positive should be the minority class we are looking to identify. We will always use the entirety of this dataset.
		#We will generate a random under sample (RUS) for the majority class to match the desired ratio specified by the user.

		#The weighted vector D will change based on examples correct/incorrect classification from previous renditions.
		#In some sense this is a cv-folds (esque) technique where incorrectly labelled examples will have a higher probability of
		#being included in the next RUS sample s.t. the meta algorithm will continue to train weak learners to account for these
		#discrepencies.

		neg_sample = np.random.choice(neg, size=len(pos)*(maj2min_ratio), replace=True, p=D)


		#Merge the minority class with our RUS majority class sample to form our temp training set.
		#Note: imbalanced_X_train will be a sparse matrix due to the nature of tokenzing language.
		X_train = list([x.toarray() for x,y in pos]) + list(x.toarray() for x,y in neg_sample)
		y_train = list([y for x,y in pos]) + list(y for x,y in neg_sample)


		#Train the weak classifier; in this case via Naive Bayes.
		p0Vect,p1Vect,pClass1, precision, recall, accuracy, tp, fp, tn, fn = trainNB0(X_train, y_train)

		#Calculate the error rate.
		#While we are training on RUS subsets of the entirety of training,
		#error itself should be extrapolate to entirety of training.
		weak_y_pred = pred(imbalanced_Y_train, p0Vect, p1Vect, pClass1)


		#Error = Percent of incorrect predictions from current weak classifier
		error = (np.sum(np.sign(aggClassEst) != np.mat(imbalanced_Y_train)))/m


		# print("D:",D.T)

		#Calculate alpha, the weight this classifier will contribute to the overall ensemble classifier.
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))

		#Create a dictionary with all the relevant features of the weak classifier.
		weakC['p0Vect'] = p0Vect #Note this Vector should change with each weak classifier as the bag of words representation will change with RUS.
		weakC['p1Vect'] = p1Vect #Note this Vector should be the same accross iterations as we are using the full training set of the minority class.
		weakC['p1'] = pClass1 #Note: this should equal the %minority the user specified as we have artificially created samples of the desired proportions.
		weakC['alpha'] = alpha #Weight of classifier contribution to ensemble.

		#Append our weak classifier to our list of classifiers.
		weakClassArr.append(weakC)

		#Update D for next iteration
		expon = np.multiply(-1*alpha*mat(imbalanced_Y_train).T,classEst)
		D = np.multiply(D,np.exp(expon))

		#Normalize the vector (to sum to 1). Other, more complex functions, could be used here.
		#Impact of this would be subtle and most likely negligent but could be an interesting thought to further investigate.
		D = D/D.sum()

		#Update aggregate class estimates given the new weak classifier.
		aggClassEst += alpha*classEst
		# print("aggClassEst: ",aggClassEst.T)
		aggErrors = np.sign(aggClassEst) != np.mat(imbalanced_Y_train)
		errorRate = aggErrors.sum()/m
		print("total error: ",errorRate,"\n")
		if errorRate == 0.0: 
			break
	return weakClassArr