#!/usr/bin/env python
# coding: utf-8

# '''
#   This program shell reads phrase data for the kaggle phrase sentiment classification problem.
#   The input to the program is the path to the kaggle directory "corpus" and a limit number.
#   The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
#   It creates a "phrasedocs" variable with a list of phrases consisting of a pair
#     with the list of tokenized words from the phrase and the label number from 1 to 4
#   It prints a few example phrases.
#   In comments, it is shown how to get word lists from the two sentiment lexicons:
#       subjectivity and LIWC, if you want to use them in your features
#   Your task is to generate features sets and train and test a classifier.
# 
#   This version uses cross-validation with the Naive Bayes classifier in NLTK.
#   It computes the evaluation measures of precision, recall and F1 measure for each fold.
#   It also averages across folds and across labels.
# '''
# # open python and nltk packages needed for processing
# import os
# import sys
# import random
# import nltk
# from nltk.corpus import stopwords
# from nltk.collocations import *
# bigram_measures = nltk.collocations.BigramAssocMeasures()

# In[8]:


## define feature definition functions
# this function define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'V_(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

#list of negation words to create a negation feature:
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor', 'poor', 'dull','ugly', 'worst', 'waste','brutal', 'blood', 'bloody', 'murder', 'revenge','horror', 'sad', 'cry','unhappy', 'noise', 'sick','repeatedly', 'repeat', 'predictable', 'nonsense','underrated']


# Creating the NOT features function
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features

#Defining the POS features function
def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


# In[9]:


## cross-validation ##
# this function takes the number of folds, the feature sets and the labels
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the performance for each fold and the average performance at the end
def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels

    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # computes evaluation measures for this fold and
        #   returns list of measures for each label
        print('Fold', i)
        (precision_list, recall_list, F1_list)                   = eval_measures(goldlist, predictedlist, labels)
        # take off triple string to print precision, recall and F1 for each fold
        '''
        print('\tPrecision\tRecall\t\tF1')
        # print measures for each label
        for i, lab in enumerate(labels):
            print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
              "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        '''
        # for each label add to the sums in the total lists
        for i in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[i] += precision_list[i]
            total_recall_list[i] += recall_list[i]
            total_F1_list[i] += F1_list[i]

    # find precision, recall and F measure averaged over all rounds for all labels
    # compute averages from the totals lists
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]
    # the evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),           "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels),           "{:10.3f}".format(sum(recall_list)/num_labels),           "{:10.3f}".format(sum(F1_list)/num_labels))

    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first intialize a dictionary for label counts and then count them
    label_counts = {}
    for lab in labels:
      label_counts[lab] = 0 
    # count the labels
    for (doc, lab) in featuresets:
      label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision),       "{:10.3f}".format(recall), "{:10.3f}".format(F1))


# In[10]:


# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output: returns lists of precision, recall and F1 for each label
#      (for computing averages across folds and labels)
def eval_measures(gold, predicted, labels):
    
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        if (TP == 0) or (FP == 0) or (FN == 0):
          recall_list.append (0)
          precision_list.append (0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)

## function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  dirPath0 = os.getcwd()
  dirPath = dirPath0+'/corpus'
  print('dirPath in processkaggle function: ', dirPath)
  
  f = open('./corpus/train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  # add all the phrases

  # each phrase has a list of tokens and the sentiment label (from 0 to 4)
  ### bin to only 3 categories for better performance
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))

  # possibly filter tokens
  # lowercase - each phrase is a pair consisting of a token list and a label
  docs = []
  for phrase in phrasedocs:
    lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
    docs.append (lowerphrase)
  # print a few
  for phrase in docs[:100]:
    print (phrase)

  # continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  print(len(all_words))
    

  # get the 5000 most frequently appearing keywords in the corpus
  word_items = all_words.most_common(100)
  word_features = [word for (word,count) in word_items]

  stopwords = nltk.corpus.stopwords.words('english')

  # remove some negation words 
  negationwords.extend(['ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'pathetic','disgusting','boring','predictable', 'confusing', 'nonsense','waste', 'worthless', 'weak', 'outdated', 'unimpressive', 'blood', 'bloody', 'violence', 'unhappy', 'cry', 'murder', 'revenge', 'pain', 'plain', 'scary', 'poor', 'ugly', 'dirty', 'aggrogance', 'distress', 'stressful','weak', 'noisy', 'nasty','offend','offensive'])

  newstopwords = [word for word in stopwords if word not in negationwords]
  # remove stop words from the all words list
  new_all_words_list = [word for (sent,cat) in docs for word in sent if word not in newstopwords]

  # continue to define a new all words dictionary, get the 100 most common as new_word_features
  new_all_words = nltk.FreqDist(new_all_words_list)
  new_word_items = new_all_words.most_common(100)

  new_word_features = [word for (word,count) in new_word_items]

  #Creating Bigram features
  finder = BigramCollocationFinder.from_words(all_words_list)
  # define the top 200 bigrams using the chi squared measure
  bigram_features = finder.nbest(bigram_measures.chi_sq, 200)

  def bigram_document_features(document, word_features, bigram_features):
      document_words = set(document)
      document_bigrams = nltk.bigrams(document)
      features = {}
      for word in word_features:
          features['V_{}'.format(word)] = (word in document_words)
      for bigram in bigram_features:
          features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
      return features

  # feature sets from feature definition functions above
  featuresets = [(document_features(d, word_features), c) for (d, c) in docs]
  negfeaturesets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in docs]
  bigramfeaturesets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in docs]
  POSfeaturesets = [(POS_features(d, new_word_features), c) for (d, c) in docs]
  # train classifier and show performance in cross-validation
  # make a list of labels
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5

  #printing cross validation results for every feature and combination feature 
  #Unigram or Bag-of-Words (BOW) features; this is the baseline:
  print("\nOriginal Featureset")
  cross_validation_PRF(num_folds, featuresets, labels)
  #Bigrams
  print("\nBigrams Featureset")
  cross_validation_PRF(num_folds, bigramfeaturesets, labels)
  #Negation -- SHOULD BE CREATED
  print("\nNegated Featureset")
  cross_validation_PRF(num_folds, negfeaturesets, labels)
  #POS features -- SHOULD BE CREATED
  print("\nPOS Featureset")
  cross_validation_PRF(num_folds, POSfeaturesets, labels)


# In[11]:


"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
dirPath0 = os.getcwd()
dirPath = dirPath0+'/corpus'
print('dirpath in program execution code (last line in the code): ', dirPath)
processkaggle(dirPath, 5000)


# In[ ]:





# In[ ]:





# In[ ]:




