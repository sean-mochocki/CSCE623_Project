#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

import itertools
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
from sortedcontainers import SortedDict
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

fig_path = '/root/CSCE623_Project/figures'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    updated_classes = []
    x_labels =[]
    for idx, x in enumerate(classes):
        update_string = x
        update_string = update_string + ': '
        update_string = update_string + str(idx)
        updated_classes.append(update_string)
        x_labels.append(str(idx))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.xticks(tick_marks, x_labels, fontsize = 8)
   # plt.xticks(tick_marks, classes, rotation=45, fontsize=2)
    plt.yticks(tick_marks, updated_classes, fontsize=8)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize='x-small',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{fig_path}/{title}.png')

# def confusion_matrix(y_data, predicted_data):
#     confusion_matrix = metrics.confusion_matrix(y_validation, predicted)
#     df_cm = pd.DataFrame(confusion_matrix, range(20), range(20))
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 7}) # font size
#     plt.show()

# Note - should remove headers footers and quotes, as machine learning algorithms tend to focus on these
#https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
# twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)
# twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True)

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

# categories will be used for confusion matrix
categories = twenty_train.target_names

labels = ['1: atheism', '2: christian', '3: space', '4: politics', '5: religion', '6: autos', '7: windows', '8: mideast', '9: crypt', '10: motocycles', '11: graphics', '12: ibm-hardware',
          '13: hardware', '14: electronics', '15: forsale', '16: sci.med', '17: windows.', '18: baseball', '19: gun', '20: hockey']

# The proportions of values in the training data set is as follows. No data set is underrepresented
# 0: 480, 1: 584, 2: 591, 3: 590, 4: 578, 
# 5: 593, 6: 585, 7: 594, 8: 598, 9: 597, 
# 10: 600, 11: 595, 12: 591, 13: 594, 14: 593, 
# 15: 599, 16: 546, 17: 564, 18: 465, 19: 377,

#Split up the training data into validation and training data
X = twenty_train.data
y = twenty_train.target

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state = 42)
count_dict = dict(Counter(y_train).items())
# print("Training data: ")
# print(SortedDict(count_dict))

count_dict_val = dict(Counter(y_validation).items())
# print("Validation data: ")
# print(SortedDict(count_dict_val))

# print("Test data: ")
# print(SortedDict(Counter(twenty_test.target).items()))

# print ("There are ", len(X_train), " measurements in the training database")
# print ("There are ", len(X_validation), " measurements in the validation database")
# The proportions of values in the training set once validation data is split with random_state = 42 is as follows:
# 0: 383, 1: 480, 2: 476, 3: 467, 4: 452,
# 5: 487, 6: 476, 7: 455, 8: 476, 9: 495,
# 10: 492, 11: 470, 12: 477, 13: 475, 14: 466,
# 15: 477, 16: 425, 17: 462, 18: 358, 19: 302,

# First do some preliminary analysis to make sure that training and test data categories are evenly distributed
#There are 7532 measurements in the test set
#There are 11314 measurements in the training set

# Here are the separate categories to be distinguished
# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
#  'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
#  'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
#  'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


# twenty_train.target_names
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print(len(twenty_train.data))

# print(twenty_train.target_names[twenty_train.target[0]])
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print(twenty_train.target[:10])
# for t in twenty_train.target[:10]:print(twenty_train.target_names[t])
#np.bincount()

# def get_word_indices (vocab_mapping, list_of_words):
#     word_index = np.zeros(list_of_words.len)
#     for i in list_of_words:
#         word_index = vocab_mapping.vocabulary_.get(u list_of_words)
#     return word_index

def word_count_plot (vect_counts, fig_name):
    vect_counts
    """
    This function creates a plot showing words against number of occurences. It's input is a sparse matrix. It counts the
    number of time each word appears and graphs it.
    """

    X_train_counts_wordcount = vect_counts.sum(axis=0)
    print("Word count is: ", X_train_counts_wordcount[0])
    print(X_train_counts.shape[1])
    plt.plot(X_train_counts_wordcount.T)
    plt.xlabel("Features")
    plt.ylabel("Number of appearances")
    plt.title("Features vs number of appearances")
    plt.savefig(f'{fig_path}/{fig_name}.png')
    return None

def get_term(dict, search_index):
    return list(dict.keys())[list(dict.values()).index(search_index)]
# If performing bag_of_words tests, state that here
bag_of_words = False

if bag_of_words:
    # Perform bag of words approach. This creates a sparse matrix where each word is assigned a number
    count_vect = CountVectorizer(strip_accents='unicode', min_df = 50, max_df = 400, max_features = 10000, ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    #print(count_vect.vocabulary_)
    #print(X_train_counts.shape)

    #word_count_plot(X_train_counts)

    # X_train_counts_wordcount = X_train_counts.sum(axis=0)
    # print("Word count is: ", X_train_counts_wordcount[0])

    # print("trying to make plot")
    # plt.plot(X_train_counts_wordcount.T)
    #plt.show()
    # # print(count_vect.vocabulary_.get(u'the'))
    # # print(count_vect.vocabulary_.get(u'alpha'))

    # print("the second word appears: ", X_train_counts.shape[1], " times")
    # CountVectorizer just turns articles into features with frequencies of words



    #bow_nb = MultinomialNB().fit(X_train_counts, y_train)

    text_bow = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB()),])
    text_bow.fit(X_train, y_train)
    predicted = text_bow.predict(X_validation)
    print(np.mean(predicted == y_validation))
    
    #print (metrics.classification_report(y_validation, predicted))
    conf_matrix = confusion_matrix(y_validation, predicted)
    vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=text_bow.classes_)
    vis.plot()
    plt.title("Bag_of_words and Multinomial Naive Bayes")
    plt.savefig(f'{fig_path}/xBOFMNB.png')
    #plt.show()
    
    # Investigate count vectorizer more, maybe create histograms. This would be a good place to explore class differences

    # Perform Term Frequency times Inverse Document Frequency. 
    # Term frequency divides each word in a document by the total words in that document
    # Inverse document frequency decreases the weights of words based on how often they appear in many documents

run_experiment_bool = True
tfidf_bool = True
naive_bayes_bool = False
svm_bool = True
random_forest_bool = False


if tfidf_bool:
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(twenty_train.data)
    # tfidf_transformer = TfidfTransformer(use_idf=True, max_features = 100)
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(tfidf_transformer.vocabulary_)

    tf_idf_vectorizer = TfidfVectorizer(max_features = 100, max_df = 1000, min_df = 50)
    tf_idf_counts = tf_idf_vectorizer.fit_transform(X_train)

    feature_names = tf_idf_vectorizer.get_feature_names_out()
    dense = tf_idf_counts.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    #print(df.head)
    x = tf_idf_vectorizer.vocabulary_
    Cloud = WordCloud(background_color="white").generate_from_frequencies(df.T.sum(axis=1))
    Cloud.to_file(f'{fig_path}/tf_idf_cloud.png')
    
    #print(tf_idf_counts)
    #print(tf_idf_counts.idf_)
    # print(X_train_tfidf.shape)
    # print(X_train_tfidf)
    # #print(X_train_tfidf.n_features_in_())
    # #print(X_train_tfidf.feature_names_in_())

    # # Train a Naive Bayes classifier on the data
    # # https://scikit-learn.org/stable/modules/naive_bayes.html

    # mnb = MultinomialNB().fit(X_train_tfidf, y_train)
    # predicted = mnb.predict()
    # print (metrics.classification_report(y_validation, predicted))

    # conf_matrix = confusion_matrix(y_validation, predicted)
    # vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=text_bow.classes_)
    # vis.plot()
    # plt.title("TF-IDF and Multinomial Naive Bayes")
    # plt.show()

    if naive_bayes_bool:

        # Create a pipeline that performs the three relevant functions
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_validation)
        # The predicted value on the tf-idf validation set is 0.844
        #print(np.mean(predicted == y_validation))
        
        conf_matrix = confusion_matrix(y_validation, predicted)
        plot_confusion_matrix(conf_matrix, categories, title = 'TF-IDF Naive Bayes Confusion Matrix')

        print (metrics.classification_report(y_validation, predicted))

    if svm_bool:
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range = (1,3), min_df = 10)), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2',
                                                                                                            alpha=1e-3, random_state = 42,
                                                                                                            max_iter = 5, tol=None)),])
        
        text_clf.fit(X_train, y_train)

        if run_experiment_bool:
            print("Inside experiment loop")
            parameters = { 'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5),}
            gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
            gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
            print(gs_clf.best_score_)
            print(gs_clf.best_params_)


        predicted = text_clf.predict(X_validation)
        # The predicted value of the tf-idf validation set is 0.891
        print("Prediction accuracy on validation set is: ", np.mean(predicted == y_validation))
        print (metrics.classification_report(y_validation, predicted))

        # Plot Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(y_validation, predicted)
        plot_confusion_matrix(confusion_matrix, categories, title = "TF-IDF SVM Confusion Matrix")

        #Maybe consider word2vec
        # We can look at word2vec as future work. Or we can possibly take a model out of the box and start messing with it.