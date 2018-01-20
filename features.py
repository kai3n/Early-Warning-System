TARGET_WORDS_SET = set(['shot', 'pissing', 'kill', 'victim', 'stoned', 'hate', 'police', 'rohypnol', 'arrest', 'black', 'trauma', 'fatal', 'pill', 'hurting', 'crash', 'dope', 'awful', 'roaches', 'die', 'gun', 'turn up', 'careless', 'roofies', 'turnt up', 'troublemaker,tragic', 'forgetme', 'ruffies', 'throwing', 'smoke dope', 'dead', 'sex', 'sexually', 'sexual', 'savage', 'shirtless', 'critical', 'suspect', 'horrible', 'screwed', 'torturing', 'broken', 'get baked', 'smart drug', 'dammit', 'accident', 'marijuana', 'guilty', 'assault', 'argue', 'burning', 'drunk', 'pot', 'sexually assault', 'racism', 'fearfully', 'punching', 'probation', 'cocaine', 'found drunk', 'cheating', 'out of control', 'divorce', 'vulgar', 'hammered', 'roche', 'racist', 'troubling', 'break up', 'dumped', 'spit', 'drowning', 'fuck', 'addies', 'wasted', 'critically', 'dirty', 'adderall', 'heart', 'paparazzo', 'unsafe', 'argument', 'recklessly', 'daterape drug', 'shit', 'deadly', 'fears', 'sick', 'scandal', 'reckless', 'get high', 'tipsy', 'arguing', 'fucken', 'mad', 'spitting', 'hurt', 'found guilty', 'fighting', 'shade', 'womanizer', 'egg'])

# uni = [ a if(a[0:2]=='__') else a.lower() for a in re.findall(r"\w+", text) ]
# bi  = nltk.bigrams(uni)
# tri = nltk.trigrams(uni)


import nltk
import re


from nltk.stem.snowball import SnowballStemmer
from nltk import bigrams, trigrams
from nltk.corpus import stopwords

# stopwords = stopwords.words('english')


def tokenize_and_stem(text):

    stemmer = SnowballStemmer("english")

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def extract_features(proc_tweets):

    stemmer = SnowballStemmer("english")
    features = []                                             #DATADICT: all_tweets =   [ (words, sentiment), ... ]
    for (sentiment, text) in proc_tweets:
        words = [word if(word[0:2]=='__') else word.lower() \
                    for word in text.split() \
                    if len(word) >= 3]
        words = [stemmer.stem(w) for w in words]                #DATADICT: words = [ 'word1', 'word2', ... ]
        features.append([sentiment, words])

    for feature in features:
        feature[1] = get_word_features(feature[1])
    return features


def get_word_features(words):
    bag = {}
    words_uni = ['has({})'.format(ug) for ug in words]
    words_bi = ['has({})'.format(','.join(map(str, bg))) for bg in bigrams(words)]
    words_tri = ['has({})'.format(','.join(map(str, tg))) for tg in trigrams(words)]
    for f in words_uni + words_bi + words_tri:
        bag[f] = 1
    return bag

from sklearn.feature_extraction.text import TfidfVectorizer


tweet = ['Watchin Espn..Jus seen this new Nike Commerical with a Puppet Lebron..sh*t was hilarious...LMAO!!!',
         '@wordwhizkid Lebron is a beast... nobody in the NBA comes even close.',
         'downloading apps for my iphone! So much fun :-) There literally is an app for just about anything.']


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(tweet) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
print(tfidf_matrix)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print(clusters)
print(km.predict([[0,2,3,5,1,1,1,1,1,1,4,2,0,2,3,5,1,1,1,1,1,1,4,2,0,2,3,5,1,1,1,1,1,1,4,2,0,2,3,5,1,1,1,1,1,1,4,2,0,2,3,5,1,1,1,1,1,1,4,2,9,8,7,6,5],
                  [0, 2, 3, 5, 1, 1, 1, 1, 1, 1, 4, 2, 0, 2, 3, 5, 1, 1, 1, 1, 1, 1, 4, 2, 0, 2, 3, 5, 1, 1, 1, 1, 1, 1,
                   4, 2, 0, 2, 3, 5, 1, 1, 1, 1, 1, 1, 4, 2, 0, 2, 3, 5, 1, 1, 1, 1, 1, 1, 4, 2, 9, 8, 7, 6, 5]]))



from sklearn.cluster import KMeans

#
# X = extract_features(tweet)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(tweet)


