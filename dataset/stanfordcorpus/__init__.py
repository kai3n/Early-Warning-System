"""
http://help.sentiment140.com/for-students
Format
Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)
"""

TARGET_WORDS_SET = set(['accident', 'adderall', 'addies', 'anger', 'argue', 'arguing', 'argument', 'arrest', 'assault', 'awful', 'beef', 'black', 'break up', 'broken', 'burning', 'careless', 'cheating', 'cocaine', 'crash', 'critical', 'critically', 'cuss', 'dammit', 'daterape drug', 'dead', 'deadly', 'die', 'dirty', 'divorce', 'dope', 'drowning', 'drunk', 'dui', 'dumped', 'egging', 'fatal', 'fearfully', 'fears', 'fighting', 'forgetme', 'found drunk', 'found guilty', 'fuck', 'fucken', 'get baked', 'get high', 'guilty', 'gun', 'hammered', 'harrass', 'hate', 'heart', 'hit', 'horrible', 'hurt', 'hurting', 'jail', 'kill', 'mad', 'marijuana', 'meth', 'out of control', 'pain', 'paparazzo', 'pill', 'pissing', 'police', 'pot', 'prison', 'probation', 'prostitute', 'punch', 'punching', 'racism', 'racist', 'reckless', 'recklessly', 'roaches', 'roche', 'rohypnol', 'roofies', 'ruffies', 'savage', 'scandal', 'screwed', 'sex', 'sexual', 'sexually', 'sexually assault', 'shade', 'shirtless', 'shit', 'shot', 'sick', 'smart drug', 'smoke dope', 'speed', 'spit', 'spitting', 'stoned', 'suspect', 'suspend', 'swear', 'throwing', 'throwing eggs', 'tipsy', 'torturing', 'tragic', 'trauma', 'trouble', 'troublemaker', 'troubling', 'turn up', 'turnt up', 'unsafe', 'victim', 'vulgar', 'wasted', 'weed', 'whore', 'womanizer', 'xanax'])
FULLDATA = 'training.1600000.processed.noemoticon.csv'
TESTDATA = 'testdata.manual.2009.06.14.csv'

POLARITY= 0 # in [0,5]
TWID    = 1
DATE    = 2
SUBJ    = 3 # NO_QUERY
USER    = 4
TEXT    = 5

import csv, re, random

regex = re.compile(r'\w+|\".*?\"')


def get_tweets_raw_data(out_file):

    tweets = []
    # read all tweets and labels
    with open(out_file, 'r', encoding = "ISO-8859-1" ) as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"', escapechar='\\')
        for row in reader:
            try:
                tweets.append( [row[POLARITY], row[TEXT]] )
            except:
                continue

        # treat neutral and irrelevant the same
        for t in tweets:
            if (t[1] == 'positive'):
                t[1] = 'pos'
            elif (t[1] == 'negative'):
                t[1] = 'neg'
            elif (t[1] == 'irrelevant')|(t[1] == 'neutral'):
                t[1] = 'neu'

    return tweets # 0: Text # 1: class # 2: subject # 3: query


def store_tweets_raw_data(data, out_file):

    with open(out_file, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', escapechar='\\')
        for each in data:
            spamwriter.writerow([each])



def get_class(polarity):
    if polarity in ['0', '1']:
        return 'neg'
    elif polarity in ['3', '4']:
        return 'pos'
    elif polarity == '2':
        return 'neu'
    else:
        return 'err'


def get_query(subject):
    if subject == 'NO_QUERY':
        return []
    else:
        return regex.findall(subject)


def get_all_queries(in_file):

    fp = open(in_file , 'r')
    rd = csv.reader(fp, delimiter=',', quotechar='"' )

    queries = set([])

    for row in rd:
        queries.add(row[3])

    print(queries)

    for q in queries:
        print(q, "\t",)

    return queries


def sample_csv(in_file, out_file, K=100):

    fp = open(in_file , 'r')
    fp2 = open(out_file , 'w')

    for i in range(0,K):
        line = fp.readline()
        fp2.write(line)

    fp.close()
    fp2.close()

    return 0


def random_sample_csv(in_file, out_file, K=100):

    fp = open(in_file , 'r')
    fq = open(out_file, 'w')

    rows = [None] * K

    i = 0
    for row in fp:
        i+=1
        j = random.randint(1,i)
        if i < K:
            rows[i] = row
        elif j <= K:
            rows[j-1] = row

    for row in rows:
        fq.write(row)

    min(1, K/i)


def get_normalised_csv(in_file, out_file):
    fp = open(in_file , 'r')
    rd = csv.reader(fp, delimiter=',', quotechar='"' )

    fq = open(out_file, 'w')
    wr = csv.writer(fq, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL )

    for row in rd:
        queries = get_query(row[SUBJ])
        wr.writerow( [row[TEXT], get_class(row[POLARITY]), row[SUBJ]] + [len(queries)] + queries )


def get_normalised_tweets(in_file):
    fp = open(in_file , 'r')
    rd = csv.reader(fp, delimiter=',', quotechar='"' )
    #print in_file, count_lines( in_file )

    tweets = []
    count = 0
    for row in rd:
        numQueries = int(row[3])
        tweets.append( row[:3] + [row[4:4+numQueries]] )
        count+=1

    #print count
    #print 'len(tweets) =', len(tweets)
    return tweets


def count_lines(filename):
    count = 0
    with open(filename, 'r') as fp:
        for line in fp:
            count+=1
    return count


if __name__ == '__main__':

    # get_all_queries( 'testdata.manual.2009.06.14.csv' )
    # get_all_queries( 'training.1600000.processed.noemoticon.csv' )

    from preprocess import Preprocesor
    p = Preprocesor()

    tweets = get_tweets_raw_data('training.1600000.processed.noemoticon.csv')
    res = []
    for _, text in tweets:
        for word in text.split():
            if word in TARGET_WORDS_SET:
                res.append(p.preprocess(text))
    store_tweets_raw_data(res, 'train2.txt')

    #random_sample_csv(FULLDATA, FULLDATA+'.sample.csv')
    #sample_csv(TESTDATA, TESTDATA+'.sample.csv')

    #get_normalised_csv(FULLDATA+'.sample.csv', FULLDATA+'.norm.csv')

    #random_sample_csv(FULLDATA, FULLDATA+'.100000.sample.csv', K=100000)
    #get_normalised_csv(FULLDATA+'.100000.sample.csv', FULLDATA+'.100000.norm.csv')


