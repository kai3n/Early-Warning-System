from collections import Counter

from TwitterSearch import *
from utils import email

def main():
    targets = []
    try:
        tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
        tso.set_keywords(['Justin Bieber'])  # let's define all words we would like to have a look for
        tso.set_language('en')  # we want to see German tweets only
        tso.set_include_entities(False)  # and don't give us all those entity information

        #TODO: store these keys in the different file
        ts = TwitterSearch(
            consumer_key='EsnBdOOkurnvkSODDYWBqkySQ',
            consumer_secret='nFQF2GTOBqbrI28FgCWxHKQV1Ysp5gIMSYVAWrKdUjdIIezwHI',
            access_token='830323475608805376-Kdl0UOnNc3HlVXMvZIiGyeiahC6ZHUE',
            access_token_secret='JecshkKZsg1uEJTTtPKmXZmrX08mjwyu0ipX53phMuXuY'
        )

        # this is where the fun actually starts :)
        for tweet in ts.search_tweets_iterable(tso):
            # print('{},{}'.format(tweet['created_at'], tweet['text']))
            targets.append(tweet['text'])
        targets = Counter(targets)
        top10_issues = sorted(targets, key=lambda x: targets[x], reverse=True)[:10]
        print(top10_issues)

        #TODO: text preprocessing

        #TODO: make features

        #TODO: do clustering

        #TODO: do sentiment analysis

        # notify if it's unsafe through the email
        sender_name = 'Gumgum'
        sender_email = 'jpak1021@gmail.com'
        sender_pw = 'xxxxxxxxxx'
        subject = 'Early Warning Alarm for Justin Bieber'
        content = 'We are writing this to let you know that Justin Beiber is in trouble now.'

        e = email.Email()(sender_email, sender_pw, sender_name, subject)
        e.set_recipient('diadld2@naver.com')
        e.send_email(content)

    except TwitterSearchException as e:  # take care of all those ugly errors if there are some
        print(e)


if __name__ == '__main__':
    main()
