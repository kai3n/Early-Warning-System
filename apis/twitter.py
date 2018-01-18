from TwitterSearch import *

try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords(['Justin', 'Bieber']) # let's define all words we would like to have a look for
    tso.set_language('en') # we want to see German tweets only
    tso.set_include_entities(False) # and don't give us all those entity information

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key = 'EsnBdOOkurnvkSODDYWBqkySQ',
        consumer_secret = 'nFQF2GTOBqbrI28FgCWxHKQV1Ysp5gIMSYVAWrKdUjdIIezwHI',
        access_token = '830323475608805376-Kdl0UOnNc3HlVXMvZIiGyeiahC6ZHUE',
        access_token_secret = 'JecshkKZsg1uEJTTtPKmXZmrX08mjwyu0ipX53phMuXuY'
     )

     # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):
        print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)