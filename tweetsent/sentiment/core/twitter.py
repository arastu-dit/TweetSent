'''
    Using twitter api to get live tweets on a specific topic
'''
import tweepy

from time import sleep

#You need to create an app to get ACCESS_TOKEN, and ACCESS_SECRET
CONSUMER_KEY        = 'UVcHopwj8BktAw8IJym85LMQW'
CONSUMER_SECRET     = 'cKUcH19cBJLFQp2TxXyKxk4K93xkcGjf07AR2aqQmMXLXF8V62'
ACCESS_TOKEN        = '578870876-hYKkWT8HQ54CJFCxLh0UyFjPDi7yXvpTfosiZT5w'
ACCESS_SECRET       = '0Ux6iXhROIQqao5cnptByUxx9W9Sr4vnadCFxLMFoi61O'

def auth():
    '''
        Authentiate to twitter.
        Returns : The api object.
    '''
    auth = tweepy.OAuthHandler  (CONSUMER_KEY   , CONSUMER_SECRET   )
    auth.set_access_token       (ACCESS_TOKEN   , ACCESS_SECRET     )
    return tweepy.API(auth, parser=tweepy.parsers.JSONParser())

def get_last_tweets(tag, api, limit= 10):
    '''
        Get the last tweets corresponding the given tag.
        Args    :
            tag     : A tag to get tweets for.
            api     : The api object.
        Returns : None if api call limit is reached or a list of string (tweets).
    '''
    result      = None
    try :
        result  = api.search('#'+tag, count= limit, tweet_mode= 'extended', result_type= 'recent')
        result  = [x['retweeted_status']['full_text'] if x.get('retweeted_status') else x['full_text'] for x in  result['statuses']]
    except :
        pass

    return result
