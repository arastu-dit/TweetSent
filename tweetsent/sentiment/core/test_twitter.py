'''
    Unit tests for twitter.py
'''

from tweepy     import api
from .twitter   import *

def test_auth():
    '''
        Tests twitter.auth
    '''
    api_obj     = auth()
    assert  type(api_obj) == \
            type(api)


def test_get_last_tweets():
    '''
        Tests twitter.get_last_tweets
    '''
    api_obj     = auth()
    last_tweets = get_last_tweets('Bitcoin', api_obj)
    assert  len(last_tweets) > \
            0
