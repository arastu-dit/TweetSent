'''
    Unit tests for clean.py
'''

from .clean   import *

def test_clean_text         ():
    '''
        Tests clean.clean_text
    '''
    tweet   = '@user_name #hashtag 1234 #$% just  a tweet'
    assert  clean_text(tweet) == \
            'just a tweet'

def test_tokenize_text      ():
    '''
        Tests clean.tokenize_text
    '''
    tweet   = 'just a tweet to test tokenization'
    assert  tokenize_text(tweet) == \
            ['just', 'a', 'tweet', 'to', 'test', 'tokenization']

def test_stem_text          ():
    '''
        Tests clean.stem_text
    '''
    tokens  = ['just', 'a', 'tweet', 'to', 'test', 'tokenization']
    assert  stem_text(tokens)  == \
            ['just', 'a', 'tweet', 'to', 'test', 'token']

def test_remove_stop_words  ():
    '''
        Tests clean.remove_stop_words
    '''
    tokens  = ['just', 'a', 'tweet', 'to', 'test', 'token']
    assert  remove_stop_words(tokens)   == \
            ['tweet', 'test', 'token']

def test_replace_antonyms   ():
    '''
        Tests clean.replace_antonym
    '''
    tweet   = 'not good, i don\'t like this product and i won\'t buy it'

    assert  replace_antonyms(tweet) == \
            'bad, i dislike this product and i sell it'

def test_clean_and_stem     ():
    '''
        Tests clean.clean_and_stem
    '''
    tweet   = 'not good, i don\'t like this product and i won\'t buy it, i will just keep using my old one #badproduct'

    assert  clean_and_stem(tweet) == \
            ' '.join(['bad', 'dislik', 'product', 'sell', 'keep', 'use', 'old', 'one'])
