'''
    Unit tests for ml.py
'''

import os

from .ml                                import *
from sklearn.naive_bayes                import GaussianNB, BernoulliNB

DS_PATH = os.path.join(
    os.path.dirname(__file__),
    'test_data'  ,
    'dstest.csv'
    )

def test_get_added_name     ():
    '''
        Tests ml.get_added_name
    '''
    paths       = [
        os.path.join('a', 'b', 'c', 'd.ext'),
        os.path.join('a', 'b', 'c')]

    expected    = [
        os.path.join('a', 'b', 'c', 'd_save.ext'),
        os.path.join('a', 'b', 'c_save')]

    results     = [get_added_name(x) for x in paths]

    assert expected == results

def test_classfier_load_ds():
    '''
        Tests ml.Classifier
    '''
    cl1         = BernoulliNB()
    cl2         = GaussianNB()
    classifier = Classifier(
        classifiers     = {
            cl1     : {}    ,
            cl2     : {
                'toarray'   : True}},
        ds_path         = DS_PATH       ,
        clean_data      = True          ,
        min_df          = 0             ,
        data_size       = 99            ,
        train_size      = 0.8           ,
        tfidf           = True          ,
        text_column     = 5             ,
        category_column = 1             ,
        encoding        = 'ISO-8859-1'  ,
        header          = None          ,
        index_col       = 1             )


    assert  len(classifier.df)           == 99      and \
            len(classifier.df_remaining) >  99      and \
            classifier.vectorized        != None


    classifier.save('test_models')
    models  = load_models('test_models')
    assert  len(models) > 0 and \
            predict(models[0][1],'bad bad bad') == 0
