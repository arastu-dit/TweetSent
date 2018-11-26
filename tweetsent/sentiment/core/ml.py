'''
    ML logic.
'''
from __future__ import print_function

import pandas as pd
import random
import json
import os
import pickle

from pprint                             import pprint
from .clean                             import clean_and_stem
from itertools                          import chain

from sklearn.externals                  import joblib
from sklearn.feature_extraction.text    import TfidfVectorizer, CountVectorizer
from sklearn.model_selection            import train_test_split
from sklearn.neural_network             import MLPClassifier
from sklearn.neighbors                  import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model               import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.svm                        import SVC, LinearSVC
from sklearn.linear_model               import SGDClassifier
from sklearn.gaussian_process           import GaussianProcessClassifier
from sklearn.tree                       import DecisionTreeClassifier
from sklearn.ensemble                   import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes                import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics                    import accuracy_score


def delete(id):
    '''
        Deletes a model with a given id.
        Args    :
            id  : Model id.
    '''
    dir     = os.path.dirname(__file__)
    path    = os.path.join(dir, 'models')
    files   = [x for x in os.listdir(path) if id in x]

    path    = os.path.join(
        os.path.dirname(__file__)   ,
        'models'                    ,
        files[0]                    )

    os.remove(path)

def get_added_name(path, added_name= '_save'):
    '''
        Creates the save file name from the original file name.
        Args    :
            path        : Path to the original file.
            added_name  : The text added to file name to generate the new file name.
        Returns : The new file path
    '''
    basename    = os.path.basename(path)
    dirname     = os.path.dirname (path)
    file_name   = '.'.join(basename.split('.')[:-1])    if '.' in basename else basename
    extension   = ('.'+ basename.split('.')[-1])        if '.' in basename else ''
    return os.path.join(
        dirname                         ,
        file_name+ added_name+ extension)

def create_from_params(**kwargs):
    '''
        Train a model from web request params.
    '''
    print(kwargs)
    classifier  = Classifier(
        classifiers     = {
            eval(kwargs['classifier']) : {'toarray': eval(kwargs['toarray'])}},
        ds_path         = os.path.join(
            os.path.dirname(__file__),
            'data'  ,
            kwargs['ds_path'])                          ,
        clean_data      = eval(kwargs['clean_data'])    ,
        min_df          = eval(kwargs['min_df'])        ,
        data_size       = eval(kwargs['data_size'])     ,
        train_size      = eval(kwargs['train_size'])    ,
        tfidf           = eval(kwargs['tfidf'])         ,
        text_column     = kwargs['text_column']         ,
        category_column = kwargs['category_column']     ,
        encoding        = kwargs['encoding']            ,
        header          = eval(kwargs['header'])        ,
        index_col       = kwargs['index_col']           ,
        max_features    = eval(kwargs['max_features'])  )

    classifier.save()

def split(x):
    '''
        A hack to get pickle working!
    '''
    return x.split()

def predict(model, text):
    '''
        Predicts the emotion of the tweet using the trained model.
        Args    :
            model       : The trained model.
            tweet       : text to classify.
            words_occ   : words occurences
    '''
    if model['Cleaned']:
        text        = clean_and_stem(text)

    vectorized      = model['model'].vectorizer.transform([text])
    if model['model'].toarray :
        vectorized  = vectorized.toarray()

    return model['model'].predict(vectorized)

def load_models(root= 'models'):
    '''
        Loads saved models.
        Args    :
            root    : The directory where the models are saved.
    '''

    dir = os.path.dirname(__file__)
    path= os.path.join(dir, root)
    models  = []
    for x in os.listdir(path):
        if x.split('.')[-1] == 'pkl':
            name    = x[:-4]
            id      = name.split('_')[9]
            model   = {
                    'Classifier'    : name.split('_')[0]        ,
                    'Data set'      : name.split('_')[1]        ,
                    'Cleaned'       : eval(name.split('_')[2])  ,
                    'Min dif'       : int(name.split('_')[3])   ,
                    'Data size'     : int(name.split('_')[4])   ,
                    'Train size'    : float(name.split('_')[5]) ,
                    'TFIDF'         : eval(name.split('_')[6])  ,
                    'Max features'  : eval(name.split('_')[7])  ,
                    'Accuracy'      : float(name.split('_')[8]) ,
                    'model'         : joblib.load(os.path.join(path, x))}
            model['model'].full_name= ', '.join(name.split('_')[:-1])
            models.append([id, model])
    return models

def predict_all(tweets):
    '''
        Predict The tweet sentiment using all the trained models.
    '''
    result  = []

    all_reds    = [0, 4]
    for tweet in tweets:
        preds   = []
        for x in load_models():
            preds.append([x[1]['model'].full_name, predict(x[1], tweet),x[1]['Accuracy'] -0.50 if x[1]['Accuracy'] > 0.50 else 0])

        precs       = {}
        for pred in all_reds :
            arr         = [pred_[2] for pred_ in preds if pred_[1][0] == pred]
            precs[pred] = sum(arr) if len (arr) else 0

        result.append([
            tweet,
            preds,
            {pred: precs[pred]/sum(precs.values())*100 if sum(precs.values()) else 0 for pred in precs},
            ])

    sents   = []
    for tweet in result:
        if tweet[2][0]> tweet[2][4] :
            sents.append('Negative')
        elif tweet[2][0]< tweet[2][4] :
            sents.append('Positive')

        else :
            sents.append('Neutral')



    pprint(result)
    return result,[
        sents.count('Negative')/ len(sents),
        sents.count('Positive')/ len(sents),
        sents.count('Neutral')/ len(sents),
        ]

class Classifier():
    '''
        A custom classifier class
        Args    :
            classifiers     : Sklearn classifier classes.
            ds_path         : Data set path.
            clean_data      : Uses clean.py to clean data
            min_df          : Minimum number of a token occurences.
            data_size       : Used to extract a sub set for training, only the generated sub set will be used.
            tfidf           : Uses a tfidf vectorizer.
            text_column     : Text column.
            category_column : Category column.
            encoding        : Data set encoding.
            header          : Data set header row index.
            index_col       : The index column.
    '''
    def __init__(
        self            ,
        classifiers     ,
        ds_path         ,
        clean_data      ,
        min_df          ,
        data_size       ,
        train_size      ,
        tfidf           ,
        text_column     ,
        category_column ,
        encoding        ,
        header          ,
        index_col       ,
        max_features=None):

        self.classifiers            = classifiers
        self.ds_path                = ds_path
        self.clean_data             = clean_data
        self.min_df                 = min_df
        self.data_size              = data_size
        self.train_size             = train_size
        self.tfidf                  = tfidf
        self.encoding               = encoding
        self.header                 = header
        self.max_features           = max_features

        self.text_column            = text_column       if header!=None else int(text_column)
        self.category_column        = category_column   if header!=None else int(category_column)
        self.index_col              = index_col         if header!=None else int(index_col)

        print('>>> Loading: {} ...'.format(self.ds_path))
        self.df, self.df_remaining  = self.load_ds()
        self.labels                 = self.df[str(self.category_column) if clean_data else self.category_column].values
        print('>>> Vectorizing ...')
        self.vectorizer             = self.vectorize()
        values                      = self.df['clean' if self.clean_data else self.text_column ].values
        self.vectorized             = self.vectorizer.fit_transform(values)

        for x in self.classifiers:
            xd  = classifiers[x]
            print('Training: {}'.format(x.__class__.__name__))
            train_data          = self.split(xd.get('toarray'))
            model               = x.fit(train_data[0], train_data[2])
            model.toarray       = xd.get('toarray')
            predictions         = model.predict(train_data[1])
            accuracy            = accuracy_score(train_data[3], predictions)
            model.vectorizer    = self.vectorizer

            self.classifiers[x]['accuracy' ]= accuracy
            print(accuracy)
            self.classifiers[x]['model']    = model

    def load_ds(self):
        '''
            Loads the data set from disk.
            Args    :
                ds_path : The path to the data set.
                encoding: Data set encoding.
        '''
        #Load the original data set
        df = pd.read_csv(
            self.ds_path                ,
            encoding    = self.encoding ,
            header      = self.header   ,
            index_col   = self.index_col)

        #Check if there is a clean data set
        clean_path  = get_added_name(self.ds_path, '_clean')
        if not os.path.isfile(clean_path):

            df['clean'] = df[self.text_column].apply(clean_and_stem)

            df = df[df['clean'] != '']
            with open(clean_path, 'w') as f:
                df[[self.category_column, 'clean']].to_csv(f)

        df      = pd.read_csv(
            clean_path if self.clean_data else self.ds_path             ,
            encoding    = self.encoding                                 ,
            header      = 0     if self.clean_data else self.header     ,
            index_col   = None  if self.clean_data else self.index_col  )
        return df.sample(frac= 1)[:self.data_size], df.sample(frac= 1)[self.data_size:]

    def vectorize(self):
        '''
            Vectorizes the data set.
        '''
        if self.tfidf :
            return TfidfVectorizer(
                analyzer    = 'word'                                                ,
                stop_words  = 'english'                                             ,
                lowercase   = True                                                  ,
                tokenizer   = None if not self.clean_data else split                ,
                min_df      = self.min_df                                           ,
                max_features= self.max_features)
        else :
            return CountVectorizer(
                analyzer    = 'word'                                                ,
                stop_words  = 'english'                                             ,
                lowercase   = True                                                  ,
                tokenizer   = None if not self.clean_data else split                ,
                min_df      = self.min_df                                           ,
                max_features= self.max_features)

    def split(self, toarray):
        '''
            Split the data set into test and train data.
            Args    :
                toarray : Convert to array.
        '''
        return  train_test_split(
            self.vectorized.toarray() if toarray else self.vectorized   ,
            self.labels                                                 ,
            train_size   = self.train_size                              ,
            random_state = random.randint(0,1000)                       )

    def save(self, root= 'models'):
        '''
            Save the trained classifiers.
        '''
        name_fomat  = ('{}_'*10)[:-1]+ '.pkl'
        dir = os.path.dirname(__file__)
        path= os.path.join(dir, root)

        for x in self.classifiers:
            name = name_fomat.format(
                x.__class__.__name__            ,
                os.path.basename(self.ds_path)  ,
                self.clean_data                 ,
                self.min_df                     ,
                self.data_size                  ,
                self.train_size                 ,
                self.tfidf                      ,
                self.max_features               ,
                self.classifiers[x]['accuracy'] ,
                random.getrandbits(128)         )

            joblib.dump(self.classifiers[x]['model'], os.path.join(path, name))
