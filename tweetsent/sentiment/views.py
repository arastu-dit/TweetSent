from django.shortcuts   import render, redirect
from .core.ml           import *
from .core.twitter      import *

import json
# Create your views here.


def classifiers(request):
    models = load_models('models')
    for x in models:
        del x[1]['model']
    return render(request, 'classifiers.html',{'models': models})

def delete_classifier(request):
    delete(request.GET.get('id'))
    return redirect('/classifiers')

def add_classifier(request):
    params = [
        'classifier'        ,
        'ds_path'           ,
        'clean_data'        ,
        'min_df'            ,
        'data_size'         ,
        'train_size'        ,
        'tfidf'             ,
        'text_column'       ,
        'category_column'   ,
        'encoding'          ,
        'header'            ,
        'index_col'         ,
        'toarray'           ,
        'max_features'
        ]
    create_from_params(
        **{x: request.GET.get(x) for x in params}
        )
    return redirect('/classifiers')


def tweets(request):
    tag     = request.GET.get('tag', 'bitcoin')
    tweets  = get_last_tweets(tag, auth())
    result,sents  = predict_all(tweets)

    return render(request, 'tweets.html',
        {
            'result': result                                            ,
            'labels': json.dumps([i for i in range(1,len(result)+1)])   ,
            'data'  : [json.dumps([int(result[j][1][i][1][0])+1 for j in range(len(result))]) for i in range(len(result[0][1]))],
            'avgs'  : [
                json.dumps([x[2][0] for x in result]),
                json.dumps([x[2][4] for x in result])],
            'sents' : json.dumps(sents)
        })


def compare(request):
    tags    = request.GET.get('tag', 'bitcoin, eth, tesla')
    tags    = [x.strip() for x in tags.split(',')][:3]
    alphs   = ['A','B','C']
    resp = {}

    for tag in tags:
        i = tags.index(tag)
        tweets  = get_last_tweets(tag, auth())
        result,sents  = predict_all(tweets)
        
        resp ['tag'+alphs[i]]       = tag
        resp ['result'+alphs[i]]    = result
        resp ['labels'+alphs[i]]    = json.dumps([i for i in range(1,len(result)+1)])
        resp ['data'+alphs[i]]      = [json.dumps([int(result[j][1][i][1][0])+1 for j in range(len(result))]) for i in range(len(result[0][1]))]
        resp ['avgs'+alphs[i]]      = [
            json.dumps([x[2][0] for x in result]),
            json.dumps([x[2][4] for x in result])]
        resp ['sents'+alphs[i]]     = json.dumps(sents)

    return render(request, 'compare.html', resp)
