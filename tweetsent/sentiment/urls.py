from django.urls import path
from . import views

urlpatterns = [
    path(''                 , views.tweets          , name='tweets'         ),
    path('tweets'                 , views.tweets          , name='tweets'         ),
    path('compare'                 , views.compare          , name='compare'         ),
    path('add_classifier'   , views.add_classifier  , name='add_classifier' ),
    path('delete_classifier'   , views.delete_classifier  , name='delete_classifier' ),
    path('classifiers'      , views.classifiers     , name='classifiers'    ),
    ]
