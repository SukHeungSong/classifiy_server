# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from bottle import route, run, request
# def getModel():

clf = joblib.load('classify.model')
cate_dict = joblib.load('cate_dict.dat')
vectorizer = joblib.load('vectorizer.dat')

joblib.dump(clf,'n_classify.model')
joblib.dump(cate_dict,'n_cate_dict.dat')
joblib.dump(vectorizer,'n_vectorizer.dat')

cate_id_name_dict = dict([(v, k) for k, v in cate_dict.items()])

pred = clf.predict(vectorizer.transform(['[신한카드5%할인][서우한복] 아동한복 여자아동 금나래 (분홍)']))[0]
# print (cate_id_name_dict[pred])


from threading import  Condition
_CONDITION = Condition()
@route('/classify')
def classify():
    print ("classify called")
    img = request.GET.get('img','')
    name = request.GET.get('name', '')
    pred = clf.predict(vectorizer.transform([name]))[0]
    return {'cate':cate_id_name_dict[pred]}


run(host='0.0.0.0', port=8887)


# getModel()