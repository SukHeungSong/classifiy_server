# -*- coding: utf-8 -*-

# feture를 기존 저장되어 있는 data와 비교하여 가장 유사도가 높은 ID를 찾아내서 분류한다

# stop word removal
# from konlpy.tag import Twitter

from nltk import collocations
from nltk.tokenize import word_tokenize

from nltk import regexp_tokenize

from nltk.tag import pos_tag


#한글 형태소 분석기
from konlpy.tag import Twitter
pos_tagger = Twitter()

import nltk

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# SVM을 이용해서 분류학습한다
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
import numpy as np
# 만들어진 모델을 나중에 분류할때 사용하기 위해..
from sklearn.externals import joblib

from flask import request

import flask
import optparse

from bottle import route, run, template, request, get, post
import requests
# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/classify', methods=['GET'])
def classifier():
    # # name = flask.request.args.get('name', '')
    #
    # name = request.args.get('name')
    # print(name)


    clf = joblib.load('classify.model')
    cate_dict = joblib.load('cate_dict.dat')
    vectorizer = joblib.load('vectorizer.dat')

    # cate_id_name_dict = dict(map(lambda (k, v): (v, k), cate_dict.items()))


    print("classify called")
    # img = request.GET.get('img','')
    # name = request.GET.get('name', '')
    # pred = clf.predict(vectorizer.transform([name]))[0]
    # print(cate_dict[pred])
    # return {'cate'}


    # logging.info('Image: %s', name)
    # result = app.clf.classify_image(image)

    return flask.jsonify("test")


def setModel() :
    measures = collocations.BigramAssocMeasures()
    twitter = Twitter()

    path = "./soma_classifier.csv"
    train_df = pd.read_csv(path)
    # train_df = pd.read_pickle("soma_goods_train.df")

    # nltk.download("stopwords")
    # nltk.download("punkt")
    # nltk.download("maxent_treebank_pos_tagger")
    # nltk.download('maxent_treebank_pos_tagger')

    # nltk.download("all")

    # pattern = r'''(?x) ([A-Z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`]'''
    # 학습하는 상품 name들을 리스트로 만든다.
    d_list = []
    # 카테고리 분류명에 구분자(';')를 줘서 리스트를 만든다
    cate_list = []


    # iterrows() : 행의 이름과 값들을 쌍으로 조회
    for each in train_df.iterrows():
        # join() : ';'를 구분자로 내용들을 합친다
        cate = ";".join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])

        d_list.append(each[1]['name'])
        cate_list.append(cate)
    # print(type(d_list[0]))
    print(d_list)
    # print(len(d_list))
    # words =[]
    # print('Collocations among tagged words:')
    # for d in d_list :
    #     words.append(' '.join(twitter.nouns(d)))
    #
    # print(words)
    pattern = r'''(?x) ([A-Za-z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`]'''

    pat = r'[a-zA-Z]+'
    # pat = r'[a - zA - Z]+'
    words=[]
    words_kor = []
    # print('Collocations among tagged words:')
    for d in d_list :
        tokens = word_tokenize(d)
        # tokens = d.split()
        # print(tokens)
        # print(twitter.nouns(d))
        words_kor.append(' '.join(twitter.nouns(d)))
        words.append(' '.join(tokens))

    # print(words)
    # print(words_kor)
    # print(len(words_kor))
    words_en = []


    for word in words:
        temp = []
        # print("word : " + word)
        tokens_en = regexp_tokenize(word, pat)
        print(tokens_en)
        temp.append(' '.join(tokens_en))
        for temp in tokens_en:
            # print("temp : " + temp)
            # tagged_word = nltk.pos_tag(temp.split())
            tagged_word = nltk.pos_tag(word_tokenize(temp))
            print(tagged_word)
            nouns = [token for token, pos in tagged_word if pos.startswith('N')]
            # print(nouns)
            words_en.append(nouns)

        # words_en.append(temp)


        # nouns = [word for word, pos in tagged_sent if pos == 'Noun']
        # print(nouns)
        # words_en.append(nouns)

    # print(words_en)
    # print(len(words_en))
    # for word in words_en :
    #     print(word)
        # print(word[0])

        # tagged_word = nltk.pos_tag(word[0].split())
        # nouns = [token for token, pos in tagged_word if pos.startswith('N')]
        # words_en.append(' '.join(nouns))


    # print(len(words_en))
    # print(len(words))
    # 여기까지!!!!!!~~~~~~~~~~~!~!~!~!~!@~!@~!@~!@!~@!@~!~!~!~!~!~!~!~!~!~!~!~!~!~!!!!!!!~~~~~~~~~~~!~!~!~!~!@~!@~!@~!@!~@!@~!~!~!~!~!~!~!~!~!~!~!~!~!~!


    # 같은종류를 묶어서 하나로...group by와 같다!!
    # print(set(cate_list))
    # object to list
    # print(list(set(cate_list)))
    # 각 카테고리명에 대해서 serial 한 숫자 id를 부여한다.
    # cate_dict[카테고리명] = serial_id   형태이다.
    # dict() : value : key(id); 형태로 저장이 된다,

    cate_dict = dict(zip(list(set(cate_list)), range(len(set(cate_list)))))

    # print(cate_dict), {'디지털/가전;PC부품;CPU' : O, '패션의류;아동의류;한복':1}
    # print(cate_dict['디지털/가전;PC부품;CPU']), 0

    # 단어 형태의 카테고리명에 대응되는 serial_id값들을 넣는다.
    y_list = []
    for each in train_df.iterrows():
        cate = ";".join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])
        y_list.append(cate_dict[cate])

    # print(y_list)
    # print(len(y_list))

    # 각 상품별로 name의 문장에 있는 단어들을 빈도수를 matrix형태로 만든다.
    # vectorizer = CountVectorizer()
    # x_list = vectorizer.fit_transform(d_list)
    # x_list = vectorizer.fit_transform(words)

    # print(x_list.shape)
    # print(x_list)
    # 단어들 출력
    # for x in x_list:
    #     print(x.indices)
    #     print(x.data)
    #     print(x.indptr)

        # for word in x.indices :
        #     doc = vectorizer.get_feature_names()[word]
        #     print(doc)


    # vectorizer100 = CountVectorizer(max_features=100)
    # x100_list = vectorizer100.fit_transform(d_list)
    # print(set(x100_list))

    # print(len(vectorizer100.get_feature_names()))

    # 단어들 출력
    # for x10 in x100_list:
    #     for word in x10.indices :
    #         temp = kkma.pos(vectorizer100.get_feature_names()[word])

    # svc_param = {'C': np.logspace(-2, 0, 20)}
    # # svc_param = {'C': np.logspace(-2, 0, 5)}
    # #
    # gs_svc = GridSearchCV(LinearSVC(loss='l2'), svc_param, cv=5, n_jobs=4)
    # gs_svc.fit(x_list, y_list)
    #
    # print(gs_svc.best_params_, gs_svc.best_score_)
    #
    #
    # clf = LinearSVC(C=gs_svc.best_params_['C'])
    # clf.fit(x_list, y_list)
    #
    # joblib.dump(clf, 'classify.model', compress=3)
    # joblib.dump(cate_dict, 'cate_dict.dat', compress=3)
    # joblib.dump(vectorizer, 'vectorizer.dat', compress=3)
    return

def test():

    clf = joblib.load('classify.model')
    cate_dict = joblib.load('cate_dict.dat')
    vectorizer = joblib.load('vectorizer.dat')

    joblib.dump(clf, 'n_classify.model')
    joblib.dump(cate_dict, 'n_cate_dict.dat')
    joblib.dump(vectorizer, 'n_vectorizer.dat')

    # name = request.GET.get('name', '')
    # name = "조끼"

    cate_id_name_dict = dict(map(lambda k, v: v, k, cate_dict.items()))
    pred = clf.predict(vectorizer.transform(['[신한카드5%할인][서우한복] 아동한복 여자아동 금나래 (분홍)']))[0]


    name = '[신한카드5%할인][예화-좋은아이들] 아동한복 여아 1076 빛이나노랑'
    img = ''
    u = 'http://localhost:8887/classify?name=%s&img=%s'
    r = requests.get(u % (name, img)).json()
    print(r)

    print(pred)
    print(cate_id_name_dict[pred])
    return

def flaskrun(app, default_host="127.0.0.1",
                  default_port="3000"):
    """
    Takes a flask.Flask instance and runs it. Parses
    command-line flags to configure the app.
    """

    # Set up the command-line options
    parser = optparse.OptionParser()
    parser.add_option("-H", "--host",
                      help="Hostname of the Flask app " + \
                           "[default %s]" % default_host,
                      default=default_host)
    parser.add_option("-P", "--port",
                      help="Port for the Flask app " + \
                           "[default %s]" % default_port,
                      default=default_port)

    # Two options useful for debugging purposes, but
    # a bit dangerous so not exposed in the help message.
    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug",
                      help=optparse.SUPPRESS_HELP)
    parser.add_option("-p", "--profile",
                      action="store_true", dest="profile",
                      help=optparse.SUPPRESS_HELP)

    options, args = parser.parse_args()

    # If the user selects the profiling option, then we need
    # to do a little extra setup
    if options.profile:
        from werkzeug.contrib.profiler import ProfilerMiddleware

        app.config['PROFILE'] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app,
                       restrictions=[30])
        options.debug = True

    app.run(
        debug=options.debug,
        host=options.host,
        port=int(options.port)
    )

setModel()

# test()



# lan = socket.gethostbyname(socket.gethostname())
# print("ip : " + lan)

#     server start
# if __name__ == '__main__':
#     flaskrun(app)