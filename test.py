
# feture를 기존 저장되어 있는 data와 비교하여 가장 유사도가 높은 ID를 찾아내서 분류한다
#
#


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# SVM을 이용해서 분류학습한다
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
import numpy as np



def test() :

    path = "./soma_classifier1.csv"
    train_df = pd.read_csv(path)
    # print(train_df)

    # 학습하는 상품 name들을 리스트로 만든다.
    d_list = []
    # 카테고리 분류명에 구분자(';')를 줘서 리스트를 만든다
    cate_list = []

    # iterrows() : 행의 이름과 값들을 쌍으로 조회
    for each in train_df.iterrows():
        # print(each);
        # join() : ';'를 구분자로 내용들을 합친다
        cate = ";".join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])
        d_list.append(each[1]['name'])
        cate_list.append(cate)

    # print(d_list)
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


    # 각 상품별로 name의 문장에 있는 단어들을 빈도수를 matrix형태로 만든다.
    vectorizer = CountVectorizer()
    x_list = vectorizer.fit_transform(d_list)

    # print(x_list)

    # 단어들 출력
    # for x in x_list:
    #     for word in x.indices :
    #         print(vectorizer.get_feature_names()[word])

    vectorizer10 = CountVectorizer(max_features=10)
    x10_list = vectorizer10.fit_transform(d_list)

    # print(len(vectorizer10.get_feature_names()))
    # # 단어들 출력
    # for x10 in x10_list:
    #     for word in x10.indices :
    #         print(vectorizer10.get_feature_names()[word])


    svc_param = {'C': np.logspace(-2, 0, 20)}
    gs_svc = GridSearchCV(LinearSVC(loss='l2'), svc_param, cv=5, n_jobs=4)
    gs_svc.fit(x_list, y_list)
    print(gs_svc.best_params_, gs_svc.best_score_)

    return

test()
