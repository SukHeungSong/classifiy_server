import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer



def test() :

    path = "./soma_classifier.csv"
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
    # 같은종류를 묶어서 하나로...group by와 같다!!
    # print(set(cate_list))
    # object to list
    # print(list(set(cate_list)))

    # 각 카테고리명에 대해서 serial 한 숫자 id를 부여한다.
    # cate_dict[카테고리명] = serial_id   형태이다.
    # dict() : value : key(id); 형태로 저장이 된다,

    cate_dict = dict(zip(list(set(cate_list)), range(len(set(cate_list)))))

    # print(cate_dict)
    # print(cate_dict['디지털/가전;PC부품;CPU'])

    # 각 상품별로 name의 문장에 있는 단어들을 빈도수를 matrix형태로 만든다.
    vectorizer = CountVectorizer()

    # 단어 형태의 카테고리명에 대응되는 serial_id값들을 넣는다.
    y_list = []
    for each in train_df.iterrows():
        cate = ";".join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])
        y_list.append(cate_dict[cate])

    return

test()