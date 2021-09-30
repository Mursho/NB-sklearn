# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle

import jieba
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
stop_word_ls=["(",")","（","）","▪","•",".","&","/","广州","广东","-","广州市"]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def word_cut(path):
    text_with_spaces=""
    text_file=open(path,'r',encoding='utf-8').read()
    text_cut=jieba.cut_for_search(text_file)
    for word in text_cut:
        if word != '·' and word !=" " and word!='\n' and word not in stop_word_ls:
            text_with_spaces+=word
            text_with_spaces+=" "
    return text_with_spaces


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # shenghuoriyong=word_cut("shenghuoriyong.txt")
    # wenjiaoyule=word_cut("wenjiaoyule.txt")
    # yinshi=word_cut("yinshi.txt")
    # yundongjiankang=word_cut("yundongjiankang.txt")
    # fushimeirong=word_cut("fushimeirong.txt")
    # jiaotongchuxing=word_cut("jiaotongchuxing.txt")
    # bunch=Bunch(label=[],contents=[])
    # bunch.label.append("交通出行")
    # bunch.contents.append(jiaotongchuxing)
    # bunch.label.append("生活日用")
    # bunch.contents.append(shenghuoriyong)
    # bunch.label.append("文教娱乐")
    # bunch.contents.append(wenjiaoyule)
    # bunch.label.append("饮食")
    # bunch.contents.append(yinshi)
    # bunch.label.append("运动健康")
    # bunch.contents.append(yundongjiankang)
    # bunch.label.append("服饰美容")
    # bunch.contents.append(fushimeirong)
    # tfidfspace=Bunch(label=bunch.label,tdm=[],vocabulary={})
    # vectorizer=TfidfVectorizer(sublinear_tf=True)
    # tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
    # tfidfspace.vocabulary=vectorizer.vocabulary_
    # print(tfidfspace.vocabulary)
    # with open("tfdifspace.dat",'wb') as f:
    #     pickle.dump(tfidfspace,f)


    with open("tfdifspace.dat",'rb')as f:
        model=pickle.load(f)
    clf=MultinomialNB(alpha=0.001).fit(model.tdm,model.label)


    test="团建"
    str=" ".join(jieba.cut(test))
    print(str)
    bunch=Bunch(content=[])
    bunch.content.append(str)
    tfidfspace=Bunch(tdm=[],vocabulary={})
    tfidfspace.vocabulary=model.vocabulary
    vectorizer=TfidfVectorizer(sublinear_tf=True,vocabulary=model.vocabulary)
    tfidfspace.tdm=vectorizer.fit_transform(bunch.content)

    predicted=clf.predict(tfidfspace.tdm)


    print(predicted)


    # shenghuoriyong_ls=shenghuoriyong.split(" ")
    # wenjiaoyule_ls=wenjiaoyule.split(" ")
    # yinshi_ls=yinshi.split(" ")
    # yundongjiankang_ls=yundongjiankang.split(" ")
    # fushimeirong_ls=fushimeirong.split(" ")
    # dataset=[]
    # dataset.append(shenghuoriyong_ls)
    # dataset.append(wenjiaoyule_ls)
    # dataset.append(yinshi_ls)
    # dataset.append(yundongjiankang_ls)
    # dataset.append(fushimeirong_ls)
    # print(dataset)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
