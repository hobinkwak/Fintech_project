from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from marcap import marcap_data
import numpy as np
import re
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from hmmlearn import hmm
import FinanceDataReader as fdr
import json
from ast import literal_eval
from gensim.models import Word2Vec, KeyedVectors
from statsmodels.tsa.stattools import coint
import requests
from bs4 import BeautifulSoup


krx = fdr.StockListing('KRX')
symbol_name_krx = krx[['Symbol','Name']]
code_to_stock = dict(zip(symbol_name_krx.Symbol, symbol_name_krx.Name))
stock_to_code = dict(zip(symbol_name_krx.Name, symbol_name_krx.Symbol))
db = json.load(open('data/theme_db.json', 'r'))
user_db = json.load(open('data/user_db.json', 'r'))


# ************ 목 차 ************

# 1. Data 생성함수 (util함수)
# 2. 문서요약
# 3. HMM주가패턴
# 4. 종목추천

# *****************************


#-----------------------------------------------------------------
#---------------------1. Data 생성함수---------------------------------------
#-----------------------------------------------------------------

def make_theme_db():
    df1 = pd.read_html('http://m.infostock.co.kr/sector/sector.asp?mode=w')[1][[0]]
    df2 = pd.read_html('http://m.infostock.co.kr/sector/sector.asp?mode=w')[1][[2]]
    db = pd.DataFrame(np.append(df1.values, df2.values), columns=['theme'])
    db.theme = db.theme.apply(lambda x : x[2:])
    db = db.sort_values(by='theme', ignore_index=True)
    db = {key : [] for key in db.theme}

    for i in range(0, 1000):
        try:
            df = pd.read_html(f'http://m.infostock.co.kr/sector/sector_detail.asp?code={i}&theme=2%uCC28%uC804%uC9C0&mode=w')[1]
            theme = df.iloc[0,1]
            df = df[3:]
            df.columns = df.iloc[0].to_list()
            df = df.drop(3).reset_index(drop=True)
            db[theme] = list(df.관련종목.map(lambda x: x[1:-8].strip()).values)
        except:
            continue
    db_keys = list(db.keys())
    for key in db_keys:
        if db[key] == []:
            del db[key]
    return db

def make_user_theme_db():
    mydb = json.load(open('data/theme_db.json','r'))
    random_keys = np.random.choice(list(mydb.keys()), size=5, replace=False)
    user_db = {key : mydb[key] for key in random_keys}
    for key in user_db:
        random_size= np.random.randint(2,4)
        user_db[key] = list(np.random.choice(np.array(user_db[key]), replace=False, size=min(random_size, len(user_db[key]))))
    return user_db

def total_record(cprc):
    return (stock_code, cprc)

def make_stock_vectors():
    """Stock2Vec용 데이터 만들기"""
    krx_list=krx['Symbol'].to_list()
    krx_dict = dict(zip(krx.Symbol, krx.Name))

    data = None
    index = 0
    for stock_code in krx_list:
        try:
            temp_df = fdr.DataReader(stock_code, '2000-01-01', '2020-12-31')['Close']
        except:
            continue
        if len(temp_df) == 0:
            continue
        temp_df = temp_df.pct_change()
        temp_df = temp_df.dropna()
        temp_df = pd.DataFrame(temp_df)
        temp_df.columns.values[0] = 'rate'
        temp_df['rate'] = temp_df['rate'].map(lambda x : [total_record(x)])
        if index == 0:
            data = pd.DataFrame(temp_df.copy())
            index = index+1
        else:
            data = data.append(temp_df)
            index = index+1
    #data.to_csv('data/stock_return.csv')
    #data = pd.read_csv('data/stock_return.csv', parse_dates=True, converters={'rate':literal_eval})
    train_data=data.groupby(['Date']).sum()
    train_data=pd.DataFrame(train_data)
    train_data.columns.values[0] = 'rate'
    train_data['rate'] = train_data['rate'].map(lambda x : sorted(x, key=lambda x: x[1], reverse=True))
    train_data['rate'] = train_data['rate'].map(lambda x : [i[0] for i in x])
    train_list=train_data.values.tolist()
    train_list = [item[0] for item in train_list]

    stock_vectors=[]
    for line in train_list:
        temp1=[]
        for code in line:
            temp1.append(krx_dict[code])
        stock_vectors.append(temp1)
    np.save('data/stock_vectors.npy',np.array(stock_vectors, dtype=object), allow_pickle=True)
    stock_vectors = list(np.load('stock_vectors.npy', allow_pickle=True))
    return stock_vectors

#-----------------------------------------------------------------
#---------------------2. 문서요약---------------------------------------
#-----------------------------------------------------------------

class SentenceTokenizer(object): 
    def __init__(self): 
        self.kkma = Kkma()
        self.twitter = Okt()
        self.stopwords = ['중인','만큼','마찬가지','꼬집었',"연합뉴스","데일리","동아일보","중앙일보","조선일보","기자","아","휴","아이구","아이쿠","아이고","어","나","우리","저희","따라","의해","을","를","에","의","가"]
        
        
    def url2sentences(self, url): 
        article = Article(url, language = 'ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)
        for idx in range(0, len(sentences)) : 
            if len(sentences[idx]) <= 10 : 
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = '' 
        return sentences 
    
    def text2sentences(self, text) : 
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)) : 
            if len(sentences[idx]) <= 10 : 
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = '' 
        return sentences 
                        
    def get_nouns(self, sentences) : 
        nouns = [] 
        for sentence in sentences : 
            if sentence != '' : 
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence)) if noun not in self.stopwords and len(noun) > 1]) )
        return nouns
    
class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence
    
    
class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]

            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}
    
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
 
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])        
        return summary

def crawler(company_name, maxpage):
    df_result = None    
    page = 1 
    company_code = stock_to_code[company_name]
    while page <= int(maxpage): 
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page) 
        source_code = requests.get(url).text
        html = BeautifulSoup(source_code, "lxml") 
        titles = html.select('.title')
        title_result=[]
        for title in titles: 
            title = title.get_text() 
            title = re.sub('\n','',title)
            title_result.append(title)
        links = html.select('.title') 
        link_result =[]
        for link in links: 
            add = 'https://finance.naver.com' + link.find('a')['href']
            link_result.append(add)
        result= {"기사제목" : title_result, "링크" : link_result} 
        df_temp = pd.DataFrame(result)
        if df_result is not None:
            df_result = pd.concat([df_result, df_temp])
        else:
            df_result = df_temp
        page += 1
    return df_result

def TuDDaVillage_summarizer(company):
    """ 사용자의 보유아이템(테마) 중 기사를 보고싶은 테마를 넣어주세요 """
    company = user_db[company]
    randint = np.random.choice(np.arange(0,len(company)), replace=False, size=min(3, len(company)))
    company = [company[ri] for ri in randint]
    for i, comp in enumerate(company):
        result = crawler(comp, 2)
        if i ==0:
            result_sample = result.iloc[np.random.choice(np.arange(0, len(result)), size=1, replace=False)]
        else:
            tmp = result.iloc[np.random.choice(np.arange(0, len(result)), size=1, replace=False)]
            result_sample = pd.concat([result_sample, tmp], axis=0)
    
    for i in range(len(result_sample)):
        print(f"****{company[i]} 기사****")
        print()
        print("제목 :", result_sample.iloc[i].values[0])
        print()
        textrank = TextRank(result_sample.iloc[i].values[1])
        for row in textrank.summarize(3):
            print(row)
        if i == len(result_sample)-1:
            continue
        else:
            print('\n\n')
    return
           

#-----------------------------------------------------------------
#---------------------3. HMM주가패턴---------------------------------------
#-----------------------------------------------------------------
        
def up_down_flat(company):
    """ 사용자의 보유아이템(테마) 중 주가패턴을 보고싶은 테마를 넣어주세요 """
    company = user_db[company]
    if len(company) == 1:
        company = company[0]
        df = fdr.DataReader(stock_to_code[company])
        price = df.Close
        df['log_return'] = np.log(price) - np.log(price.shift(5))
        df = df.dropna()
        data = np.array(df['log_return'])
        data = np.reshape(data,(data.shape[0], 1))
    
    else:
        company_data = krx.loc[krx['Name'].isin(company)][['Name','ListingDate']]
        newest_date = company_data.sort_values(by='ListingDate').values[-1,1]
        for i, ticker in enumerate(company):
            if i ==0:
                close_df = fdr.DataReader(stock_to_code[ticker], start = newest_date, end=newest_date.today())[['Close']]
                marcap_df = marcap_data(start = newest_date, end=newest_date.today(), code=stock_to_code[ticker])[['Marcap']] 
                df = pd.concat([close_df,marcap_df], axis=1)
            else:
                close_tmp = fdr.DataReader(stock_to_code[ticker], start = newest_date, end=newest_date.today())[['Close']]
                marcap_tmp = marcap_data(start = newest_date, end=newest_date.today(), code=stock_to_code[ticker])[['Marcap']] 
                tmp = pd.concat([close_tmp, marcap_tmp], axis=1)
                df = pd.concat([df, tmp], axis=1)
        df = df.dropna()
        df.columns = list(range(len(company)*2))
        total = np.round(df[list(range(1, len(company)*2, 2))].values / df[list(range(0, len(company)*2, 2))].values).sum(axis=1)
        price = df[list(range(1, len(company)*2, 2))].sum(axis=1) / total
        log_return = (np.log(price) - np.log(price.shift(5))).dropna()
        log_return = np.array(log_return)
        data = np.reshape(log_return, (log_return.shape[0], 1))
        
    num_state = 3
    model = hmm.GaussianHMM(n_components=num_state, tol=0.0005, n_iter=100000)
    model.fit(data)    
    pred_hidden_state = model.predict(data)
    
    def transformer(x):
        if x == np.argmin(mu.squeeze()):
            return 0
        elif x == np.argmax(mu.squeeze()):
            return 2
        else:
            return 1
        
    mu = np.array(model.means_) * 52
    sigma = np.array([np.sqrt(x) for x in model.covars_]) * np.sqrt(52)
    sigma = np.reshape(sigma, (num_state, 1))

    pred_hidden_state = np.array(list(map(transformer, pred_hidden_state)))
    tmp = sorted(list(zip(mu.squeeze(), sigma.squeeze())), key=lambda tup: tup[0])
    mu = np.array([tup[0] for tup in tmp]).reshape(num_state,1)
    sigma = np.array([tup[1] for tup in tmp]).reshape(num_state,1)
    
    
    print(company)
    print("\nmu :\n", mu)
    print("\nvol :\n", sigma)
    
    #히든
    plt.figure(figsize=(10,5))
    plt.plot(pred_hidden_state[-300:], 'r-')
    plt.grid(True, alpha=0.5)
    plt.show()

    #주가
    plt.figure(figsize=(10,5))
    plt.plot(price[-300:])
    plt.grid(True, alpha=0.5)
    plt.show()

    #히든놈의 분포
    plt.figure(figsize=(10,5))
    for i in range(num_state):
        x = np.linspace(mu[i] - 4*sigma[i], mu[i]+4*sigma[i], 100)
        plt.plot(x, sp.stats.norm.pdf(x, mu[i], sigma[i]), linewidth=3, label = 'State : '+str(i))
    plt.title("Hidden States (" + str(num_state) + " States)", fontsize=15)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.legend(loc = 'upper left', fontsize=15)
    plt.grid(True)
    plt.show()

    # 각 State 비중
    for i in range(num_state):
        prob = len(np.where(pred_hidden_state==i)[0]) / len(pred_hidden_state)
        print(f"State {i} : {prob * 100}"+'%')
    
    return pred_hidden_state, price
        

#-----------------------------------------------------------------
#---------------------4. 종목추천---------------------------------------
#-----------------------------------------------------------------
    
def stock_recommendation(company):
    stock_vectors = list(np.load('data/stock_vectors.npy', allow_pickle=True))
    model = Word2Vec(sentences=stock_vectors, size=100, window=5, min_count=1, workers=4, sg=1)
    model.wv.save_word2vec_format('stock2vec_skipgram_tmp')

    try:
        result=model.wv.most_similar(company, topn=50)
        similar_stocks = [tup[0] for tup in result]
        similarity = [tup[1] for tup in result]
        result = dict(result)

        selected_stocks = {}
        a = fdr.DataReader(stock_to_code[company])[['Close']]['2016-01-01':]
        for key in db:
            if company in db[key]:
                selected_stocks[key] = [] 
                for stock in similar_stocks:
                    if stock in db[key]:
                        b = fdr.DataReader(stock_to_code[stock])[['Close']]
                        corr = pd.concat([a, b], axis=1).corr().iloc[0,1]
                        if corr > 0:
                            selected_stocks[key].append(stock)
                if selected_stocks[key] == []:
                    del selected_stocks[key]
    except:
        selected_stocks = {}
        a = fdr.DataReader(stock_to_code[company])[['Close']]['2011-01-01':]
        for key in db:
            if company in db[key]:
                selected_stocks[key] = [] 
                ls = list(map(lambda x:stock_to_code[x], db[key]))
                for x in ls:
                    if code_to_stock[x] == company: continue
                    b = fdr.DataReader(x)[['Close']]
                    total = pd.concat([a, b], axis=1).dropna()
                    total.columns = ['a','b']
                    _, pvalue, _ = coint(total['a'], total['b'])
                    if pvalue < 0.05:
                        selected_stocks[key].append(code_to_stock[x])
    return selected_stocks