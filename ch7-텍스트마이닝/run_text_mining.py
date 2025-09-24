
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import re
import pickle
import json
import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 경고 메시지 끄기
warnings.filterwarnings('ignore')

# --- 1. 데이터 로딩 및 전처리 (NSMC 데이터) ---
print("--- 1. NSMC 데이터 로딩 및 전처리 시작 ---")
nsmc_train_df = pd.read_csv('./ratings_train.txt', encoding='utf-8', sep='	', engine='python')
nsmc_test_df = pd.read_csv('./ratings_test.txt', encoding='utf-8', sep='	', engine='python')

# Null 값 제거
nsmc_train_df = nsmc_train_df.dropna(how = 'any')
nsmc_test_df = nsmc_test_df[nsmc_test_df["document"].notnull()]

# 한글만 남기기
nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x: re.sub(r'[^ ㄱ - | 가-힣]+'," ",x))
nsmc_test_df["document"]= nsmc_test_df["document"].apply(lambda x: re.sub(r'[^ ㄱ - | 가-힣]+'," ",x))
print("--- 1. NSMC 데이터 로딩 및 전처리 완료 ---")

# --- 2. Okt 형태소 분석기 설정 ---
print("--- 2. Okt 형태소 분석기 설정 시작 ---")
okt = Okt()
def okt_tokenizer(text):
  tokens = okt.morphs(text)
  return tokens
print("--- 2. Okt 형태소 분석기 설정 완료 ---")

# --- 3. TF-IDF 모델 로딩 ---
# 모델이 이미 학습되어 저장되어 있다고 가정하고 로드합니다.
# 만약 모델 파일이 없다면, 이 부분을 주석 해제하고 tfidf.fit()을 실행해야 합니다.
print("--- 3. TF-IDF 모델 로딩 시작 ---")
tfidf_model_save_path = "./tfidf_model.pkl"
try:
    with open(tfidf_model_save_path, "rb") as file:
        tfidf = pickle.load(file)
    print(f"✅ TF-IDF 모델 불러오기 완료: {tfidf_model_save_path}")
except FileNotFoundError:
    print(f"⚠️ TF-IDF 모델 파일 '{tfidf_model_save_path}'을(를) 찾을 수 없습니다. 모델을 새로 학습합니다.")
    # 모델 학습 (시간이 오래 걸릴 수 있습니다)
    tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
    tfidf.fit(nsmc_train_df["document"])
    with open(tfidf_model_save_path, "wb") as file:
        pickle.dump(tfidf, file)
    print(f"✅ TF-IDF 모델 학습 및 저장 완료: {tfidf_model_save_path}")
print("--- 3. TF-IDF 모델 로딩 완료 ---")

# --- 4. 감성 분류 모델 로딩 ---
# 모델이 이미 학습되어 저장되어 있다고 가정하고 로드합니다.
print("--- 4. 감성 분류 모델 로딩 시작 ---")
model_save_path = "./SA_lr_best.pkl"
try:
    with open(model_save_path, "rb") as file:
        SA_lr_best = pickle.load(file)
    print(f"✅ 감성 분류 모델 불러오기 완료: {model_save_path}")
except FileNotFoundError:
    print(f"⚠️ 감성 분류 모델 파일 '{model_save_path}'을(를) 찾을 수 없습니다. 모델을 새로 학습합니다.")
    # 모델 학습 (시간이 오래 걸릴 수 있습니다)
    SA_lr = LogisticRegression(random_state=0)
    nsmc_train_tfidf = tfidf.transform(nsmc_train_df["document"])
    SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])
    
    # GridSearchCV 부분 (선택 사항, 시간이 매우 오래 걸림)
    # params = {"C" : [1,3,3.5, 4, 4.5, 5]}
    # SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv = 3, scoring="accuracy", verbose=1)
    # SA_lr_grid_cv.fit(nsmc_train_tfidf,nsmc_train_df["label"])
    # SA_lr_best = SA_lr_grid_cv.best_estimator_
    
    SA_lr_best = SA_lr # GridSearchCV를 건너뛰고 기본 모델 사용
    
    with open(model_save_path, "wb") as file:
        pickle.dump(SA_lr_best, file)
    print(f"✅ 감성 분류 모델 학습 및 저장 완료: {model_save_path}")
print("--- 4. 감성 분류 모델 로딩 완료 ---")

# --- 5. 모델 평가 ---
print("--- 5. 모델 평가 시작 ---")
nsmc_test_tfidf = tfidf.transform(nsmc_test_df["document"])
test_predict = SA_lr_best.predict(nsmc_test_tfidf)
print("감성 분석 정확도 : ", round(accuracy_score(nsmc_test_df["label"], test_predict),3))
print("--- 5. 모델 평가 완료 ---")

# --- 6. 사용자 입력 감성 분석 ---
print("--- 6. 사용자 입력 감성 분석 시작 ---")
# st = input("감성 분석하기위한 문장을 입력 해주세요: ") # Jupyter Notebook에서만 사용
st = "이 영화 정말 재미있네요!" # 테스트를 위한 예시 문장
print(f"입력 문장: {st}")

st = re.compile(r"[ㄱ - | 가-힣]+").findall(st)
st = [" ".join(st)]

st_tfidf = tfidf.transform(st)
st_predict = SA_lr_best.predict(st_tfidf)

if(st_predict == 0):
  print(st," -> 부정")
else:
  print(st," -> 긍정")
print("--- 6. 사용자 입력 감성 분석 완료 ---")

# --- 7. 코로나 뉴스 데이터 분석 및 워드클라우드 준비 ---
print("--- 7. 코로나 뉴스 데이터 분석 및 워드클라우드 준비 시작 ---")
file_name = '코로나_naver_news'
with open("./"+file_name+'.json',encoding='utf-8') as j_f:
  data = json.load(j_f)

data_title = []
data_description = []

for item in data:
  data_title.append(item["title"])
  data_description.append(item["description"])

data_df = pd.DataFrame({"title":data_title, "description" : data_description})

# 제목, 내용 한글만 남기기
data_df["title"]= data_df["title"].apply(lambda x: re.sub(r'[^ ㄱ - | 가-힣]+'," ",x))
data_df["description"]= data_df["description"].apply(lambda x: re.sub(r'[^ ㄱ - | 가-힣]+'," ",x))

# 파일 csv로 저장 (선택 사항)
data_df.to_csv("./"+file_name+".csv", encoding="utf-8")

# 'title_label'과 'description_label' 컬럼 생성
# TF-IDF 변환
data_df_title_tfidf = tfidf.transform(data_df["title"])
data_df_description_tfidf = tfidf.transform(data_df["description"])

# 감성 예측
data_df["title_label"] = SA_lr_best.predict(data_df_title_tfidf)
data_df["description_label"] = SA_lr_best.predict(data_df_description_tfidf)

print("제목 감성 분포:\n", data_df["title_label"].value_counts())
print("내용 감성 분포:\n", data_df["description_label"].value_counts())

columns_name = ['title','title_label','description','description_label']
NEG_data_df = pd.DataFrame(columns=columns_name)
POS_data_df = pd.DataFrame(columns=columns_name)

for i, data in data_df.iterrows():
    title = data["title"]
    description = data["description"]
    t_label = data["title_label"]
    d_label = data["description_label"]
    
    if d_label == 0: # 부정 감성 샘플만 추출
        new_data_df = pd.DataFrame([[title, t_label, description, d_label]], columns=columns_name)
        NEG_data_df = pd.concat([NEG_data_df, new_data_df], ignore_index=True)
    else : # 긍정 감성 샘플만 추출
        new_data_df = pd.DataFrame([[title, t_label, description, d_label]], columns=columns_name)
        POS_data_df = pd.concat([POS_data_df, new_data_df], ignore_index=True)

NEG_data_df.to_csv('./'+file_name+'_NES.csv', encoding='utf-8')
POS_data_df.to_csv('./'+file_name+'_POS.csv', encoding='utf-8')

print(len(NEG_data_df), len(POS_data_df))

POS_description = POS_data_df['description']
POS_description_noun_tk = []
for d in POS_description:
    POS_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출

POS_description_noun_join = []
for d in POS_description_noun_tk:
    d2 = [w for w in d if len(w) > 1] #길이가 1인 토큰은 제외
    POS_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성

NEG_description = NEG_data_df['description']
NEG_description_noun_tk = []
for d in NEG_description:
    NEG_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출
    
NEG_description_noun_join = []
for d in NEG_description_noun_tk:
    d2 = [w for w in d if len(w) > 1]  #길이가 1인 토큰은 제외
    NEG_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성

print("--- 7. 코로나 뉴스 데이터 분석 및 워드클라우드 준비 완료 ---")

# --- 8. Matplotlib 및 WordCloud 폰트 설정 ---
print("--- 8. Matplotlib 및 WordCloud 폰트 설정 시작 ---")
# Windows 기본 한글 폰트 경로
font_location = 'C:/Windows/Fonts/malgun.ttf'
try:
    font_name = fm.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)
    print(f"✅ Matplotlib 폰트 설정 완료: {font_name}")
except Exception as e:
    print(f"❌ Matplotlib 폰트 설정 오류: {e}. 'malgun.ttf' 폰트가 없거나 경로가 잘못되었을 수 있습니다.")
    print("다른 폰트 경로를 시도하거나 폰트를 설치해야 할 수 있습니다.")

max_words_to_display = 15 # 바 차트에 나타낼 단어의 수

# --- 9. 긍정/부정 뉴스 단어 빈도 시각화 (바 차트) ---
print("--- 9. 긍정/부정 뉴스 단어 빈도 시각화 시작 ---")
POS_tfidf_wc = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
POS_dtm_wc = POS_tfidf_wc.fit_transform(POS_description_noun_join)
POS_vocab_wc = dict()
for idx, word in enumerate(POS_tfidf_wc.get_feature_names_out()):
    POS_vocab_wc[word] = POS_dtm_wc.getcol(idx).sum()
POS_words_wc = sorted(POS_vocab_wc.items(), key=lambda x: x[1], reverse=True)

NEG_tfidf_wc = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
NEG_dtm_wc = NEG_tfidf_wc.fit_transform(NEG_description_noun_join)
NEG_vocab_wc = dict()
for idx, word in enumerate(NEG_tfidf_wc.get_feature_names_out()):
    NEG_vocab_wc[word] = NEG_dtm_wc.getcol(idx).sum()
NEG_words_wc = sorted(NEG_vocab_wc.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(max_words_to_display), [i[1] for i in POS_words_wc[:max_words_to_display]], color="blue")
plt.title(f"긍정 뉴스의 단어 상위 {max_words_to_display}개", fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max_words_to_display), [i[0] for i in POS_words_wc[:max_words_to_display]], rotation=70)

plt.subplot(1, 2, 2)
plt.bar(range(max_words_to_display), [i[1] for i in NEG_words_wc[:max_words_to_display]], color="red")
plt.title(f"부정 뉴스의 단어 상위 {max_words_to_display}개", fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max_words_to_display), [i[0] for i in NEG_words_wc[:max_words_to_display]], rotation=70)

plt.tight_layout()
plt.show()
print("--- 9. 긍정/부정 뉴스 단어 빈도 시각화 완료 ---")

# --- 10. LDA 토픽 모델링 및 시각화 ---
print("--- 10. LDA 토픽 모델링 및 시각화 시작 ---")
description = data_df['description']
description_noun_tk = []
for d in description:
    description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출

description_noun_tk2 = []
for d in description_noun_tk:
    item = [i for i in d if len(i) > 1]  #토큰의 길이가 1보다 큰 것만 추출
    description_noun_tk2.append(item)

dictionary = corpora.Dictionary(description_noun_tk2)
# print(dictionary[1]) # 작업 확인용 출력

corpus = [dictionary.doc2bow(word) for word in description_noun_tk2]
# print(corpus) # 작업 확인용 출력

k = 4  #토픽의 개수 설정
lda_model = gensim.models.ldamulticore.LdaMulticore(corpus, iterations = 12, num_topics = k, id2word = dictionary, passes = 1, workers = 10)

print(lda_model.print_topics(num_topics = k, num_words = 15))

# LDA 모델 저장 (선택 사항)
# with open("lda_model.pkl", "wb") as f:
#     pickle.dump(lda_model, f)

# pyLDAvis 시각화
lda_vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_vis)
print("--- 10. LDA 토픽 모델링 및 시각화 완료 ---")

# --- 11. WordCloud 생성 (예시) ---
# WordCloud는 노트북 환경에서 이미지를 직접 보여주므로, 스크립트에서는 파일로 저장하는 예시를 제공합니다.
print("--- 11. WordCloud 생성 시작 ---")
# 모든 명사 토큰을 하나의 문자열로 합치기
all_nouns = ' '.join(POS_description_noun_join + NEG_description_noun_join)

# WordCloud 객체 생성 시 폰트 경로 지정
# font_location 변수는 위에서 정의된 'C:/Windows/Fonts/malgun.ttf'를 사용합니다.
wordcloud = WordCloud(
    font_path=font_location,
    background_color='white',
    width=800,
    height=600,
    max_words=100,
    stopwords=STOPWORDS # STOPWORDS는 wordcloud 라이브러리에서 제공하는 기본 불용어
).generate(all_nouns)

# 워드클라우드 이미지 파일로 저장
wordcloud_image_path = "./wordcloud_output.png"
wordcloud.to_file(wordcloud_image_path)
print(f"✅ WordCloud 이미지 저장 완료: {wordcloud_image_path}")

# 이미지 표시 (선택 사항, Jupyter 환경에서만 직접 표시됨)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
print("--- 11. WordCloud 생성 완료 ---")
