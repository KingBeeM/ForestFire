# ForestFire (~ 2023.06.23)
<br/>

## [1. Streamlit 웹 서비스](https://kingbeem-forestfire-app-zxbk0n.streamlit.app/ "Streamlit Link")<br/>

## [2. 개인 ipynb 파일](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/pdf/storesales.ipynb/ ".ipynb Link")<br/>

## [3. PDF 파일](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/pdf/StoreSales_Attention.pdf/ "PDF Link")<br/>

---
![image](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/main-store.png)

## ✔ 목적
Kaggle Store Sales 대회에서 시계열 데이터를 사용하여 Corporación Favorita라는 대규모 에콰도르 **식료품 소매업체 매출 예측** 및 Streamlit 활용 **웹 서비스 구현**<br/>

### [Kaggle Store Sales 대회](https://www.kaggle.com/competitions/store-sales-time-series-forecasting, "Kaggle Link") <br/>

## ✔ 데이터
Store Sales 대회에서 제공하는 데이터를 사용하였습니다.<br/>

### [Store Sales Data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data, "Data Link") <br/>

## ✔ ERD
![image](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/STORESALES_EDR.png)
<br/>

## ✔ 팀 구성
- 사용언어 : Python 3.10.10
- 작업툴 : VS Code
- 인원 : 4명
- 주요 업무 : Streamlit 라이브러리 활용 웹 서비스 구현 및 머신러닝을 활용한 매장 매출 예측
- 기간 : 2023.05.01 ~ 2023.05.17

## ✔ 주요 기능
- **INTRO 페이지**
  - Store Sales 대회에 대한 소개, 목표, 분석 단계에 대한 설명
- **DATA 페이지**
  - Store Sales 대회에서 제공하는 데이터에 대한 정보와 기술통계량 제공

![image1](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/github-data.png)
- **Exploratory Data Analysis**
  - 시간 지남에 따른 상점별 매출 시각화
  - 시간 지남에 따른 제품군별 매출 시각화
  - 주/월별 지연값에 대한 매출 시각화
  - 주/월별 매출에 대한 시각화
  - 시간 지남에 따른 상점별 거래량 시각화
  - 월별 거래량 시각화
  - 매출과 거래량 간 상관분석 시각화
  - 요일별 거래량 시각화
  - 시간 지남에 따른 유가 시각화
  - 유가와 매출 및 거래량 간 상관분석 및 시각화
  - 유가와 제품군별 매출 간 상관분석 및 시각화

![image2](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/github-eda.png)
- **STAT**
  - 상관분석에 대한 설명 및 지연값들 간 상관분석 시각화
  - ACF / PACF에 대한 설명 및 제품군별 ACF / PACF 시각화
  - Features(추세, 계절성, 지연값)에 대한 설명 및 시각화
  - 주/월 단순이동평균선 및 지수이동평균선 시각화

![image3](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/stat.png)
- **ML**
  - RandomForest 모델을 사용한 상점별 매출 및 거래량 예측

![image4](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/ML.png)
## ✔ 설치 방법

### Windows
- 버전 확인
  - VS Code / PyCharm : 
    - Python: 3.10.9
  - 라이브러리 : 
    - beautifulsoup4: 4.11.1
    - bs4: 0.0.1
    - db-dtypes: 1.1.1
    - Flask: 2.2.2
    - folium: 0.14.0
    - geopandas: 0.13.0
    - google-cloud-bigquery: 3.11.0
    - googlemaps: 4.10.0
    - keras: 2.12.0
    - lxml: 4.9.1
    - matplotlib: 3.7.0
    - missingno: 0.5.2
    - numpy: 1.23.5
    - opencv-python: 4.7.0.72
    - pandas: 1.5.3
    - pandas-gbq: 0.19.2
    - pingouin: 0.5.3
    - plotly: 5.9.0
    - scikit-learn: 1.2.1
    - seaborn: 0.12.2
    - selenium: 4.8.3
    - shapely: 2.0.1
    - statsmodels: 0.13.5
    - streamlit: 1.20.0
    - streamlit-option-menu: 0.3.5
    - streamlit-pandas-profiling: 0.1.3
    - tensorflow: 2.12.0
    - torch: 2.0.0
    - torchvision: 0.15.1
    - tornado: 6.1
    - tqdm: 4.64.1
    - xgboost: 1.7.5

- 프로젝트 파일을 다운로드 받습니다.
```
git clone https://github.com/KingBeeM/ForestFire.git
```
- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 10 기준)
```
virtualenv venv
source venv/Scripts/activate
```
- 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```
- streamlit 명령어를 실행합니다.
```
streamlit run app.py
```