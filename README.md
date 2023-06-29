# ForestFire (~ 2023.06.23)
<br/>

## [1. Streamlit Service](https://kingbeem-forestfire-app-zxbk0n.streamlit.app/ "Streamlit Link")<br/>

## [2. Personal Code](https://github.com/KingBeeM/ForestFire/tree/main/file/code/ ".Code Link")<br/>

## [3. Deep Learning](https://github.com/KingBeeM/ForestFire/tree/main/file/code/DL_EfficientNet/ ".DL Link")<br/>

## [4. PDF](https://github.com/KingBeeM/ForestFire/tree/main/file/ppt/Forestfire.pdf/ "PDF Link")<br/>

---

## ✔ 목적
Gangwon-do Forest Fire Prediction and Damage Minimization Project: Model Development Using Machine Learning and Deep Learning
<br/>

## ✔ 데이터
| 제공 사이트          | 제공 기관   | 한글 이름  | 사용 테이블 이름 |
|---------------------|------------|--------|-------------|
| 공공데이터포털      | 기상청     | 기상청_지상(종관, ASOS) 일자료 조회서비스 | weather_days|
| 공공데이터포털      | 기상청     | 기상청_관측지점정보 | weather_stations|
| 공공데이터포털      | 산림청    | 산림청_산불발생통계(대국민포털) | forestfire_occurs_add |
| 공공데이터포털      | 행정안전부 | 산불발생이력 | forestfire_occurs|
| 국가공간정보포털    | 국토교통부 | 행정구역_읍면동(법정동) | gangwon_UMD |
| 국가공간정보포털    | 국토교통부 | 행정구역시군구_경계 | gangwon_SSG |
| 행정표준코드관리시스템 | 국토교통부 | 행정구역_코드(법정동) | gangwon_code|
<br/>

## ✔ ERD
![image](https://github.com/KingBeeM/ForestFire/tree/main/file/img/ERD.png)
<br/>

## ✔ Flow Chart
![image](https://github.com/KingBeeM/ForestFire/tree/main/file/img/flowchart.png)
<br/>

## ✔ 팀 구성
- 사용언어 : Python
- 작업툴 : VS Code / PyCharm / Google Colab / Google BigQuery / QGIS / IBM SPSS Statistics
- 인원 : 4명
- 주요 업무 : 머신러닝과 딥러닝을 통한 강원도 산불예측모델 및 산불분류모델
- 기간 : 2023.05.22 ~ 2023.06.23

## ✔ 주요 기능
- **HOME**
  - 강원도 산불위험지수(DWI) 지도시각화
    - 강원도 지역을 9개로 나누어서 각각 지역에 대해 ML 모델을 만듬
    - 실시간 API 요청을 통한 각 지역의 산불위험지수(DWI)를 구함
    - 각 지역별 산불위험지수를 지도시각화를 통해 보여줌

![image1](https://github.com/KingBeeM/ForestFire/tree/main/file/img/HOME_img.png)
- **EDA**
  - 

![image1](https://github.com/KingBeeM/ForestFire/tree/main/file/img/EDA_img.png)
- **STAT**
  - 

![image3](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/model_img.png)
- **ML**
  - 

![image4](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/model_img.png)
- **DL**
  - EfficentNet-B7 모델을 사용한 산불 이미지 분류 모델

![image5](https://github.com/KingBeeM/storesales_streamlit_by_Kaggle/blob/main/img/DL.png)
## ✔ 설치 방법

### Windows
- 버전 확인
  - VS Code / PyCharm : 
```
  Python: 3.10.9
```
  - Google Colab
  - 라이브러리 : 
```
  beautifulsoup4: 4.11.1
  bs4: 0.0.1
  db-dtypes: 1.1.1
  Flask: 2.2.2
  folium: 0.14.0
  geopandas: 0.13.0
  google-cloud-bigquery: 3.11.0
  googlemaps: 4.10.0
  keras: 2.12.0
  lxml: 4.9.1
  matplotlib: 3.7.0
  missingno: 0.5.2
  numpy: 1.23.5
  opencv-python: 4.7.0.72
  pandas: 1.5.3
  pandas-gbq: 0.19.2
  pingouin: 0.5.3
  plotly: 5.9.0
  scikit-learn: 1.2.1
  seaborn: 0.12.2
  selenium: 4.8.3
  shapely: 2.0.1
  statsmodels: 0.13.5
  streamlit: 1.20.0
  streamlit-option-menu: 0.3.5
  streamlit-pandas-profiling: 0.1.3
  tensorflow: 2.12.0
  torch: 2.0.0
  torchvision: 0.15.1
  tqdm: 4.64.1
  xgboost: 1.7.5
```
- 프로젝트 파일을 다운로드 받습니다.
```
git clone https://github.com/KingBeeM/ForestFire.git
```
- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 11 기준)
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