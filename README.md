# housing-price-forecasting

이 프로젝트는 부동산 실거래 데이터의 고질적인 문제인 이벤트 기반 희소성(Event-driven Sparsity)을 해결하고, 딥러닝 모델의 시퀀스 학습 효율을 극대화하기 위한 데이터 구성 프레임워크를 제안합니다. 도메인 지식 기반의 VAHI(Volatility-Adjusted Hybrid Imputation) 알고리즘을 통해 파편화된 데이터를 연속적인 시계열로 복원합니다.

------------------------------------------------------------------------

### Project Structure

-   **data folder**: Contains all input and output files for analysis.
    -   **Apart Deal.csv**: 국내 주택 거래 데이터. (https://www.kaggle.com/datasets/brainer3220/korean-real-estate-transaction-data)
    -   **법정동 기준 시군구 단위.csv**: 매칭 목적 시군구 코드.
    -   **지하철 역세권 지가지수.csv**: 국내 지하철 지수 데이터
    -   **subway_index_wide_fixed.csv**: 지하철 지수 1차 전처리.
    -   **subway_index_wide_fixed_2.csv**: 지하철 지수 2차 수동 처리.
    -   **전국초중등학교위치표준데이터.csv**: 국내 학교 데이터
    -   **school_counts_processed.csv**: 학교 데이터 1차 전처리.
    -   **school_counts_sigungu.csv**: 학교 데이터 시군구 단위 2차 전처리.
    -   **apart_master_final_2026.csv**: 전처리 데이터 통합 및 처리. (코드 통해 생성 필요)
    -   **apart_master_final_2026_2.csv**: 통합 전처리 데이터 변수명 수정. (코드 통해 생성 필요)
    -   **apart_master_final_2026_3.csv**: 통합 전처리 데이터 변수 정렬. (코드 통해 생성 필요)
    -   **apart_master_final_2026_4.csv**: 통합 전처리 데이터 샘플링. (코드 통해 생성 필요)
    -   **apart_master_final_2026_5.csv**: 통합 전처리 데이터 샘플ID 추가. (코드 통해 생성 필요)
    -   **apart_fixed_master.csv**: 통합 전처리 데이터 최종 필터링. (코드 통해 생성 필요)
    -   **region_mapping.csv**: 통합 전처리 데이터 맵핑 파일.
    -   **sido_*.csv**: 통합 전처리 데이터 시도 분리 (e.g., sido_서울특별시.csv, sido_부산광역시.csv).

-   **models folder**: Stores trained model weights and state dictionaries (.pth files).

-   **source files**:
    -   **1. preprocessing.ipynb**: Scripts for data merging and cleaning.
    -   **2. estate price LGBM.ipynb**: LightGBM forecasting analysis.
    -   **3. estate price XGBoost.ipynb**: XGBoost-based forecasting model.
    -   **4. estate price LSTM.ipynb**: Time-series modeling using RNN-LSTM.
    -   **5. estate price GRU.ipynb**: Gated Recurrent Unit for sequence learning.
    -   **6. estate price Transformer.ipynb**: Attention-based price forecasting.
    -   **Readme.md**: This document.

------------------------------------------------------------------------

### Prerequisites

To run these scripts, you'll need the following environments and libraries installed.

#### Python Environment

- Python 3.9+

- PyTorch (GPU support recommended)

- Scikit-learn, Pandas, Numpy

- XGBoost, LightGBM

``` python
pip install torch pandas numpy scikit-learn xgboost lightgbm
```

------------------------------------------------------------------------

### Workflow and Manual Steps

1.  **Data Cleaning & Standardization**
    -   500만 건의 실거래 로그를 정제하고 '부천시' 등 행정구역 통합 로직을 적용합니다.
2.  **Feature Engineering**
    -   지하철 역세권 지가지수를 시계열로 정렬하고, 지역별 학교 수를 집계하여 공간적 특징량을 생성합니다.
3.  **Timeline Expansion**
    -   각 샘플(sample_id)별로 100개월의 타임라인을 생성합니다.
4.  **VAHI Imputation**
    -   VAHI 알고리즘을 적용하여 지역 변동성과 개별 선형성을 결합한 하이브리드 가격 보간을 수행합니다.
5.  **Model Training & Evaluation**
    -   Sample-wise Split(7:1:2) 전략을 사용하여 자산의 개별성을 유지한 채 예측 성능을 측정합니다.

------------------------------------------------------------------------
