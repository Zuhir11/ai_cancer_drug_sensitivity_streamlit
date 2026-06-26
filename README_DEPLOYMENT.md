# Yapay Zeka Kanser İlaç Duyarlılığı


Bu proje, kanser hücre hatlarında ilaç duyarlılığı tahmini için geliştirilmiş Streamlit tabanlı bir yapay zeka arayüzüdür.

## Proje yapısı



## Türkçe

Not: Bazı CSV dosyaları Excel ile doğrudan açıldığında tüm veriler ilk hücrede tek bir sütun olarak görünebilir. Bu durumda dosyanın hatalı olduğu anlamına gelmez. Kullanıcılar Excel’de “Veri” menüsündeki “Metni Sütunlara Dönüştür” seçeneğini kullanarak veya dosyayı uygun ayırıcı karakterle içe aktararak verileri okunabilir sütunlar halinde düzenleyebilir.

## English

Note: When some CSV files are opened directly with Excel, all data may appear in the first cell or in a single column instead of being separated into columns. This does not mean that the file is corrupted. Users can fix the layout by using the “Text to Columns” option in Excel or by importing the file with the correct delimiter so that the data is displayed in readable columns.





```text
yapay_zeka_kanser_ilac_duyarliligi/
│
├── app.py
├── requirements.txt
├── README_DEPLOYMENT.md
│
├── data/
│   ├── cell_drug_selection_list.csv
│   ├── known_predictions.csv
│   └── cell_profiles.csv
│
└── results/
    ├── preprocessing/
    ├── final_mlp_model/
    ├── model_performance/
    │   └── final_model_performance_comparison.csv
    │
    ├── xgboost_shap_explainability/
    │   └── plots/
    │       ├── xgboost_shap_feature_family_importance.png
    │       └── xgboost_top20_shap_features.png
    │
    ├── lightgbm_shap_table/
    │   └── tables/
    │       └── lightgbm_shap_feature_family_importance.csv
    │
    ├── gnn_explainability/
    │   └── permutation_importance/
    │       └── gnn_permutation_importance.csv
    │
    ├── mlp_permutation_importance/
    │   └── tables/
    │       └── mlp_permutation_importance.csv
    │
    └── mlp_diagnostics/
```

## Notlar

- Büyük `tam_fusion_egitim_veri_seti.csv` dosyası bu projeye eklenmemiştir.
- Cloud sürümünde hafif veri dosyaları kullanılmaktadır.
- Nihai tahmin modeli: MLP Neural Network.
- XGBoost SHAP görselleri, XGBoost Regressor modeli için açıklanabilirlik analizi sunar.
- Nihai MLP modeli için açıklanabilirlik, MLP Permutation Importance bölümü ile verilmiştir.
- Bu sistem yalnızca araştırma ve akademik gösterim amacıyla hazırlanmıştır.


## Full Dataset and Source Files

The Streamlit Cloud application uses lightweight application-ready data files.

The full processed fusion training dataset and source files are available on Kaggle:

https://www.kaggle.com/datasets/zuhirmarwan/ai-cancer-drug-sensitivity-multi-omics-data/data

Original data source: Cell Model Passports / Sanger.  
Research and educational use only.


## Contact

LinkedIn:

https://www.linkedin.com/in/zuhir-marwan-62641538b/
