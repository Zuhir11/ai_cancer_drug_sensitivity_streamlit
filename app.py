import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import glob
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy import sparse

import torch
import torch.nn as nn


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


# ============================================================
# STREAMLIT CLOUD READY RELATIVE PATHS
# ============================================================
# Bu bölüm Streamlit Cloud için düzenlendi.
# Artık C:\\Users\\... gibi bilgisayara özel yollar kullanılmıyor.
#
# Expected deployment folder structure:
#
# app.py
# data/
#   cell_drug_selection_list.csv
#   app_ready_fusion_dataset.csv
# results/
#   preprocessing/
#   final_mlp_model/
#   model_performance/
#   xgboost_shap_explainability/
#   lightgbm_shap_table/
#   gnn_explainability/
#   mlp_permutation_importance/
#   mlp_diagnostics/
#
# Not:
# tam_fusion_egitim_veri_seti.csv dosyası çok büyük olduğu için
# Streamlit Cloud için daha küçük bir dosya kullanılacaktır:
# data/app_ready_fusion_dataset.csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJE_KLASORU = BASE_DIR
DATA_KLASORU = os.path.join(BASE_DIR, "data")
SONUC_KLASORU = os.path.join(BASE_DIR, "results")

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/zuhirmarwan/ai-cancer-drug-sensitivity-multi-omics-data/data"
LINKEDIN_PROFILE_URL = "https://www.linkedin.com/in/zuhir-marwan-62641538b/"

# Full 1.5 GB fusion dataset is not used on Streamlit Cloud.
# Instead, the app uses lightweight prepared files:
# 1) Known predictions for all existing Cell Line + Drug combinations.
# 2) One representative full multi-omics profile for each Cell Line.
FUSION_VERI_YOLU = os.path.join(
    DATA_KLASORU,
    "unused_app_ready_fusion_dataset.csv"
)

KNOWN_PREDICTIONS_YOLU = os.path.join(
    DATA_KLASORU,
    "known_predictions.csv"
)

CELL_PROFILES_YOLU = os.path.join(
    DATA_KLASORU,
    "cell_profiles.csv"
)

ADIM02_KLASOR = os.path.join(SONUC_KLASORU, "preprocessing")
ADIM03C_KLASOR = os.path.join(SONUC_KLASORU, "final_mlp_model")
ADIM09_KLASOR = DATA_KLASORU

ARAYUZ_VERI_YOLU = os.path.join(
    ADIM09_KLASOR,
    "cell_drug_selection_list.csv"
)

MLP_MODEL_YOLU = os.path.join(
    ADIM03C_KLASOR,
    "modeller",
    "mlp_regressor_best_model.pt"
)

FINAL_PERFORMANS_YOLU = os.path.join(
    SONUC_KLASORU,
    "model_performance",
    "final_model_performance_comparison.csv"
)

LIGHTGBM_SHAP_FAMILY_YOLU = os.path.join(
    SONUC_KLASORU,
    "lightgbm_shap_table",
    "tables",
    "lightgbm_shap_feature_family_importance.csv"
)

XGBOOST_SHAP_GRAFIK_KLASOR = os.path.join(
    SONUC_KLASORU,
    "xgboost_shap_explainability",
    "plots"
)

SHAP_FEATURE_FAMILY_IMAGE_YOLU = os.path.join(
    XGBOOST_SHAP_GRAFIK_KLASOR,
    "xgboost_shap_feature_family_importance.png"
)

SHAP_TOP20_FEATURES_IMAGE_YOLU = os.path.join(
    XGBOOST_SHAP_GRAFIK_KLASOR,
    "xgboost_top20_shap_features.png"
)

GNN_IMPORTANCE_YOLU = os.path.join(
    SONUC_KLASORU,
    "gnn_explainability",
    "permutation_importance",
    "gnn_permutation_importance.csv"
)

MLP_IMPORTANCE_YOLU = os.path.join(
    SONUC_KLASORU,
    "mlp_permutation_importance",
    "tables",
    "mlp_permutation_importance.csv"
)

MLP_DIAGNOSTICS_KLASOR = os.path.join(
    SONUC_KLASORU,
    "mlp_diagnostics"
)

MLP_LOSS_CURVE_YOLU = os.path.join(
    MLP_DIAGNOSTICS_KLASOR,
    "01_mlp_loss_curve.png"
)

MLP_TRAIN_VAL_GAP_YOLU = os.path.join(
    MLP_DIAGNOSTICS_KLASOR,
    "02_mlp_train_validation_gap_loss.png"
)

MLP_ACTUAL_PREDICTED_YOLU = os.path.join(
    MLP_DIAGNOSTICS_KLASOR,
    "03_mlp_actual_vs_predicted_ln_ic50.png"
)

MLP_ERROR_DISTRIBUTION_YOLU = os.path.join(
    MLP_DIAGNOSTICS_KLASOR,
    "04_mlp_error_distribution.png"
)


st.set_page_config(
    page_title="Cancer Drug Sensitivity AI",
    page_icon="🔗",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0px;
        line-height: 1.15;
    }

    .main-subtitle {
        font-size: 18px;
        color: #486581;
        margin-top: 8px;
        margin-bottom: 5px;
        line-height: 1.45;
    }

    .section-card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 22px;
        border-left: 6px solid #2f80ed;
    }

    .info-card {
        background-color: #eef6ff;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #2f80ed;
        margin-bottom: 16px;
        line-height: 1.55;
    }

    .warning-card {
        background-color: #fff8e6;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #f2a900;
        margin-bottom: 16px;
        line-height: 1.55;
    }

    .success-card {
        background-color: #eefaf3;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #27ae60;
        margin-bottom: 16px;
        line-height: 1.55;
    }

    .small-card {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0px 2px 7px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        line-height: 1.5;
    }

    .footer-text {
        color: #627d98;
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 25px;
        color: #102a43;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #486581;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="main-title">{title}</div>
            <div class="main-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def info_box_tr_en(tr_text, en_text):
    st.markdown(
        f"""
        <div class="info-card">
            <b>Türkçe</b><br>
            {tr_text}
            <br><br>
            <b>English</b><br>
            {en_text}
        </div>
        """,
        unsafe_allow_html=True
    )


def warning_box_tr_en(tr_text, en_text):
    st.markdown(
        f"""
        <div class="warning-card">
            <b>Türkçe</b><br>
            {tr_text}
            <br><br>
            <b>English</b><br>
            {en_text}
        </div>
        """,
        unsafe_allow_html=True
    )


def success_box_tr_en(tr_text, en_text):
    st.markdown(
        f"""
        <div class="success-card">
            <b>Türkçe</b><br>
            {tr_text}
            <br><br>
            <b>English</b><br>
            {en_text}
        </div>
        """,
        unsafe_allow_html=True
    )


def external_links_section():
    st.markdown("## Veri Seti ve İletişim / Dataset and Contact")

    info_box_tr_en(
        "Uygulamanın hızlı çalışması için Streamlit Cloud sürümünde hafif veri dosyaları kullanılmaktadır. "
        "Tam işlenmiş fusion eğitim veri seti ve kaynak dosyalar Kaggle üzerinde paylaşılmıştır. "
        "Proje ve akademik iletişim için LinkedIn profili kullanılabilir.",
        "To keep the Streamlit Cloud application lightweight and fast, this version uses reduced application-ready data files. "
        "The full processed fusion training dataset and source files are shared on Kaggle. "
        "For project and academic contact, the LinkedIn profile can be used."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.link_button(
            "Kaggle Dataset / Full Data and Source Files",
            KAGGLE_DATASET_URL
        )

    with col2:
        st.link_button(
            "LinkedIn / Contact",
            LINKEDIN_PROFILE_URL
        )


def image_explanation_card(title, image_path, tr_text, en_text):
    st.markdown(f"### {title}")

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(
            "Görsel bulunamadı / Image not found:\n\n"
            + image_path
        )

    info_box_tr_en(tr_text, en_text)


def shap_original_graphics_section():
    st.markdown("---")

    st.markdown("## XGBoost SHAP Grafik Görselleri / XGBoost SHAP Plot Images")

    info_box_tr_en(
        "Bu bölümde gösterilen SHAP görselleri XGBoost Regressor modeli için hesaplanmıştır. "
        "Analizde test verisinden rastgele seçilen 1000 örnek kullanılmış ve XGBoost native pred_contribs=True yöntemi uygulanmıştır. "
        "Bu grafikler nihai MLP modeline ait değildir; XGBoost modelinin açıklanabilirlik analizini göstermektedir.",
        "The SHAP plots shown in this section were calculated for the XGBoost Regressor model. "
        "The analysis used 1000 randomly selected samples from the test set and applied the native XGBoost pred_contribs=True method. "
        "These plots do not belong to the final MLP model; they represent the explainability analysis of the XGBoost model."
    )

    image_explanation_card(
        "XGBoost SHAP Feature Family Importance",
        SHAP_FEATURE_FAMILY_IMAGE_YOLU,
        "Bu grafik, XGBoost Regressor modeli için özellik gruplarının toplam SHAP önemini göstermektedir. Sonuçlara göre Drug Fingerprint grubu XGBoost kararlarında en baskın özellik ailesidir. Proteomics, Gene Expression / Other, Drug Target ve RNA-PPI grupları da XGBoost modeline katkı sağlamaktadır.",
        "This plot shows the total SHAP importance of feature groups for the XGBoost Regressor model. According to the results, the Drug Fingerprint group is the dominant feature family in the XGBoost decisions. Proteomics, Gene Expression / Other, Drug Target, and RNA-PPI also contribute to the XGBoost model."
    )

    image_explanation_card(
        "XGBoost Top 20 SHAP Feature Importance",
        SHAP_TOP20_FEATURES_IMAGE_YOLU,
        "Bu grafik, XGBoost Regressor modeli için tekil özellikler arasında en yüksek ortalama mutlak SHAP değerine sahip ilk 20 özelliği göstermektedir. FP ile başlayan özellikler ilacın Morgan Fingerprint bileşenlerini temsil eder. Bu durum, XGBoost modelinde kimyasal yapı bilgisinin tahmin performansında önemli rol oynadığını göstermektedir.",
        "This plot shows the top 20 individual features with the highest mean absolute SHAP values for the XGBoost Regressor model. Features starting with FP represent Morgan Fingerprint components of the drug. This indicates that chemical structure information plays an important role in XGBoost prediction performance."
    )


def best_model_plot_analysis_section():
    st.markdown("---")

    st.markdown(
        "## En İyi Model Grafik Analizi / Best Model Plot Analysis"
    )

    info_box_tr_en(
        "Bu bölümde en iyi performansı veren MLP modelinin eğitim davranışı ve test seti tahmin kalitesi dört farklı grafik üzerinden incelenmektedir. "
        "Bu grafikler modelin öğrenme sürecini, overfitting riskini, gerçek-tahmin uyumunu ve hata dağılımını değerlendirmek için kullanılmıştır.",
        "This section analyzes the training behavior and test-set prediction quality of the best-performing MLP model using four diagnostic plots. "
        "These plots are used to evaluate the learning process, overfitting risk, actual-predicted agreement, and prediction error distribution."
    )

    image_explanation_card(
        "1. Loss Curve",
        MLP_LOSS_CURVE_YOLU,
        "Loss curve, modelin eğitim sürecinde hatasının nasıl değiştiğini gösterir. Train loss ve validation loss değerlerinin birlikte azalması modelin veriden öğrenme yaptığını gösterir. Bu çalışmada train loss düzenli olarak azalırken validation loss belirli bir seviyeden sonra daha yavaş değişmiştir. Bu durum modelin öğrenme yaptığını ve ilerleyen epochlarda train-validation farkının izlenmesi gerektiğini göstermektedir.",
        "The loss curve shows how the model error changes during training. A decrease in both training loss and validation loss indicates that the model is learning from the data. In this study, training loss decreased consistently, while validation loss changed more slowly after a certain point. This indicates successful learning and shows that the train-validation gap should be monitored in later epochs."
    )

    image_explanation_card(
        "2. Train - Validation Gap",
        MLP_TRAIN_VAL_GAP_YOLU,
        "Bu grafik validation loss ile train loss arasındaki farkı göstermektedir. Farkın zamanla büyümesi overfitting eğilimine işaret edebilir. Bu modelde belirli bir train-validation farkı oluşmasına rağmen test setinde elde edilen RMSE ve R² değerleri modelin güçlü genelleme performansı gösterdiğini desteklemektedir.",
        "This plot shows the difference between validation loss and training loss. An increasing gap may indicate a tendency toward overfitting. Although a certain train-validation gap appears in this model, the RMSE and R² values obtained on the test set support that the model still has strong generalization performance."
    )

    image_explanation_card(
        "3. Actual vs Predicted LN_IC50",
        MLP_ACTUAL_PREDICTED_YOLU,
        "Bu grafik gerçek LN_IC50 değerleri ile MLP modelinin tahmin ettiği LN_IC50 değerlerini karşılaştırmaktadır. Noktaların diyagonal çizgiye yakın olması modelin doğru tahmin yaptığını gösterir. Bu grafikte noktaların büyük kısmı diyagonal çizgi etrafında yoğunlaşmıştır. RMSE = 0.9887, MAE = 0.7253 ve R² = 0.8712 değerleri modelin güçlü tahmin performansını desteklemektedir.",
        "This plot compares the true LN_IC50 values with the LN_IC50 values predicted by the MLP model. Points close to the diagonal line indicate accurate predictions. In this plot, most points are concentrated around the diagonal line. The values RMSE = 0.9887, MAE = 0.7253, and R² = 0.8712 support the strong predictive performance of the model."
    )

    image_explanation_card(
        "4. Error Distribution",
        MLP_ERROR_DISTRIBUTION_YOLU,
        "Error distribution grafiği tahmin hatalarının dağılımını göstermektedir. Hataların sıfır etrafında yoğunlaşması modelin belirgin bir sistematik sapma göstermediğini ifade eder. Ortalama hatanın sıfıra yakın olması, modelin genel olarak dengeli tahminler yaptığını göstermektedir.",
        "The error distribution plot shows the distribution of prediction errors. Errors concentrated around zero indicate that the model does not show a strong systematic bias. A mean error close to zero suggests that the model generally makes balanced predictions."
    )

    st.success(
        "Türkçe:\n\n"
        "Bu dört grafik birlikte değerlendirildiğinde, MLP modelinin başarılı şekilde öğrendiği, test setinde güçlü tahmin performansı gösterdiği ve hata dağılımının büyük ölçüde dengeli olduğu görülmektedir. "
        "Train-validation gap belirli düzeyde overfitting eğilimi gösterebilse de nihai test performansı MLP modelinin en iyi model olarak seçilmesini desteklemektedir.\n\n"
        "English:\n\n"
        "When these four plots are evaluated together, the MLP model is shown to learn successfully, achieve strong predictive performance on the test set, and produce a largely balanced error distribution. "
        "Although the train-validation gap may indicate a certain degree of overfitting tendency, the final test performance supports selecting the MLP model as the best model."
    )


def additional_model_tests_section():
    st.markdown("---")

    st.markdown(
        "## Ek Model Deneyleri / Additional Model Experiments"
    )

    info_box_tr_en(
        "Nihai model seçimini güçlendirmek amacıyla MLP modeline ek olarak farklı model ve iyileştirme yaklaşımları da test edilmiştir. "
        "Bu kapsamda Optuna ile MLP hiperparametre optimizasyonu, MLP-LightGBM-XGBoost tabanlı ensemble denemesi, stacking yaklaşımı ve CatBoost modeli denenmiştir. "
        "Ayrıca PPI ağ bilgisini kullanmak amacıyla GNN/GAT/GraphSAGE tabanlı modeller de değerlendirilmiştir. "
        "Yapılan karşılaştırmalarda bu ek modeller mevcut MLP modelinin genel performansını aşamamış veya yalnızca çok sınırlı bir fark üretmiştir. "
        "Bu nedenle sistemde nihai tahmin modeli olarak MLP Neural Network korunmuştur.",
        "To strengthen the final model selection, several additional models and improvement strategies were tested in addition to the MLP model. "
        "These experiments included Optuna-based MLP hyperparameter optimization, an MLP-LightGBM-XGBoost ensemble experiment, a stacking approach, and the CatBoost model. "
        "In addition, GNN/GAT/GraphSAGE-based models were evaluated to incorporate PPI network information. "
        "In the comparative evaluations, these additional models did not outperform the existing MLP model overall or produced only a very limited difference. "
        "Therefore, the MLP Neural Network was retained as the final prediction model in the system."
    )

    st.markdown(
        """
        **Türkçe - Test edilen ek yaklaşımlar:**

        - Optuna tabanlı MLP hiperparametre optimizasyonu
        - MLP + LightGBM + XGBoost ensemble denemesi
        - Stacking meta-model yaklaşımı
        - CatBoost Regressor
        - GNN / GAT / GraphSAGE tabanlı PPI ağ deneyleri

        **English - Additional tested approaches:**

        - Optuna-based MLP hyperparameter optimization
        - MLP + LightGBM + XGBoost ensemble experiment
        - Stacking meta-model approach
        - CatBoost Regressor
        - GNN / GAT / GraphSAGE-based PPI network experiments
        """
    )

    success_box_tr_en(
        "Bu ek deneyler, MLP modelinin rastgele seçilmediğini; farklı makine öğrenmesi, derin öğrenme ve ağ tabanlı yöntemlerle karşılaştırıldıktan sonra nihai model olarak belirlendiğini göstermektedir.",
        "These additional experiments show that the MLP model was not selected arbitrarily; it was chosen as the final model after being compared with different machine learning, deep learning, and graph-based approaches."
    )


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def temiz_metin_mi(deger):
    if pd.isna(deger):
        return False

    metin = str(deger).strip()

    if metin == "":
        return False

    gecersizler = {
        "nan",
        "none",
        "null",
        "unknown",
        "bilinmiyor",
        "na",
        "n/a",
        "-",
        "--"
    }

    if metin.lower() in gecersizler:
        return False

    if len(metin) < 2:
        return False

    return True


def ilac_adi_gecerli_mi(deger):
    if not temiz_metin_mi(deger):
        return False

    metin = str(deger).strip()

    if re.fullmatch(r"\d+", metin):
        return False

    if re.fullmatch(r"[\d\.\-\_\s]+", metin):
        return False

    if not re.search(r"[A-Za-z]", metin):
        return False

    if len(metin) < 3:
        return False

    return True


def ln_ic50_aralik_bilgisi_goster():
    st.info(
        "Türkçe:\n\n"
        "LN_IC50 yorumu araştırma amaçlıdır; klinik eşik değeri değildir.\n\n"
        "Genel olarak daha düşük LN_IC50 değeri, hücre hattının ilaca karşı daha yüksek duyarlılık gösterebileceğini ifade eder. "
        "Daha yüksek LN_IC50 değeri ise daha düşük duyarlılık veya direnç eğilimi anlamına gelebilir.\n\n"
        "LN_IC50 < 0  → Yüksek duyarlılık\n"
        "0 – 2        → Orta-yüksek duyarlılık\n"
        "2 – 4        → Orta duyarlılık\n"
        "> 4          → Düşük duyarlılık veya direnç eğilimi\n\n"
        "English:\n\n"
        "The LN_IC50 interpretation is for research purposes only; it is not a clinical threshold.\n\n"
        "In general, a lower LN_IC50 value may indicate higher drug sensitivity of the cell line. "
        "A higher LN_IC50 value may indicate lower sensitivity or a tendency toward resistance.\n\n"
        "LN_IC50 < 0  → High sensitivity\n"
        "0 – 2        → Moderate-high sensitivity\n"
        "2 – 4        → Moderate sensitivity\n"
        "> 4          → Low sensitivity or resistance tendency"
    )


def tahmin_aciklamasi_goster():
    st.markdown("### Tahmin Açıklaması / Prediction Explanation")

    st.info(
        "Türkçe:\n\n"
        "Bu tahmin, ilacın kimyasal parmak izi özellikleri, hedef bilgisi, hedef yolak bilgisi "
        "ve seçilen hücre hattının multi-omics özellikleri kullanılarak eğitilmiş MLP modeli tarafından üretilmiştir.\n\n"
        "Hücre hattı adı model girdisi olarak doğrudan kullanılmamıştır; yalnızca doğru hücreye ait biyolojik özellikleri "
        "seçmek için kullanılmıştır.\n\n"
        "English:\n\n"
        "This prediction was generated by the trained MLP model using the drug chemical fingerprint features, "
        "target information, target pathway information, and the selected cell line's multi-omics profile.\n\n"
        "The cell line name was not directly used as a model input; it was only used to select the biological features "
        "belonging to the correct cell line."
    )


def ln_ic50_yorumu(deger):
    if deger < 0:
        return (
            "Yüksek duyarlılık",
            "High sensitivity",
            "Düşük LN_IC50 değeri, hücrenin ilaca daha duyarlı olabileceğini gösterir.",
            "A low LN_IC50 value indicates that the cell line may be more sensitive to the drug.",
            "success"
        )

    elif 0 <= deger < 2:
        return (
            "Orta-yüksek duyarlılık",
            "Moderate-high sensitivity",
            "Bu aralık, ilaca karşı belirgin bir duyarlılık eğilimi gösterebilir.",
            "This range may indicate a clear tendency toward drug sensitivity.",
            "success"
        )

    elif 2 <= deger < 4:
        return (
            "Orta duyarlılık",
            "Moderate sensitivity",
            "Bu aralık, orta düzeyde ilaç duyarlılığı olarak yorumlanabilir.",
            "This range can be interpreted as moderate drug sensitivity.",
            "warning"
        )

    else:
        return (
            "Düşük duyarlılık / direnç eğilimi",
            "Low sensitivity / resistance tendency",
            "Yüksek LN_IC50 değeri, ilaca karşı daha düşük duyarlılık veya direnç eğilimi gösterebilir.",
            "A high LN_IC50 value may indicate lower sensitivity or a tendency toward resistance.",
            "error"
        )


def yorum_mesaji_goster(ln_degeri):
    tr_baslik, en_baslik, tr_aciklama, en_aciklama, seviye = ln_ic50_yorumu(ln_degeri)

    mesaj = (
        f"**LN_IC50 Yorumu / LN_IC50 Interpretation**\n\n"
        f"**Türkçe:** {tr_baslik}. {tr_aciklama}\n\n"
        f"**English:** {en_baslik}. {en_aciklama}\n\n"
        f"Bu yorum araştırma amaçlıdır, klinik karar değildir.\n\n"
        f"This interpretation is for research purposes only and is not a clinical decision."
    )

    if seviye == "success":
        st.success(mesaj)
    elif seviye == "warning":
        st.warning(mesaj)
    else:
        st.error(mesaj)


def smiles_to_morgan_fp(smiles, n_bits=2048, radius=2):
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit yüklü değil. Lütfen RDKit kurunuz.")

    if not temiz_metin_mi(smiles):
        raise ValueError("SMILES değeri boş veya geçersiz.")

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        raise ValueError("Geçersiz SMILES formatı.")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    arr = np.zeros((n_bits,), dtype=np.float32)

    for i in range(n_bits):
        arr[i] = fp.GetBit(i)

    return arr


@st.cache_data
def load_csv(path):
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_known_predictions_data():
    df = pd.read_csv(
        KNOWN_PREDICTIONS_YOLU,
        dtype={
            "model_id": str,
            "DRUG_ID": str,
            "CELL_LINE_NAME": str,
            "DRUG_NAME_x": str,
            "TARGET": str,
            "TARGET_PATHWAY": str
        },
        low_memory=False
    )

    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["DRUG_ID"] = df["DRUG_ID"].astype(str).str.strip()

    return df


@st.cache_data
def load_cell_profiles_data():
    df = pd.read_csv(
        CELL_PROFILES_YOLU,
        dtype={
            "model_id": str,
            "CELL_LINE_NAME": str,
            "DRUG_ID": str,
            "DRUG_NAME_x": str,
            "TARGET": str,
            "TARGET_PATHWAY": str
        },
        low_memory=False
    )

    df["model_id"] = df["model_id"].astype(str).str.strip()

    return df


@st.cache_resource
def load_preprocessing_objects():
    sayisal_sutunlar = joblib.load(
        os.path.join(ADIM02_KLASOR, "sayisal_sutunlar.joblib")
    )

    kategorik_sutunlar = joblib.load(
        os.path.join(ADIM02_KLASOR, "kategorik_sutunlar.joblib")
    )

    onehot_encoder = joblib.load(
        os.path.join(ADIM02_KLASOR, "onehot_encoder.joblib")
    )

    if "DRUG_ID" in sayisal_sutunlar:
        sayisal_sutunlar = [col for col in sayisal_sutunlar if col != "DRUG_ID"]

    return sayisal_sutunlar, kategorik_sutunlar, onehot_encoder


@st.cache_resource
def load_mlp_scaler():
    aday_yollar = []

    aday_yollar.extend(
        glob.glob(os.path.join(ADIM03C_KLASOR, "**", "*scaler*.joblib"), recursive=True)
    )

    aday_yollar.extend(
        glob.glob(os.path.join(ADIM03C_KLASOR, "**", "*maxabs*.joblib"), recursive=True)
    )

    aday_yollar = list(dict.fromkeys(aday_yollar))

    for yol in aday_yollar:
        try:
            nesne = joblib.load(yol)

            if hasattr(nesne, "transform"):
                return nesne, yol
        except Exception:
            pass

    raise FileNotFoundError(
        "MLP scaler dosyası bulunamadı. adim03c_mlp klasörü içinde scaler joblib dosyasını kontrol ediniz."
    )


@st.cache_resource
def load_numeric_fill_values(sayisal_sutunlar):
    aday_dosyalar = [
        "sayisal_medianlar.joblib",
        "sayisal_median_degerleri.joblib",
        "numeric_medians.joblib",
        "sayisal_imputer_degerleri.joblib",
        "sayisal_fill_values.joblib",
        "median_values.joblib"
    ]

    for dosya in aday_dosyalar:
        yol = os.path.join(ADIM02_KLASOR, dosya)

        if os.path.exists(yol):
            try:
                obj = joblib.load(yol)

                if isinstance(obj, pd.Series):
                    return obj.reindex(sayisal_sutunlar).fillna(0.0)

                if isinstance(obj, dict):
                    return pd.Series(obj).reindex(sayisal_sutunlar).fillna(0.0)

                if isinstance(obj, np.ndarray):
                    return pd.Series(obj, index=sayisal_sutunlar).fillna(0.0)

            except Exception:
                pass

    return pd.Series(0.0, index=sayisal_sutunlar)


@st.cache_resource
def load_mlp_model(input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPRegressor(input_dim=input_dim).to(device)

    try:
        state_dict = torch.load(MLP_MODEL_YOLU, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MLP_MODEL_YOLU, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


@st.cache_data
def temiz_arayuz_verisi_yukle():
    df = pd.read_csv(
        ARAYUZ_VERI_YOLU,
        dtype={
            "model_id": str,
            "CELL_LINE_NAME": str,
            "DRUG_ID": str,
            "DRUG_NAME_x": str,
            "TARGET": str,
            "TARGET_PATHWAY": str
        },
        low_memory=False
    )

    gerekli_sutunlar = [
        "model_id",
        "CELL_LINE_NAME",
        "DRUG_ID",
        "DRUG_NAME_x",
        "TARGET",
        "TARGET_PATHWAY",
        "LN_IC50"
    ]

    mevcut = [col for col in gerekli_sutunlar if col in df.columns]
    df = df[mevcut].copy()

    df = df[df["CELL_LINE_NAME"].apply(temiz_metin_mi)]
    df = df[df["DRUG_NAME_x"].apply(ilac_adi_gecerli_mi)]
    df = df[df["DRUG_ID"].apply(temiz_metin_mi)]
    df = df[df["model_id"].apply(temiz_metin_mi)]

    df["CELL_LINE_NAME"] = df["CELL_LINE_NAME"].astype(str).str.strip()
    df["DRUG_NAME_x"] = df["DRUG_NAME_x"].astype(str).str.strip()
    df["DRUG_ID"] = df["DRUG_ID"].astype(str).str.strip()
    df["model_id"] = df["model_id"].astype(str).str.strip()

    df = df.drop_duplicates().reset_index(drop=True)

    return df


def tam_veriden_satir_bul(model_id, drug_id, sayisal_sutunlar, kategorik_sutunlar):
    """
    Cloud-light version.

    Bilinen Cell Line + Drug kombinasyonları için büyük fusion dosyası okunmaz.
    Bunun yerine önceden hesaplanmış known_predictions.csv dosyası kullanılır.
    """
    known_df = load_known_predictions_data()

    secilen = known_df[
        (known_df["model_id"].astype(str) == str(model_id))
        & (known_df["DRUG_ID"].astype(str) == str(drug_id))
    ]

    if len(secilen) > 0:
        return secilen.iloc[[0]].copy()

    return None


def tam_veriden_hucre_satiri_bul(model_id, sayisal_sutunlar, kategorik_sutunlar):
    """
    Cloud-light version.

    Yeni ilaç ve toplu tahmin için her Cell Line'a ait temsilci multi-omics profilini
    cell_profiles.csv dosyasından okur.
    """
    profiles_df = load_cell_profiles_data()

    secilen = profiles_df[
        profiles_df["model_id"].astype(str) == str(model_id)
    ]

    if len(secilen) > 0:
        return secilen.iloc[[0]].copy()

    return None


def yeni_ilac_satiri_hazirla(
    base_cell_row,
    drug_name,
    smiles,
    target,
    target_pathway,
    sayisal_sutunlar,
    kategorik_sutunlar
):
    yeni_satir = base_cell_row.copy()

    fp = smiles_to_morgan_fp(smiles, n_bits=2048, radius=2)

    for i in range(2048):
        col = f"FP_{i}"
        if col in yeni_satir.columns:
            yeni_satir[col] = fp[i]

    if "DRUG_NAME_x" in yeni_satir.columns:
        yeni_satir["DRUG_NAME_x"] = drug_name

    if "TARGET" in yeni_satir.columns:
        yeni_satir["TARGET"] = target

    if "TARGET_PATHWAY" in yeni_satir.columns:
        yeni_satir["TARGET_PATHWAY"] = target_pathway

    for col in kategorik_sutunlar:
        if col == "TARGET":
            yeni_satir[col] = target
        elif col == "TARGET_PATHWAY":
            yeni_satir[col] = target_pathway

    return yeni_satir


def satiri_mlp_girdisine_donustur(
    satir_df,
    sayisal_sutunlar,
    kategorik_sutunlar,
    onehot_encoder,
    scaler,
    numeric_fill_values
):
    sayisal_df = satir_df.reindex(columns=sayisal_sutunlar)

    for col in sayisal_sutunlar:
        sayisal_df[col] = pd.to_numeric(sayisal_df[col], errors="coerce")

    sayisal_df = sayisal_df.fillna(numeric_fill_values)
    sayisal_values = sayisal_df.to_numpy(dtype=np.float32)

    sayisal_sparse = sparse.csr_matrix(sayisal_values)

    if onehot_encoder is not None and len(kategorik_sutunlar) > 0:
        kategorik_df = satir_df.reindex(columns=kategorik_sutunlar).fillna("Bilinmiyor")
        kategorik_df = kategorik_df.astype(str)

        kategorik_sparse = onehot_encoder.transform(kategorik_df)

        X = sparse.hstack(
            [sayisal_sparse, kategorik_sparse],
            format="csr"
        )
    else:
        X = sayisal_sparse

    X_scaled = scaler.transform(X)
    X_dense = X_scaled.toarray().astype(np.float32)

    return X_dense


def mlp_tahmin_yap(X_dense, model, device):
    with torch.no_grad():
        x_tensor = torch.tensor(X_dense, dtype=torch.float32).to(device)
        pred = model(x_tensor).detach().cpu().numpy()[0]

    return float(pred)


def excel_template_olustur():
    template_df = pd.DataFrame(
        {
            "Drug_Name": ["NewDrugA", "NewDrugB"],
            "SMILES": ["CCO", "CCN(CC)CC"],
            "TARGET": ["EGFR", "MTOR"],
            "TARGET_PATHWAY": ["RTK signaling", "PI3K/MTOR signaling"]
        }
    )

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="Input_Template")

    output.seek(0)
    return output


def sonuc_excel_olustur(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction_Results")

    output.seek(0)
    return output


def guvenli_dosya_adi(metin):
    metin = str(metin).strip()
    metin = re.sub(r"[^A-Za-z0-9_\-]+", "_", metin)
    metin = re.sub(r"_+", "_", metin)
    return metin.strip("_")[:80]


def bilimsel_notlar_df_olustur():
    return pd.DataFrame(
        {
            "Language": [
                "Türkçe",
                "Türkçe",
                "Türkçe",
                "English",
                "English",
                "English"
            ],
            "Note": [
                "Bu rapor araştırma ve akademik gösterim amacıyla oluşturulmuştur.",
                "LN_IC50 yorumu klinik eşik değeri değildir ve tedavi kararı için kullanılamaz.",
                "Hücre hattı adı model girdisi olarak doğrudan kullanılmamıştır; seçilen hücre hattına ait multi-omics özellikleri kullanılmıştır.",
                "This report was generated for research and academic demonstration purposes.",
                "The LN_IC50 interpretation is not a clinical threshold and must not be used for treatment decisions.",
                "The cell line name was not directly used as a model input; the multi-omics features of the selected cell line were used."
            ]
        }
    )


def prediction_report_excel_olustur(report_df, notes_df=None):
    output = BytesIO()

    metadata_df = pd.DataFrame(
        {
            "Field": [
                "Report_Type",
                "Generated_At",
                "Model",
                "Research_Use_Only",
                "Clinical_Decision_System"
            ],
            "Value": [
                "Cancer Drug Sensitivity Prediction Report",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "MLP Neural Network",
                "Yes",
                "No"
            ]
        }
    )

    if notes_df is None:
        notes_df = bilimsel_notlar_df_olustur()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        report_df.to_excel(writer, index=False, sheet_name="Prediction_Report")
        metadata_df.to_excel(writer, index=False, sheet_name="Metadata")
        notes_df.to_excel(writer, index=False, sheet_name="Scientific_Notes")

    output.seek(0)
    return output


# ============================================================
# STARTUP FILE CHECKS
# ============================================================

def gerekli_dosyalari_kontrol_et():
    gerekli_dosyalar = [
        ARAYUZ_VERI_YOLU,
        KNOWN_PREDICTIONS_YOLU,
        CELL_PROFILES_YOLU,
        MLP_MODEL_YOLU,
        os.path.join(ADIM02_KLASOR, "sayisal_sutunlar.joblib"),
        os.path.join(ADIM02_KLASOR, "kategorik_sutunlar.joblib"),
        os.path.join(ADIM02_KLASOR, "onehot_encoder.joblib")
    ]

    eksik = [dosya for dosya in gerekli_dosyalar if not os.path.exists(dosya)]

    if len(eksik) > 0:
        st.error(
            "Gerekli dosyalar bulunamadı / Required files were not found.\n\n"
            "Lütfen Streamlit Cloud klasör yapısını kontrol edin.\n\n"
            + "\n".join(eksik)
        )
        st.stop()


gerekli_dosyalari_kontrol_et()

arayuz_df = temiz_arayuz_verisi_yukle()

sayisal_sutunlar, kategorik_sutunlar, onehot_encoder = load_preprocessing_objects()
scaler, scaler_yolu = load_mlp_scaler()
numeric_fill_values = load_numeric_fill_values(tuple(sayisal_sutunlar))

input_dim = len(sayisal_sutunlar)

if onehot_encoder is not None and len(kategorik_sutunlar) > 0:
    try:
        onehot_dim = len(onehot_encoder.get_feature_names_out(kategorik_sutunlar))
    except Exception:
        onehot_dim = len(onehot_encoder.get_feature_names(kategorik_sutunlar))
    input_dim += onehot_dim

mlp_model, device = load_mlp_model(input_dim)


st.sidebar.markdown("### 🔗 Cancer Drug Sensitivity AI")
st.sidebar.caption("Multi-omics prediction interface")
sayfa = st.sidebar.radio(
    "Sayfa seçiniz / Select page",
    [
        "Ana Sayfa / Home",
        "Tahmin / Prediction",
        "Yeni İlaç Tahmini / New Drug Prediction",
        "Toplu Tahmin / Batch Prediction",
        "Model Performansları / Model Performance",
        "Açıklanabilirlik / Explainability",
        "Proje Hakkında / About Project",
        "İletişim / Contact"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Araştırma amaçlıdır / Research use only.")
st.sidebar.write("Klinik karar sistemi değildir / Not a clinical decision system.")


if sayfa == "Ana Sayfa / Home":
    page_header(
        "🔗 Cancer Drug Sensitivity AI",
        "Multi-Omics, Machine Learning, Deep Learning and GNN-Based Drug Sensitivity Prediction System"
    )

    st.markdown(
        """
        Bu arayüz, kanser hücre hatlarında ilaç duyarlılığını analiz etmek için geliştirilmiştir.

        Sistem; klasik makine öğrenmesi, ağaç tabanlı modeller, derin öğrenme ve GNN tabanlı yaklaşımları
        karşılaştırmalı olarak değerlendirmektedir.

        **Kullanılan veri türleri:**

        - Drug Morgan Fingerprints
        - RNA Expression
        - Proteomics
        - Mutation
        - CNV
        - Drug Target
        - Target Pathway
        - PPI Network

        **Cloud veri yapısı:**

        Cloud sürümünde tüm Cell Line ve Drug seçenekleri korunmuştur. Büyük fusion dosyası yerine, önceden hazırlanmış hafif seçim, tahmin ve hücre profili dosyaları kullanılmaktadır.
        """
    )

    st.markdown("---")

    st.subheader("Multi-Omics, Machine Learning, Deep Learning and GNN-Based System")

    st.markdown(
        """
        This interface was developed to analyze drug sensitivity in cancer cell lines.

        The system evaluates and compares classical machine learning, tree-based models,
        deep learning models, and GNN-based approaches.

        **Data types used:**

        - Drug Morgan Fingerprints
        - RNA Expression
        - Proteomics
        - Mutation
        - CNV
        - Drug Target
        - Target Pathway
        - PPI Network

        **Cloud data structure:**

        In the Cloud version, all Cell Line and Drug options are preserved. Instead of the large fusion file, lightweight precomputed selection, prediction, and cell-profile files are used.
        """
    )

    st.warning(
        "Bu sistem yalnızca araştırma ve akademik gösterim amacıyla hazırlanmıştır. "
        "Klinik karar verme, tanı veya tedavi amacıyla kullanılamaz.\n\n"
        "This system is for research and academic demonstration only. "
        "It must not be used for clinical decision-making, diagnosis, or treatment."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cell Line", arayuz_df["CELL_LINE_NAME"].nunique())

    with col2:
        st.metric("Drug", arayuz_df["DRUG_NAME_x"].nunique())

    with col3:
        st.metric("Target", arayuz_df["TARGET"].nunique())

    with col4:
        st.metric("Sample", len(arayuz_df))


elif sayfa == "Tahmin / Prediction":
    page_header(
        "İlaç Duyarlılığı Tahmini / Drug Sensitivity Prediction",
        "Known cell line and drug combination prediction using the trained MLP model"
    )

    st.markdown(
        """
        Bu sayfada veri setinde mevcut olan Cell Line ve ilaç kombinasyonları için MLP modeli ile `LN_IC50` tahmini yapılır.
        """
    )

    st.markdown(
        """
        This page predicts `LN_IC50` using the MLP model for existing Cell Line and drug combinations in the dataset.
        """
    )

    ln_ic50_aralik_bilgisi_goster()

    cell_line_options = sorted(arayuz_df["CELL_LINE_NAME"].dropna().unique().tolist())

    secilen_cell_line = st.selectbox(
        "Cell Line seçiniz / Select Cell Line",
        cell_line_options
    )

    cell_df = arayuz_df[arayuz_df["CELL_LINE_NAME"] == secilen_cell_line].copy()

    drug_options = (
        cell_df[["DRUG_ID", "DRUG_NAME_x", "TARGET", "TARGET_PATHWAY"]]
        .drop_duplicates()
        .sort_values("DRUG_NAME_x")
        .reset_index(drop=True)
    )

    drug_options["label"] = drug_options["DRUG_NAME_x"].astype(str)

    duplicate_mask = drug_options["label"].duplicated(keep=False)

    drug_options.loc[duplicate_mask, "label"] = (
        drug_options.loc[duplicate_mask, "DRUG_NAME_x"].astype(str)
        + " | Target: "
        + drug_options.loc[duplicate_mask, "TARGET"].astype(str)
    )

    secilen_drug_label = st.selectbox(
        "İlaç seçiniz / Select Drug",
        drug_options["label"].tolist()
    )

    secilen_drug_satiri = drug_options[
        drug_options["label"] == secilen_drug_label
    ].iloc[0]

    secilen_drug_id = secilen_drug_satiri["DRUG_ID"]

    secilen_kayitlar = cell_df[
        cell_df["DRUG_ID"].astype(str) == str(secilen_drug_id)
    ].copy()

    if st.button("Tahmin Yap / Predict"):
        secilen_kayit = secilen_kayitlar.iloc[0]

        model_id = secilen_kayit["model_id"]
        drug_id = secilen_kayit["DRUG_ID"]

        with st.spinner("Tahmin hazırlanıyor / Preparing prediction..."):
            tam_satir = tam_veriden_satir_bul(
                model_id=model_id,
                drug_id=drug_id,
                sayisal_sutunlar=sayisal_sutunlar,
                kategorik_sutunlar=kategorik_sutunlar
            )

            if tam_satir is None:
                st.error(
                    "Final veri setinde bu kombinasyon bulunamadı.\n\n"
                    "This combination was not found in the final dataset."
                )
                st.stop()

            # Cloud version:
            # For known Cell Line + Drug combinations, prediction is loaded from
            # the precomputed lightweight file known_predictions.csv.
            # This avoids uploading and reading the original 1.5 GB fusion dataset.
            tahmin_ln_ic50 = float(tam_satir["Predicted_LN_IC50"].iloc[0])

        gercek_ln_ic50 = float(tam_satir["LN_IC50"].iloc[0])
        hata = abs(gercek_ln_ic50 - tahmin_ln_ic50)

        st.success("Tahmin tamamlandı / Prediction completed.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cell Line", str(secilen_kayit.get("CELL_LINE_NAME", "-")))

        with col2:
            st.metric("İlaç / Drug", str(secilen_kayit.get("DRUG_NAME_x", "-")))

        with col3:
            st.metric("Model", "MLP Neural Network")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Gerçek LN_IC50 / True LN_IC50", f"{gercek_ln_ic50:.4f}")

        with col5:
            st.metric("Tahmin LN_IC50 / Predicted LN_IC50", f"{tahmin_ln_ic50:.4f}")

        with col6:
            st.metric("Mutlak Hata / Absolute Error", f"{hata:.4f}")

        bilgi_df = pd.DataFrame(
            {
                "Alan / Field": [
                    "Drug Name",
                    "TARGET",
                    "TARGET_PATHWAY",
                    "True LN_IC50",
                    "Predicted LN_IC50",
                    "Absolute Error",
                    "Scaler"
                ],
                "Değer / Value": [
                    secilen_kayit.get("DRUG_NAME_x", "-"),
                    secilen_kayit.get("TARGET", "-"),
                    secilen_kayit.get("TARGET_PATHWAY", "-"),
                    f"{gercek_ln_ic50:.4f}",
                    f"{tahmin_ln_ic50:.4f}",
                    f"{hata:.4f}",
                    os.path.basename(scaler_yolu)
                ]
            }
        )

        st.markdown("### Sonuç Bilgileri / Result Information")
        st.dataframe(bilgi_df, use_container_width=True)

        yorum_mesaji_goster(tahmin_ln_ic50)
        tahmin_aciklamasi_goster()

        rapor_excel = prediction_report_excel_olustur(
            bilgi_df,
            bilimsel_notlar_df_olustur()
        )

        rapor_dosya_adi = (
            "prediction_report_"
            + guvenli_dosya_adi(secilen_kayit.get("CELL_LINE_NAME", "cell_line"))
            + "_"
            + guvenli_dosya_adi(secilen_kayit.get("DRUG_NAME_x", "drug"))
            + ".xlsx"
        )

        st.download_button(
            label="Tahmin Raporunu İndir / Download Prediction Report",
            data=rapor_excel,
            file_name=rapor_dosya_adi,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


elif sayfa == "Yeni İlaç Tahmini / New Drug Prediction":
    page_header(
        "Yeni İlaç Tahmini / New Drug Prediction",
        "New drug prediction using SMILES-based Morgan Fingerprints and existing cell line profiles"
    )

    st.markdown(
        """
        Bu bölümde Cell Line sistemdeki mevcut hücrelerden seçilir. Yeni ilaç için kullanıcı SMILES, TARGET ve TARGET_PATHWAY bilgilerini girer.
        """
    )

    st.markdown(
        """
        In this section, the Cell Line is selected from the existing known cell lines. For a new drug, the user provides SMILES, TARGET, and TARGET_PATHWAY.
        """
    )

    ln_ic50_aralik_bilgisi_goster()

    if not RDKIT_AVAILABLE:
        st.error(
            "RDKit yüklü değil. Yeni ilaç tahmini için RDKit gereklidir.\n\n"
            "RDKit is not installed. RDKit is required for new drug prediction."
        )
        st.code("conda install -c conda-forge rdkit openpyxl")
        st.stop()

    cell_line_options = sorted(arayuz_df["CELL_LINE_NAME"].dropna().unique().tolist())
    target_options = sorted(arayuz_df["TARGET"].dropna().astype(str).unique().tolist())
    pathway_options = sorted(arayuz_df["TARGET_PATHWAY"].dropna().astype(str).unique().tolist())

    secilen_cell_line = st.selectbox(
        "Cell Line seçiniz / Select Cell Line",
        cell_line_options,
        key="new_drug_cell"
    )

    drug_name = st.text_input("Drug Name", value="")
    smiles = st.text_area("SMILES", value="", height=100)

    st.info(
        "Türkçe:\n\n"
        "SMILES, ilacın kimyasal yapısını metin formatında temsil eden bir gösterimdir. "
        "Yeni ilaç tahmini için geçerli ve RDKit tarafından okunabilir bir SMILES değeri girilmelidir. "
        "SMILES bilgisi PubChem veya ChEMBL gibi kimyasal veri tabanlarından alınabilir.\n\n"
        "Örnekler:\n"
        "Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O\n"
        "5-Fluorouracil: O=C1NC(=O)C(F)=CN1\n"
        "Ethanol test örneği: CCO\n\n"
        "English:\n\n"
        "SMILES is a text-based representation of the chemical structure of a drug. "
        "For a new drug prediction, the entered SMILES must be valid and readable by RDKit. "
        "SMILES values can be obtained from chemical databases such as PubChem or ChEMBL.\n\n"
        "Examples:\n"
        "Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O\n"
        "5-Fluorouracil: O=C1NC(=O)C(F)=CN1\n"
        "Ethanol test example: CCO"
    )

    st.warning(
        "Türkçe:\n\n"
        "TARGET ve TARGET_PATHWAY değerleri sistemdeki mevcut listelerden seçilmelidir. "
        "Bu tahmin yalnızca araştırma amaçlıdır ve klinik karar olarak kullanılmamalıdır.\n\n"
        "English:\n\n"
        "TARGET and TARGET_PATHWAY should be selected from the available lists in the system. "
        "This prediction is for research purposes only and must not be used as a clinical decision."
    )

    target = st.selectbox(
        "TARGET seçiniz / Select TARGET",
        target_options
    )

    target_pathway = st.selectbox(
        "TARGET_PATHWAY seçiniz / Select TARGET_PATHWAY",
        pathway_options
    )

    st.info(
        "Yeni ilaç için gerekli girdiler: Drug Name, SMILES, TARGET, TARGET_PATHWAY. "
        "Cell Line Excel içinde yazılmaz; arayüzden seçilir.\n\n"
        "Required inputs for a new drug: Drug Name, SMILES, TARGET, TARGET_PATHWAY. "
        "Cell Line is not written inside Excel; it is selected from the interface."
    )

    if st.button("Yeni İlaç İçin Tahmin Yap / Predict New Drug"):
        if not ilac_adi_gecerli_mi(drug_name):
            st.error("Geçerli bir Drug Name giriniz / Please enter a valid Drug Name.")
            st.stop()

        if not temiz_metin_mi(smiles):
            st.error("Geçerli bir SMILES giriniz / Please enter a valid SMILES.")
            st.stop()

        selected_cell_rows = arayuz_df[arayuz_df["CELL_LINE_NAME"] == secilen_cell_line]

        if len(selected_cell_rows) == 0:
            st.error("Seçilen Cell Line bulunamadı / Selected Cell Line was not found.")
            st.stop()

        model_id = selected_cell_rows.iloc[0]["model_id"]

        with st.spinner("Yeni ilaç tahmini hazırlanıyor / Preparing new drug prediction..."):
            base_cell_row = tam_veriden_hucre_satiri_bul(
                model_id=model_id,
                sayisal_sutunlar=sayisal_sutunlar,
                kategorik_sutunlar=kategorik_sutunlar
            )

            if base_cell_row is None:
                st.error("Bu Cell Line için omics özellikleri bulunamadı / Omics features were not found for this Cell Line.")
                st.stop()

            try:
                new_row = yeni_ilac_satiri_hazirla(
                    base_cell_row=base_cell_row,
                    drug_name=drug_name,
                    smiles=smiles,
                    target=target,
                    target_pathway=target_pathway,
                    sayisal_sutunlar=sayisal_sutunlar,
                    kategorik_sutunlar=kategorik_sutunlar
                )

                X_dense = satiri_mlp_girdisine_donustur(
                    new_row,
                    sayisal_sutunlar,
                    kategorik_sutunlar,
                    onehot_encoder,
                    scaler,
                    numeric_fill_values
                )

                tahmin_ln_ic50 = mlp_tahmin_yap(X_dense, mlp_model, device)

            except Exception as e:
                st.error("Tahmin sırasında hata oluştu / An error occurred during prediction.")
                st.exception(e)
                st.stop()

        st.success("Yeni ilaç tahmini tamamlandı / New drug prediction completed.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cell Line", secilen_cell_line)

        with col2:
            st.metric("Drug Name", drug_name)

        with col3:
            st.metric("Predicted LN_IC50", f"{tahmin_ln_ic50:.4f}")

        sonuc_df = pd.DataFrame(
            {
                "Alan / Field": [
                    "Cell Line",
                    "Drug Name",
                    "SMILES",
                    "TARGET",
                    "TARGET_PATHWAY",
                    "Predicted LN_IC50",
                    "Scaler"
                ],
                "Değer / Value": [
                    secilen_cell_line,
                    drug_name,
                    smiles,
                    target,
                    target_pathway,
                    f"{tahmin_ln_ic50:.4f}",
                    os.path.basename(scaler_yolu)
                ]
            }
        )

        st.markdown("### Yeni İlaç Sonucu / New Drug Result")
        st.dataframe(sonuc_df, use_container_width=True)

        yorum_mesaji_goster(tahmin_ln_ic50)
        tahmin_aciklamasi_goster()

        rapor_excel = prediction_report_excel_olustur(
            sonuc_df,
            bilimsel_notlar_df_olustur()
        )

        rapor_dosya_adi = (
            "new_drug_prediction_report_"
            + guvenli_dosya_adi(secilen_cell_line)
            + "_"
            + guvenli_dosya_adi(drug_name)
            + ".xlsx"
        )

        st.download_button(
            label="Yeni İlaç Raporunu İndir / Download New Drug Report",
            data=rapor_excel,
            file_name=rapor_dosya_adi,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


elif sayfa == "Toplu Tahmin / Batch Prediction":
    page_header(
        "Toplu Tahmin / Batch Prediction",
        "Batch prediction for multiple new drugs using an uploaded Excel file"
    )

    st.markdown(
        """
        Bu bölümde kullanıcı bir Excel dosyası yükleyerek aynı Cell Line için birden fazla yeni ilaç tahmini yapabilir.
        """
    )

    st.markdown(
        """
        In this section, the user can upload an Excel file to predict multiple new drugs for the same selected Cell Line.
        """
    )

    ln_ic50_aralik_bilgisi_goster()

    if not RDKIT_AVAILABLE:
        st.error(
            "RDKit yüklü değil. Toplu tahmin için RDKit gereklidir.\n\n"
            "RDKit is not installed. RDKit is required for batch prediction."
        )
        st.code("conda install -c conda-forge rdkit openpyxl")
        st.stop()

    st.info(
        "Excel dosyasında Cell Line yazılmamalıdır. Cell Line bu sayfadan seçilir.\n\n"
        "The Excel file must not contain Cell Line. The Cell Line is selected from this page."
    )

    st.markdown(
        """
        **Excel dosyasında bulunması gereken sütunlar / Required Excel columns:**

        - Drug_Name
        - SMILES
        - TARGET
        - TARGET_PATHWAY
        """
    )

    template_excel = excel_template_olustur()

    st.download_button(
        label="Boş Excel Şablonu İndir / Download Empty Excel Template",
        data=template_excel,
        file_name="new_drug_batch_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    cell_line_options = sorted(arayuz_df["CELL_LINE_NAME"].dropna().unique().tolist())

    secilen_cell_line = st.selectbox(
        "Cell Line seçiniz / Select Cell Line",
        cell_line_options,
        key="batch_cell"
    )

    uploaded_file = st.file_uploader(
        "Excel dosyası yükleyiniz / Upload Excel file",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error("Excel dosyası okunamadı / Excel file could not be read.")
            st.exception(e)
            st.stop()

        st.markdown("### Yüklenen Veri / Uploaded Data")
        st.dataframe(input_df, use_container_width=True)

        gerekli_sutunlar = ["Drug_Name", "SMILES", "TARGET", "TARGET_PATHWAY"]
        eksik_sutunlar = [col for col in gerekli_sutunlar if col not in input_df.columns]

        if len(eksik_sutunlar) > 0:
            st.error(
                "Eksik sütunlar / Missing columns: "
                + ", ".join(eksik_sutunlar)
            )
            st.stop()

        if st.button("Toplu Tahmin Yap / Run Batch Prediction"):
            selected_cell_rows = arayuz_df[arayuz_df["CELL_LINE_NAME"] == secilen_cell_line]

            if len(selected_cell_rows) == 0:
                st.error("Seçilen Cell Line bulunamadı / Selected Cell Line was not found.")
                st.stop()

            model_id = selected_cell_rows.iloc[0]["model_id"]

            base_cell_row = tam_veriden_hucre_satiri_bul(
                model_id=model_id,
                sayisal_sutunlar=sayisal_sutunlar,
                kategorik_sutunlar=kategorik_sutunlar
            )

            if base_cell_row is None:
                st.error("Bu Cell Line için omics özellikleri bulunamadı / Omics features were not found for this Cell Line.")
                st.stop()

            sonuc_listesi = []

            progress = st.progress(0)

            for idx, row in input_df.iterrows():
                drug_name = row.get("Drug_Name", "")
                smiles = row.get("SMILES", "")
                target = row.get("TARGET", "")
                target_pathway = row.get("TARGET_PATHWAY", "")

                sonuc_kaydi = {
                    "Cell_Line": secilen_cell_line,
                    "Drug_Name": drug_name,
                    "SMILES": smiles,
                    "TARGET": target,
                    "TARGET_PATHWAY": target_pathway,
                    "Predicted_LN_IC50": np.nan,
                    "Sensitivity_TR": "",
                    "Sensitivity_EN": "",
                    "Status": "OK",
                    "Error_Message": ""
                }

                try:
                    if not ilac_adi_gecerli_mi(drug_name):
                        raise ValueError("Invalid Drug_Name")

                    if not temiz_metin_mi(smiles):
                        raise ValueError("Invalid SMILES")

                    if not temiz_metin_mi(target):
                        raise ValueError("Invalid TARGET")

                    if not temiz_metin_mi(target_pathway):
                        raise ValueError("Invalid TARGET_PATHWAY")

                    new_row = yeni_ilac_satiri_hazirla(
                        base_cell_row=base_cell_row,
                        drug_name=drug_name,
                        smiles=smiles,
                        target=target,
                        target_pathway=target_pathway,
                        sayisal_sutunlar=sayisal_sutunlar,
                        kategorik_sutunlar=kategorik_sutunlar
                    )

                    X_dense = satiri_mlp_girdisine_donustur(
                        new_row,
                        sayisal_sutunlar,
                        kategorik_sutunlar,
                        onehot_encoder,
                        scaler,
                        numeric_fill_values
                    )

                    pred = mlp_tahmin_yap(X_dense, mlp_model, device)

                    tr_baslik, en_baslik, _, _, _ = ln_ic50_yorumu(pred)

                    sonuc_kaydi["Predicted_LN_IC50"] = round(pred, 4)
                    sonuc_kaydi["Sensitivity_TR"] = tr_baslik
                    sonuc_kaydi["Sensitivity_EN"] = en_baslik

                except Exception as e:
                    sonuc_kaydi["Status"] = "ERROR"
                    sonuc_kaydi["Error_Message"] = str(e)

                sonuc_listesi.append(sonuc_kaydi)
                progress.progress((idx + 1) / len(input_df))

            sonuc_df = pd.DataFrame(sonuc_listesi)

            st.markdown("### Toplu Tahmin Sonuçları / Batch Prediction Results")
            st.dataframe(sonuc_df, use_container_width=True)

            tahmin_aciklamasi_goster()

            sonuc_excel = prediction_report_excel_olustur(
                sonuc_df,
                bilimsel_notlar_df_olustur()
            )

            batch_dosya_adi = (
                "batch_prediction_report_"
                + guvenli_dosya_adi(secilen_cell_line)
                + ".xlsx"
            )

            st.download_button(
                label="Sonuç Raporunu Excel Olarak İndir / Download Result Report as Excel",
                data=sonuc_excel,
                file_name=batch_dosya_adi,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


elif sayfa == "Model Performansları / Model Performance":
    page_header(
        "Final Model Performansları / Final Model Performance",
        "Comparison of machine learning, deep learning and GNN-based models"
    )

    try:
        performans_df = load_csv(FINAL_PERFORMANS_YOLU)

        if "Cell_Line_Input" not in performans_df.columns:
            performans_df.insert(4, "Cell_Line_Input", "None")

        st.markdown("### Final Karşılaştırma Tablosu / Final Comparison Table")
        st.dataframe(performans_df, use_container_width=True)

        grafik_df = performans_df[["Model", "RMSE", "R2"]].copy()
        grafik_df["Model_Kisa"] = grafik_df["Model"].astype(str).replace({
            "MLP Neural Network": "MLP",
            "LightGBM Regressor": "LightGBM",
            "XGBoost Regressor": "XGBoost",
            "GNN + Drug Fingerprint Model": "GNN + FP",
            "Ridge Regression": "Ridge"
        })

        st.markdown("### RMSE Karşılaştırması / RMSE Comparison")
        st.bar_chart(grafik_df.set_index("Model_Kisa")[["RMSE"]])

        st.markdown("### R² Karşılaştırması / R² Comparison")
        st.bar_chart(grafik_df.set_index("Model_Kisa")[["R2"]])

        best_model = performans_df.iloc[0]

        st.success(
            f"En iyi model / Best model: {best_model['Model']} | "
            f"RMSE: {best_model['RMSE']:.4f} | "
            f"R2: {best_model['R2']:.4f}"
        )

        st.info(
            "Cell_Line_Input = None: Hücre hattı adı model girdisi olarak kullanılmamıştır. "
            "Model, hücre hattının multi-omics özelliklerini kullanmıştır.\n\n"
            "Cell_Line_Input = None: The cell line name was not used as a direct model input. "
            "The model used the multi-omics features of the cell line."
        )

    except Exception as e:
        st.error("Performans dosyası okunamadı / Performance file could not be read.")
        st.exception(e)


elif sayfa == "Açıklanabilirlik / Explainability":
    page_header(
        "Açıklanabilirlik Analizleri / Explainability Analyses",
        "XGBoost SHAP plots, LightGBM SHAP table, GNN permutation importance and MLP permutation importance"
    )

    st.markdown("## LightGBM SHAP Feature Family Importance Table")

    try:
        shap_family_df = load_csv(LIGHTGBM_SHAP_FAMILY_YOLU)
        st.dataframe(shap_family_df, use_container_width=True)

        shap_chart = shap_family_df.copy()
        shap_chart = shap_chart.set_index("feature_family")
        st.bar_chart(shap_chart)

    except Exception as e:
        st.error("LightGBM SHAP dosyası okunamadı / LightGBM SHAP file could not be read.")
        st.exception(e)

    st.info(
        "Not / Note:\n\n"
        "Aşağıdaki SHAP görselleri ADIM 06 kodundan gelmektedir ve XGBoost Regressor modeline aittir. "
        "Nihai model MLP olduğu için MLP açıklanabilirliği ayrıca Permutation Importance bölümünde verilmiştir.\n\n"
        "The SHAP plots below come from the ADIM 06 code and belong to the XGBoost Regressor model. "
        "Since the final model is MLP, MLP explainability is provided separately in the Permutation Importance section."
    )

    shap_original_graphics_section()

    st.markdown("---")

    st.markdown("## GNN Permutation / Ablation Importance")

    try:
        gnn_importance_df = load_csv(GNN_IMPORTANCE_YOLU)
        st.dataframe(gnn_importance_df, use_container_width=True)

        gnn_chart = gnn_importance_df[["Feature_Group", "RMSE_Increase"]].copy()
        gnn_chart = gnn_chart.set_index("Feature_Group")
        st.bar_chart(gnn_chart)

    except Exception as e:
        st.error("GNN importance dosyası okunamadı / GNN importance file could not be read.")
        st.exception(e)


    st.markdown("---")

    st.markdown("## MLP Permutation Importance")

    try:
        mlp_importance_df = load_csv(MLP_IMPORTANCE_YOLU)
        st.dataframe(mlp_importance_df, use_container_width=True)

        mlp_chart = mlp_importance_df[["Feature_Group", "RMSE_Increase"]].copy()
        mlp_chart = mlp_chart.set_index("Feature_Group")
        st.bar_chart(mlp_chart)

        st.info(
            "Türkçe:\n\n"
            "Bu analiz, en iyi performansı veren MLP modelinin hangi özellik gruplarına daha fazla bağımlı olduğunu göstermektedir. "
            "RMSE_Increase değeri yüksek olan gruplar model tahmini için daha önemlidir. "
            "Sonuçlara göre MLP modeli en çok Drug Fingerprint özelliklerine bağımlıdır. "
            "Bunu Proteomics ve Drug Target grupları takip etmektedir.\n\n"
            "English:\n\n"
            "This analysis shows which feature groups the best-performing MLP model depends on most. "
            "Feature groups with higher RMSE_Increase values are more important for model prediction. "
            "According to the results, the MLP model depends most strongly on Drug Fingerprint features, "
            "followed by Proteomics and Drug Target groups."
        )

    except Exception as e:
        st.error("MLP permutation importance dosyası okunamadı / MLP permutation importance file could not be read.")
        st.exception(e)



    st.markdown("---")
    additional_model_tests_section()

    best_model_plot_analysis_section()


elif sayfa == "Proje Hakkında / About Project":
    page_header(
        "Proje Hakkında / About Project",
        "Scientific overview of the multi-omics cancer drug sensitivity prediction system"
    )

    st.markdown(
        """
        ## Türkçe

        Bu çalışma, kanser hücre hatlarında ilaç duyarlılığı tahmini için çoklu omics verileri
        ve ilaç kimyasal özelliklerini birleştiren yapay zeka tabanlı bir sistem geliştirmeyi amaçlamaktadır.

        Geliştirilen sistemde klasik makine öğrenmesi, ağaç tabanlı modeller, derin öğrenme ve GNN tabanlı
        yaklaşımlar karşılaştırmalı olarak değerlendirilmiştir.

        **Geliştirilen modeller:**

        - Ridge Regression
        - XGBoost Regressor
        - LightGBM Regressor
        - MLP Neural Network
        - GNN + Drug Fingerprint Model

        **Ek olarak test edilen yaklaşımlar:**

        - Optuna tabanlı MLP hiperparametre optimizasyonu
        - MLP + LightGBM + XGBoost ensemble denemesi
        - Stacking meta-model yaklaşımı
        - CatBoost Regressor
        - GAT ve GraphSAGE tabanlı GNN deneyleri

        Bu ek yaklaşımlar nihai model seçimini doğrulamak için denenmiş, ancak mevcut MLP modelinin genel performansını aşamamıştır.

        **En iyi random split sonucu:**

        - MLP Neural Network

        **Yeni ilaç tahmini:**

        Yeni ilaç tahmini için Cell Line sistemdeki mevcut hücrelerden seçilir.
        Kullanıcı yeni ilaç için SMILES, TARGET ve TARGET_PATHWAY bilgilerini girer.
        SMILES değeri RDKit ile Morgan Fingerprint vektörüne dönüştürülür.

        **Bilimsel uyarı:**

        Bu sistem klinik karar verme amacıyla kullanılmamalıdır. Yalnızca araştırma, eğitim ve akademik sunum amacı taşır.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ## English

        This study aims to develop an AI-based system that integrates multi-omics data
        and drug chemical features for predicting drug sensitivity in cancer cell lines.

        The developed system compares classical machine learning, tree-based models,
        deep learning models, and GNN-based approaches.

        **Developed models:**

        - Ridge Regression
        - XGBoost Regressor
        - LightGBM Regressor
        - MLP Neural Network
        - GNN + Drug Fingerprint Model

        **Best random split result:**

        - MLP Neural Network

        **New drug prediction:**

        For new drug prediction, the Cell Line is selected from the existing known cell lines.
        The user provides SMILES, TARGET, and TARGET_PATHWAY for the new drug.
        The SMILES value is converted into a Morgan Fingerprint vector using RDKit.

        **Scientific warning:**

        This system must not be used for clinical decision-making. It is intended only for research, education, and academic presentation.
        """
    )


    st.markdown("---")
    external_links_section()


elif sayfa == "İletişim / Contact":
    page_header(
        "İletişim / Contact",
        "Project developer contact information"
    )

    st.markdown(
        """
        **Zuhir Marwan**

        Mekatronik Mühendisi ve Yapay Zeka Geliştiricisi  
        Mechatronics Engineer and Artificial Intelligence Developer

        Telefon / Phone: 05348189108
        """
    )


    st.markdown("---")
    st.markdown("## Bağlantılar / Links")

    info_box_tr_en(
        "Proje hakkında iletişim kurmak için LinkedIn profilini kullanabilirsiniz. "
        "Tam veri seti ve kaynak dosyalar Kaggle üzerinde paylaşılmıştır.",
        "You can use the LinkedIn profile to contact the project owner. "
        "The full dataset and source files are shared on Kaggle."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.link_button(
            "LinkedIn / Contact",
            LINKEDIN_PROFILE_URL
        )

    with col2:
        st.link_button(
            "Kaggle Dataset / Full Data and Source Files",
            KAGGLE_DATASET_URL
        )

