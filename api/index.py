from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import traceback

# ─── CLASS TRANSFORMER (Wajib Dideklarasikan Agar PKL Bisa Dimuat) ───────────
class CBCCalculatorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.output_features_ = [
            'Gender', 'Age', 'hb', 'rbc', 'hct', 'mcv', 'mch', 'mchc', 'rdw','plt', 'wbc', 'abs_neu', 'abs_lym', 'abs_mon', 'abs_eos', 'nlr', 'plr', 'mlr'
        ]
        return self

    def transform(self, X):
        X_calc = X.copy()
        X_calc['hct'] = (X_calc['rbc'] * X_calc['mcv']) / 10
        X_calc['mch'] = np.where(X_calc['rbc'] > 0, (X_calc['hb'] / X_calc['rbc']) * 10, 0)
        X_calc['mchc'] = np.where(X_calc['hct'] > 0, (X_calc['hb'] / X_calc['hct']) * 100, 0)
        X_calc['abs_neu'] = (X_calc['neu'] / 100.0) * X_calc['wbc']
        X_calc['abs_lym'] = (X_calc['lym'] / 100.0) * X_calc['wbc']
        X_calc['abs_mon'] = (X_calc['mon'] / 100.0) * X_calc['wbc']
        X_calc['abs_eos'] = (X_calc['eos'] / 100.0) * X_calc['wbc']
        X_calc['nlr'] = np.where(X_calc['abs_lym'] > 0, X_calc['abs_neu'] / X_calc['abs_lym'], 0)
        X_calc['plr'] = np.where(X_calc['abs_lym'] > 0, X_calc['plt'] / X_calc['abs_lym'], 0)
        X_calc['mlr'] = np.where(X_calc['abs_lym'] > 0, X_calc['abs_mon'] / X_calc['abs_lym'], 0)
        return X_calc[self.output_features_]

# ─── INISIALISASI & KONSTANTA ────────────────────────────────────────────────
app = FastAPI()

CLASS_NAMES = ["Normal", "ITP", "Dengue/DBD", "Thrombocytosis"]

# ➡️ PERUBAHAN: Matriks 14 Gejala Klinis Terbaru
SYMPTOM_WEIGHTS = {
    "Demam tinggi mendadak (2-7 hari)": [0.0, 0.0, 0.85, 0.0],
    "Nyeri pegal hebat di belakang mata, otot, dan sendi": [0.0, 0.0, 0.75, 0.0],
    "Bintik merah di kulit atau memar tiba-tiba tanpa sebab": [0.0, 0.85, 0.45, 0.05],
    "Mimisan atau gusi berdarah secara tiba-tiba": [0.0, 0.75, 0.35, 0.10],
    "Muntah darah atau BAB berwarna hitam legam": [0.0, 0.30, 0.20, 0.05],
    "Haid/Menstruasi sangat deras dan lama (pada perempuan)": [0.0, 0.60, 0.15, 0.05],
    "Sesak napas atau perut terasa bengkak/sangat begah (Gejala rembesan cairan)": [0.0, 0.0, 0.70, 0.0],
    "Ujung jari tangan/kaki sangat dingin, pucat, dan badan sangat lemas (Gejala menuju syok)": [0.0, 0.05, 0.80, 0.0],
    "Telapak tangan/kaki terasa panas terbakar dan kemerahan (Erythromelalgia)": [0.0, 0.0, 0.0, 0.80],
    "Kulit terasa sangat gatal setelah mandi atau kena air (Pruritus aquagenik)": [0.0, 0.0, 0.0, 0.75],
    "Kaki/betis tiba-tiba bengkak dan sangat nyeri (Gejala sumbatan darah)": [0.0, 0.0, 0.0, 0.85],
    "Perut kiri atas terasa mengganjal dan cepat kenyang saat makan (Gejala limpa bengkak)": [0.0, 0.05, 0.15, 0.65],
    "Rasa kliyengan (mau pingsan) disertai kesemutan/kebas": [0.0, 0.0, 0.10, 0.70],
    "Asimptomatik (tidak ada keluhan klinis)": [0.50, 0.0, 0.0, 0.0]
}

try:
    # ➡️ PERUBAHAN: Load 100% menggunakan Pickle/Joblib (Termasuk Model ML)
    # Pastikan nama file ini sesuai dengan yang diekstrak dari skrip Colab Anda
    scaler = joblib.load("std_scaler.pkl") 
    imputer = joblib.load("knn_imputer.pkl")
    ml_model = joblib.load("lr_model.pkl") 
    print("✅ Seluruh model (.pkl) berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# ─── SCHEMA INPUT DARI WEB ───────────────────────────────────────────────────
class PatientInput(BaseModel):
    Gender: Optional[float] = None
    Age: Optional[float] = None
    hb: Optional[float] = None
    rbc: Optional[float] = None
    mcv: Optional[float] = None
    rdw: Optional[float] = None
    wbc: Optional[float] = None
    neu: Optional[float] = None
    lym: Optional[float] = None
    mon: Optional[float] = None
    eos: Optional[float] = None
    plt: Optional[float] = None
    symptoms: List[str] = []
    weight_ml: float = 55.0
    weight_sym: float = 25.0
    weight_who: float = 20.0

# ─── FUNGSI LOGIKA ───────────────────────────────
def pillar_iii_who_rules(plt_val, wbc_val, hct_val):
    if np.isnan(plt_val) or np.isnan(wbc_val) or np.isnan(hct_val):
        return np.ones(4) / 4.0
    raw = np.zeros(4, dtype=float)
    if plt_val < 100 and wbc_val < 5.0 and hct_val > 45: raw[2] = 1.0  
    elif plt_val < 100 and wbc_val < 5.0: raw[2] = 0.6  
    elif plt_val < 100 and hct_val > 45: raw[2] = 0.4  
    if plt_val < 100 and 5.0 <= wbc_val <= 12.0 and 35 <= hct_val <= 50: raw[1] = 1.0
    elif plt_val < 100 and raw[2] == 0: raw[1] = 0.5  
    if plt_val > 600: raw[3] = 1.0
    elif plt_val > 450: raw[3] = 0.7
    if (150 <= plt_val <= 400) and (4.0 <= wbc_val <= 10.0) and (35 <= hct_val <= 50): raw[0] = 1.0
    elif raw.sum() == 0: raw[0] = 0.3  
    if raw.sum() > 0: return raw / raw.sum()
    return np.ones(4) / 4.0

def pillar_ii_symptom_score(selected_symptoms):
    if not selected_symptoms: return np.array([0.25, 0.25, 0.25, 0.25])
    raw = np.zeros(4, dtype=float)
    for symptom in selected_symptoms: 
        if symptom in SYMPTOM_WEIGHTS:
            raw += np.array(SYMPTOM_WEIGHTS[symptom])
            
    # Tangani Asimptomatik secara absolut
    if "Asimptomatik (tidak ada keluhan klinis)" in selected_symptoms or raw.sum() == 0:
        return np.array(SYMPTOM_WEIGHTS["Asimptomatik (tidak ada keluhan klinis)"])
        
    if raw.sum() > 0: return raw / raw.sum()
    return np.ones(4) / 4.0

def get_detailed_explanation(feature_name, shap_value, pred_class):
    direction = "mendorong kuat ke arah" if shap_value > 0 else "menahan/mengurangi risiko"
    feat_upper = feature_name.upper()
    if feature_name.lower() == "plt":
        if shap_value > 0 and pred_class in [1, 2]: return f"**Trombosit (PLT):** Penurunan ekstrem parameter ini {direction} diagnosis. Secara patofisiologis, ini merepresentasikan destruksi perifer akut atau supresi produksi."
        elif shap_value > 0 and pred_class == 3: return f"**Trombosit (PLT):** Lonjakan masif nilai absolut trombosit {direction} diagnosis. Hal ini mencerminkan aktivitas megakaryopoiesis otonom."
    elif feature_name.lower() == "abs_lym" and pred_class == 2:
        return f"**Absolute Lymphocyte (ABS_LYM):** Fluktuasi limfosit absolut {direction} diagnosis, menangkap mobilisasi imunitas adaptif fase kritis replikasi virus."
    return f"**{feat_upper}:** Berkontribusi memicu batas ambang (threshold) dalam {direction} keputusan diagnosis ini."

# ─── ENDPOINT UTAMA ──────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict_diagnosis(data: PatientInput):
    try:
        # 1. Konversi Input ke DataFrame
        if hasattr(data, "model_dump"):
            raw_dict = data.model_dump(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
        else:
            raw_dict = data.dict(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
            
        raw_dict = {k: (np.nan if v is None else v) for k, v in raw_dict.items()}
        raw_df = pd.DataFrame([raw_dict])
        
        # 2. Rekayasa Fitur Otomatis
        cbc_calc = CBCCalculatorTransformer()
        cbc_calc.fit(raw_df) 
        engineered_df = cbc_calc.transform(raw_df)
        
        # --- PENYELARASAN BENTUK DATA SCALER ---
        expected_features = list(scaler.feature_names_in_)
        aligned_df = pd.DataFrame(columns=expected_features)
        aligned_df.loc[0] = np.nan
        
        web_cols = list(engineered_df.columns)
        web_cols_upper = [c.upper() for c in web_cols]
        
        for expected_col in expected_features:
            exp_upper = expected_col.upper()
            if exp_upper in web_cols_upper:
                idx = web_cols_upper.index(exp_upper)
                aligned_df.at[0, expected_col] = engineered_df.iloc[0, idx]
            else:
                exp_clean = exp_upper.replace("%", "")
                if exp_clean in web_cols_upper:
                    idx = web_cols_upper.index(exp_clean)
                    aligned_df.at[0, expected_col] = engineered_df.iloc[0, idx]
                    
        aligned_df = aligned_df.fillna(0)
        
        # 3. Pra-pemrosesan Terpisah
        scaled_data = scaler.transform(aligned_df)
        imputed_data = imputer.transform(scaled_data)
        final_df = pd.DataFrame(imputed_data, columns=expected_features)
        
        # --- FILTER FITUR KHUSUS REGRESI LOGISTIK (LASSO) ---
        # ⚠️ PENTING: Ganti isi list ini dengan 10 fitur spesifik yang dihasilkan dari LASSO L1 Anda.
        target_features = ['PLT', 'Umur', 'PLR', 'MCV', 'MCH', 'ABS_EOS', 'ABS_NEU', 'WBC', 'RBC', 'ABS_MON']
        
        ml_features = []
        for tf in target_features:
            for exp_col in expected_features:
                if tf.lower() == exp_col.lower():
                    ml_features.append(exp_col)
                    break
        
        ml_input_df = final_df[ml_features]
        # -----------------------------------
        
        # 4. Prediksi Machine Learning
        p_ml = ml_model.predict_proba(ml_input_df)[0]
        if len(p_ml) > 4: p_ml = p_ml[:4]
        if len(p_ml) < 4: p_ml = np.pad(p_ml, (0, 4 - len(p_ml)))
        p_ml = p_ml / p_ml.sum()
        
        # 5. Fusi Tri-Brid
        hct_calc = float(engineered_df['hct'].iloc[0])
        p_sym = pillar_ii_symptom_score(data.symptoms)
        p_who = pillar_iii_who_rules(raw_dict["plt"], raw_dict["wbc"], hct_calc)
        
        w_ml, w_sym, w_who = data.weight_ml/100.0, data.weight_sym/100.0, data.weight_who/100.0
        p_final = (w_ml * p_ml) + (w_sym * p_sym) + (w_who * p_who)
        if p_final.sum() > 0: p_final = p_final / p_final.sum()
        
        pred_class = int(np.argmax(p_final))
        
        # 6. Analisis SHAP (Disesuaikan untuk Regresi Logistik)
        explainer = shap.Explainer(ml_model, ml_input_df)
        shap_explanation = explainer(ml_input_df)
        
        # ➡️ Penanganan Dinamis: Jika output berupa list (ciri khas Regresi Logistik multikelas)
        if isinstance(shap_explanation.values, list):
            sv_values = shap_explanation.values[pred_class][0]
            base_val = shap_explanation.base_values[pred_class][0]
            single_expl = shap.Explanation(values=sv_values, 
                                           base_values=base_val, 
                                           data=ml_input_df.iloc[0].values, 
                                           feature_names=ml_features)
        else:
            single_expl = shap_explanation[0, :, pred_class] if len(shap_explanation.shape) == 3 else shap_explanation[0, :]
            single_expl.data = ml_input_df.iloc[0].values
        
        # 7. Visualisasi SHAP
        plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(single_expl, max_display=7, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # 8. Pembangkit Teks CLIX-M
        sv_vals = single_expl.values
        sorted_idx = np.argsort(np.abs(sv_vals))[::-1][:3]
        explanations = []
        for idx in sorted_idx:
            feat_name = ml_features[idx]
            explanations.append(get_detailed_explanation(feat_name, sv_vals[idx], pred_class))
            
        # --- BUKTI KALKULASI REDUNDANSI (UNTUK SLIDE KANAN & PRINT) ---
        calc_results = {
            "Hematokrit (HCT)": round(float(engineered_df['hct'].iloc[0]), 2),
            "Mean Corpuscular Hemoglobin (MCH)": round(float(engineered_df['mch'].iloc[0]), 2),
            "MCH Concentration (MCHC)": round(float(engineered_df['mchc'].iloc[0]), 2),
            "Absolut Neutrofil": round(float(engineered_df['abs_neu'].iloc[0]), 2),
            "Absolut Limfosit": round(float(engineered_df['abs_lym'].iloc[0]), 2),
            "Absolut Monosit": round(float(engineered_df['abs_mon'].iloc[0]), 2),
            "Absolut Eosinofil": round(float(engineered_df['abs_eos'].iloc[0]), 2),
            "Neutrophil-Lymphocyte Ratio (NLR)": round(float(engineered_df['nlr'].iloc[0]), 2),
            "Platelet-Lymphocyte Ratio (PLR)": round(float(engineered_df['plr'].iloc[0]), 2),
            "Monocyte-Lymphocyte Ratio (MLR)": round(float(engineered_df['mlr'].iloc[0]), 2)
        }
        # ------------------------------------------------------
            
        return {
            "status": "success",
            "diagnosis": CLASS_NAMES[pred_class],
            "probabilitas_final": round(float(p_final[pred_class]*100), 2),
            "breakdown": {
                "Pilar_1_ML": round(float(p_ml[pred_class]*100), 2),
                "Pilar_2_Sym": round(float(p_sym[pred_class]*100), 2),
                "Pilar_3_WHO": round(float(p_who[pred_class]*100), 2),
            },
            "shap_image": image_base64,
            "clix_m_text": explanations,
            "kalkulasi_fisiologis": calc_results 
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
