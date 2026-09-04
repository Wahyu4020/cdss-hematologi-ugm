from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# ─── CLASS TRANSFORMER (Wajib Dideklarasikan Agar PKL Bisa Dimuat) ───────────
class CBCCalculatorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.output_features_ = [
            'Gender', 'Age', 'hb', 'rbc', 'hct', 'mcv', 'mch', 'mchc', 'rdw',
            'plt', 'wbc', 
            'abs_neu', 'abs_lym', 'abs_mon', 'abs_eos',
            'nlr', 'plr', 'mlr'
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
SYMPTOM_WEIGHTS = {
    "Petekie spontan / memar tanpa trauma":        [0.0, 0.40, 0.20, 0.0],
    "Perdarahan mukosa (epistaksis, gusi)":        [0.0, 0.35, 0.25, 0.0],
    "Demam tinggi 2–7 hari mendadak":              [0.0, 0.05, 0.45, 0.0],
    "Nyeri retro-orbital / sakit kepala hebat":    [0.0, 0.0,  0.35, 0.0],
    "Mialgia / artralgia (nyeri otot-sendi)":      [0.0, 0.0,  0.30, 0.0],
    "Ruam kulit / dengue rash":                    [0.0, 0.0,  0.40, 0.0],
    "Riwayat trombosis / DVT / emboli":            [0.0, 0.05, 0.0,  0.55],
    "Eritromelalgia / kemerahan ujung jari":       [0.0, 0.0,  0.0,  0.45],
    "Asimptomatik (tidak ada keluhan klinis)":     [0.50, 0.0, 0.0,  0.0],
}

try:
    fe_model = joblib.load("feature_engineering.pkl")
    ml_model = joblib.load("modelML.pkl")
except Exception as e:
    print(f"Error loading models: {e}")

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

# ─── FUNGSI LOGIKA (Diadaptasi dari Streamlit) ───────────────────────────────
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
    if raw.sum() > 0: return raw / raw.sum()
    return np.ones(4) / 4.0

def get_detailed_explanation(feature_name, shap_value, pred_class):
    # Logika yang sama persis dengan fungsi Streamlit Anda untuk output teks
    direction = "mendorong kuat ke arah" if shap_value > 0 else "menahan/mengurangi risiko"
    feat_upper = feature_name.upper()
    if feature_name == "plt":
        if shap_value > 0 and pred_class in [1, 2]: return f"**Trombosit (PLT):** Penurunan ekstrem parameter ini {direction} diagnosis. Secara patofisiologis, ini merepresentasikan destruksi perifer akut atau supresi produksi."
        elif shap_value > 0 and pred_class == 3: return f"**Trombosit (PLT):** Lonjakan masif nilai absolut trombosit {direction} diagnosis. Hal ini mencerminkan aktivitas megakaryopoiesis otonom."
    elif feature_name == "abs_lym" and pred_class == 2:
        return f"**Absolute Lymphocyte (ABS_LYM):** Fluktuasi limfosit absolut {direction} diagnosis, menangkap mobilisasi imunitas adaptif fase kritis replikasi virus."
    # Fallback penjelasan umum
    return f"**{feat_upper}:** Berkontribusi memicu batas ambang (threshold) dalam {direction} keputusan diagnosis ini."

# ─── ENDPOINT UTAMA ──────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict_diagnosis(data: PatientInput):
    try:
        # 1. Konversi Input ke DataFrame 
        raw_dict = data.dict(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
        # Ganti None dengan np.nan untuk Pipeline
        raw_dict = {k: (np.nan if v is None else v) for k, v in raw_dict.items()}
        raw_df = pd.DataFrame([raw_dict])
        
        # 2. Rekayasa Fitur Otomatis (Pilar I)
        engineered_df = fe_model.transform(raw_df)
        
        # 3. Prediksi Machine Learning
        p_ml = ml_model.predict_proba(engineered_df)[0]
        if len(p_ml) > 4: p_ml = p_ml[:4]
        if len(p_ml) < 4: p_ml = np.pad(p_ml, (0, 4 - len(p_ml)))
        p_ml = p_ml / p_ml.sum()
        
        # 4. Fusi Tri-Brid (Pilar II & III)
        hct_calc = float(engineered_df['hct'].iloc[0])
        p_sym = pillar_ii_symptom_score(data.symptoms)
        p_who = pillar_iii_who_rules(raw_dict["plt"], raw_dict["wbc"], hct_calc)
        
        w_ml, w_sym, w_who = data.weight_ml/100.0, data.weight_sym/100.0, data.weight_who/100.0
        p_final = (w_ml * p_ml) + (w_sym * p_sym) + (w_who * p_who)
        if p_final.sum() > 0: p_final = p_final / p_final.sum()
        
        pred_class = int(np.argmax(p_final))
        
        # 5. Analisis SHAP
        underlying_model = ml_model
        if hasattr(ml_model, "named_steps"):
            underlying_model = list(ml_model.named_steps.values())[-1]
            
        explainer = shap.TreeExplainer(underlying_model)
        shap_explanation = explainer(engineered_df)
        
        single_expl = shap_explanation[0, :, pred_class] if len(shap_explanation.shape) == 3 else shap_explanation[0, :]
        single_expl.data = engineered_df.iloc[0].values
        
        # Visualisasi SHAP
        plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(single_expl, max_display=7, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # 6. Pembangkit Teks CLIX-M
        sv_vals = single_expl.values
        sorted_idx = np.argsort(np.abs(sv_vals))[::-1][:3]
        explanations = []
        feat_names = list(engineered_df.columns)
        for idx in sorted_idx:
            feat_name = feat_names[idx]
            explanations.append(get_detailed_explanation(feat_name, sv_vals[idx], pred_class))
            
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
            "clix_m_text": explanations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))