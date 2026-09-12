from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import shap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import traceback

# ─── CLASS TRANSFORMER ───────────────────────────────────────────────────────
class CBCCalculatorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.output_features_ = [
            'gender', 'age', 'hb', 'rbc', 'hct', 'mcv', 'mch', 'mchc', 'rdw','plt', 'wbc', 'abs_neu', 'abs_lym', 'abs_mon', 'abs_eos', 'nlr', 'plr', 'mlr'
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

# Mapping Pintar untuk menjembatani Frontend ke Model Machine Learning
MAP_WEB_TO_DATASET = {
    'gender': 'L/P', 'age': 'Umur', 'hb': 'HB', 'rbc': 'RBC', 'mcv': 'MCV',
    'rdw': 'RDW', 'wbc': 'WBC', 'plt': 'PLT', 'neu': 'NEU%', 'lym': 'LYM%',
    'mon': 'MON%', 'eos': 'EOS%', 'hct': 'HCT', 'mch': 'MCH', 'mchc': 'MCHC',
    'abs_neu': 'ABS_NEU', 'abs_lym': 'ABS_LYM', 'abs_mon': 'ABS_MON', 
    'abs_eos': 'ABS_EOS', 'nlr': 'NLR', 'plr': 'PLR', 'mlr': 'MLR'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    scaler = joblib.load(os.path.join(BASE_DIR, "std_scaler.pkl"))
    imputer = joblib.load(os.path.join(BASE_DIR, "knn_imputer.pkl"))
    ml_model = joblib.load(os.path.join(BASE_DIR, "lr_model.pkl"))
    print("✅ Model berhasil dimuat!")
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
    if "Asimptomatik (tidak ada keluhan klinis)" in selected_symptoms or raw.sum() == 0:
        return np.array(SYMPTOM_WEIGHTS["Asimptomatik (tidak ada keluhan klinis)"])
    if raw.sum() > 0: return raw / raw.sum()
    return np.ones(4) / 4.0

def get_detailed_explanation(feature_name, shap_value, pred_class, raw_val):
    direction = "mendorong ke arah" if shap_value > 0 else "menahan/mengurangi risiko"
    feat_upper = feature_name.upper()
    
    # Cetak nama fitur beserta nilai aslinya di UI
    base_text = f"**{feat_upper} ({round(raw_val, 2)}):** Nilai ini {direction} keputusan diagnosis."
    
    # ─── LOGIKA PLT (TROMBOSIT) ───
    if feat_upper == "PLT":
        if raw_val < 150:
            if pred_class == 1: # ITP
                return base_text + " Penurunan trombosit (trombositopenia) terisolasi merupakan tanda khas ITP (Immune Thrombocytopenia), di mana sistem imun secara keliru menghancurkan keping darah tanpa penyebab infeksi yang jelas."
            elif pred_class == 2: # DENGUE
                return base_text + " Penurunan trombosit sangat lazim pada infeksi virus Dengue akibat supresi sumsum tulang secara langsung oleh virus dan destruksi keping darah di sirkulasi perifer."
            else:
                return base_text + " Penurunan keping darah mengindikasikan adanya gangguan produksi pada sumsum tulang atau tingginya tingkat destruksi perifer."
        elif raw_val > 450:
            if pred_class == 3: # TROMBOSITOSIS
                return base_text + " Produksi trombosit yang berlebihan (trombositosis) mengonfirmasi hiperaktivitas sumsum tulang, yang berpotensi merujuk pada kelainan mieloproliferatif."
            else:
                return base_text + " Peningkatan keping darah di atas rentang fisiologis menandakan adanya anomali pada aktivitas produksi megakariosit di dalam sumsum tulang."
                
    # ─── LOGIKA WBC (LEUKOSIT) ───
    elif feat_upper == "WBC":
        if raw_val < 4.0:
            if pred_class == 2: # DENGUE
                return base_text + " Penurunan sel darah putih (leukopenia) merupakan penanda awal yang sangat khas pada fase akut infeksi virus seperti Dengue."
            else:
                return base_text + " Leukopenia dapat terjadi akibat penyakit penekanan imun, malnutrisi, toksisitas obat, atau supresi sumsum tulang."
        elif raw_val > 11.0:
            return base_text + " Peningkatan sel darah putih (leukositosis) menandakan respons imun tubuh yang aktif untuk melawan infeksi bakteri atau inflamasi hebat."
        else:
            if pred_class == 1: # ITP
                return base_text + " Pada ITP, kelainan hematologi umumnya murni hanya terjadi pada keping darah, sehingga nilai WBC yang normal ini turut memperkuat tegaknya diagnosis ITP."
            return base_text + " Jumlah leukosit yang berada dalam rentang fisiologis menandakan produksi sel imun bawaan tidak mengalami gangguan."

    # ─── LOGIKA RBC (ERITROSIT) ───
    elif feat_upper == "RBC":
        if raw_val > 5.5 and pred_class == 2: # DENGUE
            return base_text + " Peningkatan eritrosit (hemokonsentrasi) merupakan tanda bahaya (danger sign) pada Dengue yang menunjukkan adanya sindrom kebocoran plasma darah."
        elif raw_val < 4.0:
            return base_text + " Penurunan eritrosit merupakan penanda kondisi anemia, riwayat perdarahan, atau gangguan produksi sel darah merah."
            
    # ─── LOGIKA INDEKS ERITROSIT REDUNDAN (MCV, MCH, MCHC) ───
    elif feat_upper == "MCV":
        redundansi_mcv = " Sebagai parameter turunan, MCV merupakan cerminan matematis dari rasio antara Hematokrit (HCT) dan jumlah absolut eritrosit (RBC)."
        if raw_val < 80:
            return base_text + redundansi_mcv + " MCV yang rendah (mikrositik) sering menjadi rujukan adanya penyakit penyerta seperti anemia defisiensi besi atau talasemia."
        elif raw_val > 100:
            return base_text + redundansi_mcv + " MCV yang tinggi (makrositik) mengindikasikan kemungkinan defisiensi vitamin B12/folat atau penyakit hati."
        else:
            return base_text + redundansi_mcv + " Nilai dalam rentang normal menunjukkan ukuran sel darah merah yang proporsional (normositik)."
            
    elif feat_upper == "MCH":
        redundansi_mch = " Secara fisiologis, parameter ini berkorelasi langsung dengan perhitungan rasio massa Hemoglobin (HB) terhadap jumlah eritrosit (RBC)."
        if raw_val < 27:
            return base_text + redundansi_mch + " Nilai MCH yang rendah merepresentasikan sel darah merah yang hipokromik (pucat) karena kekurangan kadar hemoglobin intraseluler."
        else:
            return base_text + redundansi_mch + " Kepadatan hemoglobin per sel darah merah terpantau berada dalam batas wajar."
            
    elif feat_upper == "MCHC":
        redundansi_mchc = " Indeks ini merupakan kalkulasi redundan dari perbandingan antara kadar Hemoglobin (HB) dan persentase Hematokrit (HCT)."
        if raw_val < 32:
            return base_text + redundansi_mchc + " Penurunan konsentrasi ini mengonfirmasi kondisi hipokromia pada eritrosit pasien."
        else:
            return base_text + redundansi_mchc + " Konsentrasi hemoglobin intraseluler terpantau seimbang dengan volume sel darah merah (normokromik)."

    # ─── LOGIKA DIFERENSIAL LEUKOSIT ───
    elif feat_upper == "ABS_LYM":
        if pred_class == 2: # DENGUE
            return base_text + " Fluktuasi limfosit secara klinis erat kaitannya dengan mobilisasi sistem imunitas adaptif untuk merespons replikasi virus Dengue di dalam tubuh."
        else:
            return base_text + " Limfosit berperan penting dalam kekebalan humoral untuk merespons penyakit virus atau inflamasi imunologis."
            
    elif feat_upper == "ABS_NEU":
        if raw_val < 2.0 and pred_class == 2: # DENGUE
            return base_text + " Penurunan neutrofil (neutropenia) sering menyertai fase akut infeksi virus akibat supresi sementara pada sumsum tulang."
        elif raw_val > 7.5:
            return base_text + " Peningkatan neutrofil (neutrofilia) adalah respons garda terdepan terhadap adanya infeksi bakteri piogenik atau trauma jaringan."
            
    elif feat_upper == "ABS_MON":
        return base_text + " Monosit bertindak sebagai fagosit pembersih; peningkatannya sering terlihat pada fase pemulihan infeksi akut atau peradangan kronis."
        
    elif feat_upper == "ABS_EOS":
        return base_text + " Fluktuasi eosinofil merupakan respons biologis yang lazim pada reaksi alergi, infeksi parasit, atau dermatitis atopik."
        
    # ─── LOGIKA RASIO INFLAMASI REDUNDAN (NLR, PLR, MLR) ───
    elif feat_upper in ["NLR", "PLR", "MLR"]:
        if feat_upper == "NLR":
            korelasi_rasio = " Rasio ini secara langsung diturunkan dari perbandingan nilai absolut Neutrofil terhadap Limfosit."
        elif feat_upper == "PLR":
            korelasi_rasio = " Rasio ini secara langsung diturunkan dari perbandingan jumlah absolut Trombosit (PLT) terhadap Limfosit."
        else:
            korelasi_rasio = " Rasio ini secara langsung diturunkan dari perbandingan nilai absolut Monosit terhadap Limfosit."
            
        return base_text + korelasi_rasio + " Fitur turunan antar-sel darah ini digunakan oleh model sebagai biomarker prediktif tambahan untuk menilai derajat keparahan inflamasi sistemik secara komprehensif."
        
    return base_text
# ─── ENDPOINT UTAMA ──────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict_diagnosis(data: PatientInput):
    try:
        if hasattr(data, "model_dump"):
            raw_dict = data.model_dump(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
        else:
            raw_dict = data.dict(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
            
        # Mengubah key ke lowercase agar konsisten
        raw_dict = {k.lower(): (np.nan if v is None else v) for k, v in raw_dict.items()}
        raw_df = pd.DataFrame([raw_dict])
        
        # 1. Kalkulasi Fisiologis
        cbc_calc = CBCCalculatorTransformer()
        cbc_calc.fit(raw_df) 
        engineered_df = cbc_calc.transform(raw_df)
        
        # 2. Penyelarasan Nama Kolom (Mencegah Bug 'Umur' dan Data Hilang)
        engineered_df.columns = [c.lower() for c in engineered_df.columns]
        engineered_df = engineered_df.rename(columns=MAP_WEB_TO_DATASET)
        
        # 3. Penyelarasan Skala Data
        expected_features = list(scaler.feature_names_in_)
        aligned_df = pd.DataFrame(columns=expected_features)
        aligned_df.loc[0] = np.nan
        
        for expected_col in expected_features:
            if expected_col in engineered_df.columns:
                aligned_df.at[0, expected_col] = engineered_df.iloc[0][expected_col]
        
        aligned_df = aligned_df.fillna(0)
        
        # 4. Scaling dan Imputasi
        scaled_data = scaler.transform(aligned_df)
        imputed_data = imputer.transform(scaled_data)
        final_df = pd.DataFrame(imputed_data, columns=expected_features)
        
        # 5. Filter Spesifik 10 Fitur Model Regresi Logistik
        target_features = ['PLT', 'MCV', 'PLR', 'HCT', 'HB', 'WBC', 'ABS_NEU', 'RDW', 'ABS_EOS', 'NLR']
        ml_features = [exp_col for tf in target_features for exp_col in expected_features if tf.lower() == exp_col.lower()]
        ml_input_df = final_df[ml_features]
        
        # 6. Prediksi Machine Learning
        # ➡️ PERBAIKAN WARNING: Gunakan .values agar model menerima array murni
        p_ml = ml_model.predict_proba(ml_input_df.values)[0]
        if len(p_ml) > 4: p_ml = p_ml[:4]
        if len(p_ml) < 4: p_ml = np.pad(p_ml, (0, 4 - len(p_ml)))
        p_ml = p_ml / p_ml.sum()
        
        # 7. Fusi Tri-Brid CDSS (Evaluasi Gejala & Aturan WHO)
        hct_calc = float(engineered_df['HCT'].iloc[0])
        p_sym = pillar_ii_symptom_score(data.symptoms)
        p_who = pillar_iii_who_rules(raw_dict["plt"], raw_dict["wbc"], hct_calc)
        
        w_ml, w_sym, w_who = data.weight_ml/100.0, data.weight_sym/100.0, data.weight_who/100.0
        p_final = (w_ml * p_ml) + (w_sym * p_sym) + (w_who * p_who)
        if p_final.sum() > 0: p_final = p_final / p_final.sum()
        
        pred_class = int(np.argmax(p_final))
        
        # 8. Analisis SHAP Linear (DIPERBAIKI ABSOLUT)
        background_data = pd.DataFrame(np.zeros((1, len(ml_features))), columns=ml_features)
        
        # Gunakan shap.Explainer umum (kompatibel untuk model linear multikelas)
        explainer = shap.Explainer(ml_model, background_data)
        shap_explanation = explainer(ml_input_df)
        
        if isinstance(shap_explanation.values, list):
            # Jika SHAP mengembalikan bentuk List
            sv_class = shap_explanation.values[pred_class][0] 
            base_val = shap_explanation.base_values[pred_class][0]
        else:
            # ➡️ PERBAIKAN ERROR: Jika SHAP mengembalikan Array 3D Numpy
            # Format shape: (1_Pasien, 10_Fitur, 4_Kelas)
            # Kita panggil: Pasien ke-0, Semua Fitur (:), Kelas ke-[pred_class]
            if len(shap_explanation.values.shape) == 3:
                sv_class = shap_explanation.values[0, :, pred_class]
                
                # Penanganan base_values (ekspektasi rata-rata)
                if len(np.array(shap_explanation.base_values).shape) == 2:
                    base_val = shap_explanation.base_values[0, pred_class]
                else:
                    base_val = shap_explanation.base_values[pred_class]
            else:
                # Fallback aman
                sv_class = shap_explanation.values[0]
                base_val = shap_explanation.base_values[0]

        single_expl = shap.Explanation(
            values=sv_class, 
            base_values=base_val, 
            data=ml_input_df.iloc[0].values, 
            feature_names=ml_features
        )
        # 9. Visualisasi SHAP
        plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(single_expl, max_display=7, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # 10. Teks Eksplanasi (CLIX-M)
        sv_vals = single_expl.values
        sorted_idx = np.argsort(np.abs(sv_vals))[::-1][:3]
        explanations = []
        for idx in sorted_idx:
            feat_name = ml_features[idx]
            explanations.append(get_detailed_explanation(feat_name, sv_vals[idx], pred_class))
            
        # 11. Bukti Komputasi
        calc_results = {
            "Hematokrit (HCT)": round(hct_calc, 2),
            "Mean Corpuscular Hemoglobin (MCH)": round(float(engineered_df['MCH'].iloc[0]), 2),
            "MCH Concentration (MCHC)": round(float(engineered_df['MCHC'].iloc[0]), 2),
            "Absolut Neutrofil": round(float(engineered_df['ABS_NEU'].iloc[0]), 2),
            "Absolut Limfosit": round(float(engineered_df['ABS_LYM'].iloc[0]), 2),
            "Absolut Monosit": round(float(engineered_df['ABS_MON'].iloc[0]), 2),
            "Absolut Eosinofil": round(float(engineered_df['ABS_EOS'].iloc[0]), 2),
            "Neutrophil-Lymphocyte Ratio (NLR)": round(float(engineered_df['NLR'].iloc[0]), 2),
            "Platelet-Lymphocyte Ratio (PLR)": round(float(engineered_df['PLR'].iloc[0]), 2),
            "Monocyte-Lymphocyte Ratio (MLR)": round(float(engineered_df['MLR'].iloc[0]), 2)
        }
            
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
