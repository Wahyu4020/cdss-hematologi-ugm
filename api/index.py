from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union
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
    Gender: Union[float, str, None] = None
    Age: Union[float, str, None] = None
    hb: Union[float, str, None] = None
    rbc: Union[float, str, None] = None
    mcv: Union[float, str, None] = None
    rdw: Union[float, str, None] = None
    wbc: Union[float, str, None] = None
    neu: Union[float, str, None] = None
    lym: Union[float, str, None] = None
    mon: Union[float, str, None] = None
    eos: Union[float, str, None] = None
    plt: Union[float, str, None] = None
    symptoms: Optional[List[str]] = []
    weight_ml: Union[float, str, None] = 55.0
    weight_sym: Union[float, str, None] = 25.0
    weight_who: Union[float, str, None] = 20.0

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
    direction = "mendorong probabilitas" if shap_value > 0 else "menahan/mengurangi risiko"
    feat_upper = feature_name.upper()
    
    base_text = f"**{feat_upper} ({round(raw_val, 2)}):** Nilai ini {direction} keputusan diagnosis."
    
    # ─── 1. TROMBOSIT (PLT) ───
    if feat_upper == "PLT":
        if raw_val < 150:
            if pred_class == 1: return base_text + " Penurunan trombosit (trombositopenia) terisolasi adalah tanda khas ITP akibat destruksi keping darah oleh autoimun."
            elif pred_class == 2: return base_text + " Trombositopenia sangat lazim pada fase akut Dengue akibat supresi sumsum tulang dan destruksi perifer."
            else: return base_text + " Trombositopenia mengindikasikan tingginya tingkat destruksi keping darah atau kegagalan produksi."
        elif raw_val > 450: return base_text + " Peningkatan trombosit (trombositosis) mengonfirmasi hiperaktivitas sumsum tulang, sering muncul sebagai respons reaktif terhadap inflamasi sistemik."
        else: return base_text + " Jumlah trombosit berada dalam rentang normal, menunjukkan fungsi hemostasis primer yang stabil."
                
    # ─── 2. LEUKOSIT (WBC) ───
    elif feat_upper == "WBC":
        if raw_val < 4.0:
            if pred_class == 2: return base_text + " Penurunan sel darah putih (leukopenia) merupakan penanda patognomonik awal pada infeksi virus akut seperti Dengue."
            else: return base_text + " Leukopenia dapat terjadi akibat supresi sumsum tulang atau efek toksik sistemik."
        elif raw_val > 11.0: return base_text + " Peningkatan sel darah putih (leukositosis) menandakan respons imun tubuh yang sangat aktif akibat infeksi bakteri atau inflamasi hebat."
        else: return base_text + " Jumlah leukosit dalam rentang fisiologis menandakan fungsi imunitas bawaan beroperasi normal."

    # ─── 3. HEMOGLOBIN (HB) & HEMATOKRIT (HCT) ───
    elif feat_upper == "HB":
        if raw_val < 12.0: return base_text + " Penurunan kadar hemoglobin (anemia) dapat diakibatkan oleh komplikasi perdarahan klinis atau defisiensi zat besi."
        elif raw_val > 16.0: return base_text + " Kadar hemoglobin di atas normal mengindikasikan polisitemia atau hemokonsentrasi akibat dehidrasi/kebocoran plasma."
        else: return base_text + " Kadar hemoglobin normal menandakan kapasitas transportasi oksigen sistemik tidak terganggu."
            
    elif feat_upper == "HCT":
        if raw_val < 35: return base_text + " Penurunan hematokrit mencerminkan kondisi hemodilusi (kelebihan cairan) atau anemia seluler."
        elif raw_val > 45 and pred_class == 2: return base_text + " Peningkatan hematokrit (hemokonsentrasi) adalah tanda bahaya mutlak pada Dengue yang merepresentasikan sindrom kebocoran plasma."
        elif raw_val > 45: return base_text + " Hematokrit tinggi menandakan tingginya viskositas (kekentalan) darah."
        else: return base_text + " Viskositas dan persentase volume sel darah merah terpantau seimbang."

    # ─── 4. ERITROSIT (RBC) ───
    elif feat_upper == "RBC":
        if raw_val < 4.0: return base_text + " Penurunan hitung eritrosit memperkuat indikasi anemia, riwayat perdarahan, atau supresi pembentukan darah merah."
        elif raw_val > 5.5: return base_text + " Peningkatan hitung eritrosit menandakan hiperaktivitas eritropoiesis atau hemokonsentrasi."
        else: return base_text + " Jumlah eritrosit berada pada ambang batas fisiologis yang sehat."
            
    # ─── 5. INDEKS ERITROSIT (MCV, MCH, MCHC, RDW) ───
    elif feat_upper == "MCV":
        if raw_val < 80: return base_text + " MCV rendah (mikrositik) sering menjadi rujukan penyakit penyerta seperti anemia defisiensi besi."
        elif raw_val > 100: return base_text + " MCV tinggi (makrositik) mengindikasikan kemungkinan defisiensi B12/folat."
        else: return base_text + " Ukuran sel darah merah proporsional (normositik)."
            
    elif feat_upper == "MCH":
        if raw_val < 27: return base_text + " MCH rendah merepresentasikan sel darah merah yang hipokromik (pucat) akibat kurangnya massa hemoglobin."
        elif raw_val > 33: return base_text + " MCH tinggi (hiperkromik) umumnya sejalan dengan membesarnya ukuran sel darah merah (makrositik)."
        else: return base_text + " Kepadatan hemoglobin per sel darah merah terpantau normokromik."
            
    elif feat_upper == "MCHC":
        if raw_val < 32: return base_text + " Penurunan MCHC mengonfirmasi kondisi hipokromia absolut."
        elif raw_val > 36: return base_text + " MCHC sangat tinggi dapat mengindikasikan sferositosis autoimun atau hemolisis."
        else: return base_text + " Konsentrasi hemoglobin intraseluler seimbang dengan volume sel."

    elif feat_upper == "RDW":
        if raw_val > 14.5: return base_text + " RDW tinggi (anisositosis) menunjukkan variasi ukuran sel darah merah yang abnormal, sangat berkaitan dengan stres inflamasi kronis atau pemulihan perdarahan."
        elif raw_val < 11.5: return base_text + " RDW rendah menunjukkan sel darah merah yang sangat seragam."
        else: return base_text + " Distribusi ukuran eritrosit normal dan seragam."

    # ─── 6. DIFERENSIAL LEUKOSIT ABSOLUT ───
    elif feat_upper == "ABS_NEU":
        if raw_val < 2.0: return base_text + " Penurunan neutrofil (neutropenia) sangat khas terjadi pada fase akut infeksi virus akibat supresi sumsum tulang."
        elif raw_val > 7.5: return base_text + " Peningkatan neutrofil (neutrofilia) adalah respons garda terdepan sistem imun terhadap bakteri piogenik atau peradangan jaringan."
        else: return base_text + " Jumlah neutrofil normal menandakan tidak ada lonjakan infeksi bakteri."
            
    elif feat_upper == "ABS_LYM":
        if raw_val < 1.0: return base_text + " Limfopenia (penurunan limfosit) adalah respons awal imunitas akibat stres infeksi virus sistemik yang parah."
        elif raw_val > 4.0: return base_text + " Limfositosis mengindikasikan mobilisasi aktif imunitas adaptif (sel T dan sel B) untuk membersihkan sisa virus atau fase pemulihan infeksi."
        else: return base_text + " Jumlah limfosit berada dalam batas kekebalan adaptif yang normal."

    elif feat_upper == "ABS_MON":
        if raw_val < 0.2: return base_text + " Monositopenia menunjukkan penurunan sel fagosit, sering kali terjadi pada infeksi akut parah."
        elif raw_val > 0.8: return base_text + " Monositosis menunjukkan hiperaktivitas makrofag pembersih jaringan, yang umum pada masa pemulihan inflamasi."
        else: return base_text + " Kadar monosit normal."
        
    elif feat_upper == "ABS_EOS":
        if raw_val < 0.05: return base_text + " Eosinopenia adalah temuan reaktif terhadap stres akut primer atau inflamasi sistemik."
        elif raw_val > 0.5: return base_text + " Eosinofilia umumnya merupakan penanda biologi khas untuk reaksi alergi atau infeksi parasit."
        else: return base_text + " Kadar eosinofil wajar tanpa indikasi alergi."
        
    # ─── 7. RASIO INFLAMASI (NLR, PLR, MLR) ───
    elif feat_upper in ["NLR", "PLR", "MLR"]:
        korelasi = "NLR (Rasio Neutrofil/Limfosit)" if feat_upper == "NLR" else "PLR (Rasio Trombosit/Limfosit)" if feat_upper == "PLR" else "MLR (Rasio Monosit/Limfosit)"
        if raw_val > 3.0: return base_text + f" Nilai {korelasi} yang tinggi secara literatur digunakan sebagai biomarker prediktif kuat adanya derajat keparahan inflamasi sistemik pada pasien."
        elif raw_val < 1.0: return base_text + f" Nilai {korelasi} yang sangat rendah sering mengikuti pola limfositosis relatif pada infeksi virus."
        else: return base_text + f" {korelasi} berada dalam batas keseimbangan (ekuilibrium) fisiologis."
        
    return base_text

# ─── ENDPOINT UTAMA ──────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict_diagnosis(data: PatientInput):
    try:
        if hasattr(data, "model_dump"):
            raw_input = data.model_dump(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
        else:
            raw_input = data.dict(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
            
        raw_dict = {}
        for k, v in raw_input.items():
            val_str = str(v).strip()
            if v is None or val_str in ["", "-"]:
                raw_dict[k.lower()] = np.nan
            else:
                try: raw_dict[k.lower()] = float(v)
                except ValueError: raw_dict[k.lower()] = np.nan
                    
        raw_df = pd.DataFrame([raw_dict])
        
        cbc_calc = CBCCalculatorTransformer()
        cbc_calc.fit(raw_df) 
        engineered_df = cbc_calc.transform(raw_df)
        
        engineered_df.columns = [c.lower() for c in engineered_df.columns]
        engineered_df = engineered_df.rename(columns=MAP_WEB_TO_DATASET)
        
        expected_features = list(scaler.feature_names_in_)
        aligned_df = pd.DataFrame(columns=expected_features)
        aligned_df.loc[0] = np.nan
        
        for expected_col in expected_features:
            if expected_col in engineered_df.columns:
                aligned_df.at[0, expected_col] = engineered_df.iloc[0][expected_col]
        
        scaled_data = scaler.transform(aligned_df)
        imputed_data = imputer.transform(scaled_data)
        final_df = pd.DataFrame(imputed_data, columns=expected_features)
        
        raw_imputed_data = scaler.inverse_transform(final_df)
        raw_imputed_df = pd.DataFrame(raw_imputed_data, columns=expected_features)
        
        target_features = ['PLT', 'MCV', 'PLR', 'HCT', 'HB', 'WBC', 'ABS_NEU', 'RDW', 'ABS_EOS', 'NLR']
        ml_features = [exp_col for tf in target_features for exp_col in expected_features if tf.lower() == exp_col.lower()]
        ml_input_df = final_df[ml_features]
        
        p_ml = ml_model.predict_proba(ml_input_df.values)[0]
        if len(p_ml) > 4: p_ml = p_ml[:4]
        if len(p_ml) < 4: p_ml = np.pad(p_ml, (0, 4 - len(p_ml)))
        p_ml = p_ml / p_ml.sum()
        
        hct_calc = float(engineered_df['HCT'].iloc[0]) if pd.notna(engineered_df['HCT'].iloc[0]) else np.nan
        safe_symptoms = data.symptoms if data.symptoms is not None else []
        p_sym = pillar_ii_symptom_score(safe_symptoms)
        p_who = pillar_iii_who_rules(raw_dict["plt"], raw_dict["wbc"], hct_calc)
        
        try: w_ml = float(data.weight_ml) / 100.0
        except: w_ml = 0.55
        try: w_sym = float(data.weight_sym) / 100.0
        except: w_sym = 0.25
        try: w_who = float(data.weight_who) / 100.0
        except: w_who = 0.20
        
        p_final = (w_ml * p_ml) + (w_sym * p_sym) + (w_who * p_who)
        if p_final.sum() > 0: p_final = p_final / p_final.sum()
        pred_class = int(np.argmax(p_final))
        
        background_data = pd.DataFrame(np.zeros((1, len(ml_features))), columns=ml_features)
        explainer = shap.Explainer(ml_model, background_data)
        shap_explanation = explainer(ml_input_df)
        
        if isinstance(shap_explanation.values, list):
            sv_class = shap_explanation.values[pred_class][0] 
            base_val = shap_explanation.base_values[pred_class][0]
        else:
            if len(shap_explanation.values.shape) == 3:
                sv_class = shap_explanation.values[0, :, pred_class]
                base_val = shap_explanation.base_values[0, pred_class] if len(np.array(shap_explanation.base_values).shape) == 2 else shap_explanation.base_values[pred_class]
            else:
                sv_class = shap_explanation.values[0]
                base_val = shap_explanation.base_values[0]

        single_expl = shap.Explanation(values=sv_class, base_values=base_val, data=ml_input_df.iloc[0].values, feature_names=ml_features)
        
        # ➡️ SHAP VISUALISASI DIPERBARUI: Tinggi 6 dan tampilkan 10 fitur
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(single_expl, max_display=10, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        sv_vals = single_expl.values
        # ➡️ PENJELASAN NARASI DIPERBARUI: Ambil 5 fitur teratas
        sorted_idx = np.argsort(np.abs(sv_vals))[::-1][:5]
        explanations = [get_detailed_explanation(ml_features[idx], sv_vals[idx], pred_class, raw_imputed_df.iloc[0][ml_features[idx]]) for idx in sorted_idx]
        
        def safe_round(val): return round(float(val), 2) if pd.notna(val) else "N/A"
        calc_results = {
            "Hematokrit (HCT)": safe_round(engineered_df['HCT'].iloc[0]),
            "Mean Corpuscular Hemoglobin (MCH)": safe_round(engineered_df['MCH'].iloc[0]),
            "MCH Concentration (MCHC)": safe_round(engineered_df['MCHC'].iloc[0]),
            "Absolut Neutrofil": safe_round(engineered_df['ABS_NEU'].iloc[0]),
            "Absolut Limfosit": safe_round(engineered_df['ABS_LYM'].iloc[0]),
            "Absolut Monosit": safe_round(engineered_df['ABS_MON'].iloc[0]),
            "Absolut Eosinofil": safe_round(engineered_df['ABS_EOS'].iloc[0]),
            "Neutrophil-Lymphocyte Ratio (NLR)": safe_round(engineered_df['NLR'].iloc[0]),
            "Platelet-Lymphocyte Ratio (PLR)": safe_round(engineered_df['PLR'].iloc[0]),
            "Monocyte-Lymphocyte Ratio (MLR)": safe_round(engineered_df['MLR'].iloc[0])
        }
        
        breakdown_dict = {}
        if w_ml > 0: breakdown_dict["Pilar_1_ML"] = round(float(p_ml[pred_class]*100), 2)
        if w_sym > 0: breakdown_dict["Pilar_2_Sym"] = round(float(p_sym[pred_class]*100), 2)
        if w_who > 0: breakdown_dict["Pilar_3_WHO"] = round(float(p_who[pred_class]*100), 2)
            
        return {
            "status": "success", "diagnosis": CLASS_NAMES[pred_class],
            "probabilitas_final": round(float(p_final[pred_class]*100), 2),
            "breakdown": breakdown_dict, "shap_image": image_base64,
            "clix_m_text": explanations, "kalkulasi_fisiologis": calc_results 
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
