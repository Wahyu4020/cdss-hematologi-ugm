from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import os
import io
import json
import pickle
import base64
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── CLASS TRANSFORMER ───────────────────────────────────────────────────────
# Menghitung fitur turunan dan mengembalikan seluruh kolom dengan nama dataset.
# Persentase hitung jenis (NEU%, LYM%, MON%, EOS%) ikut diteruskan karena
# LYM% merupakan salah satu fitur model final.
class CBCCalculatorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        rbc, hb, mcv, wbc = X["rbc"], X["hb"], X["mcv"], X["wbc"]
        X["hct"] = rbc * mcv / 10
        X["mch"] = np.where(rbc > 0, hb / rbc * 10, np.nan)
        X["mchc"] = np.where(X["hct"] > 0, hb / X["hct"] * 100, np.nan)
        for k in ["neu", "lym", "mon", "eos"]:
            X[f"abs_{k}"] = X[k] / 100.0 * wbc
        lym_ok = X["abs_lym"] > 0
        # Rasio tidak terdefinisi bila limfosit absolut <= 0: diisi NaN (bukan 0)
        X["nlr"] = np.where(lym_ok, X["abs_neu"] / X["abs_lym"], np.nan)
        X["plr"] = np.where(lym_ok, X["plt"] / X["abs_lym"], np.nan)
        X["mlr"] = np.where(lym_ok, X["abs_mon"] / X["abs_lym"], np.nan)
        return X.rename(columns=MAP_WEB_TO_DATASET)


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
    "Asimptomatik (tidak ada keluhan klinis)": [0.50, 0.0, 0.0, 0.0],
}
ASIMPTOMATIK = "Asimptomatik (tidak ada keluhan klinis)"

MAP_WEB_TO_DATASET = {
    "gender": "L/P", "age": "Umur", "hb": "HB", "rbc": "RBC", "mcv": "MCV",
    "rdw": "RDW", "wbc": "WBC", "plt": "PLT", "neu": "NEU%", "lym": "LYM%",
    "mon": "MON%", "eos": "EOS%", "hct": "HCT", "mch": "MCH", "mchc": "MCHC",
    "abs_neu": "ABS_NEU", "abs_lym": "ABS_LYM", "abs_mon": "ABS_MON",
    "abs_eos": "ABS_EOS", "nlr": "NLR", "plr": "PLR", "mlr": "MLR",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(nama):
    with open(os.path.join(BASE_DIR, nama), "rb") as f:
        return pickle.load(f)


try:
    scaler = _load("std_scaler.pkl")
    imputer = _load("knn_imputer.pkl")
    selector = _load("mi_selector.pkl")
    ml_model = _load("lr_model.pkl")
    with open(os.path.join(BASE_DIR, "feature_list.json")) as f:
        FEATURE_INFO = json.load(f)
    FEATURES_ALL = FEATURE_INFO["features_all"]            # 23 kandidat fitur (urutan pelatihan)
    FEATURES_SEL = FEATURE_INFO["features_selected"]       # 10 fitur model final
    SEL_IDX = FEATURE_INFO["selected_indices"]
    assert list(np.where(selector.get_support())[0]) == SEL_IDX, \
        "mi_selector.pkl tidak sesuai dengan feature_list.json"

    # Data latih (393 sampel) yang tersimpan pada KNNImputer, dipakai untuk:
    # (1) background SHAP dan (2) rentang nilai data latih.
    train_scaled = imputer.transform(imputer._fit_X)
    train_raw = scaler.inverse_transform(imputer._fit_X)
    TRAIN_MIN = dict(zip(FEATURES_ALL, np.nanmin(train_raw, axis=0)))
    TRAIN_MAX = dict(zip(FEATURES_ALL, np.nanmax(train_raw, axis=0)))
    shap_background = selector.transform(train_scaled)
    explainer = shap.LinearExplainer(ml_model, shap_background)
    print("Model final berhasil dimuat:", FEATURES_SEL)
except Exception as e:
    print(f"Error loading models: {e}")
    raise


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


# ─── PILAR III: ATURAN WHO ───────────────────────────────────────────────────
def pillar_iii_who_rules(plt_val, wbc_val, hct_val, gender_val, age_val):
    if np.isnan(plt_val) or np.isnan(wbc_val) or np.isnan(hct_val):
        return np.ones(4) / 4.0

    raw = np.zeros(4, dtype=float)
    hct_limit = 50.0 if gender_val == 1.0 else 46.0

    if not np.isnan(age_val) and age_val < 1.0:
        wbc_min, wbc_max = 6.0, 17.0
    elif not np.isnan(age_val) and age_val <= 12.0:
        wbc_min, wbc_max = 5.0, 13.0
    else:
        wbc_min, wbc_max = 4.0, 11.0

    # 1. Aturan Dengue
    if plt_val < 100 and wbc_val < wbc_min and hct_val > hct_limit: raw[2] = 1.0
    elif plt_val < 100 and wbc_val < wbc_min: raw[2] = 0.6
    elif plt_val < 100 and hct_val > hct_limit: raw[2] = 0.4

    # 2. Aturan ITP
    if plt_val < 100 and wbc_min <= wbc_val <= wbc_max and 35 <= hct_val <= hct_limit: raw[1] = 1.0
    elif plt_val < 100 and raw[2] == 0: raw[1] = 0.5

    # 3. Aturan Trombositosis
    if plt_val > 600: raw[3] = 1.0
    elif plt_val > 450: raw[3] = 0.7

    # 4. Aturan Normal
    if (150 <= plt_val <= 400) and (wbc_min <= wbc_val <= wbc_max) and (35 <= hct_val <= hct_limit): raw[0] = 1.0
    elif raw.sum() == 0: raw[0] = 0.3

    return raw / raw.sum() if raw.sum() > 0 else np.ones(4) / 4.0


# ─── PILAR II: SKOR GEJALA ───────────────────────────────────────────────────
def pillar_ii_symptom_score(selected_symptoms):
    """Bobot gejala dijumlahkan per kelas lalu dinormalisasi (jumlah = 1).
    Tanpa gejala -> 0,25 untuk setiap kelas.
    Asimptomatik -> vektor asimptomatik yang dinormalisasi ([1, 0, 0, 0])."""
    if not selected_symptoms:
        return np.ones(4) / 4.0
    if ASIMPTOMATIK in selected_symptoms:
        raw = np.array(SYMPTOM_WEIGHTS[ASIMPTOMATIK], dtype=float)
    else:
        raw = np.zeros(4, dtype=float)
        for s in selected_symptoms:
            if s in SYMPTOM_WEIGHTS:
                raw += np.array(SYMPTOM_WEIGHTS[s])
    return raw / raw.sum() if raw.sum() > 0 else np.ones(4) / 4.0


def parse_weight(val, default):
    try:
        w = float(val) / 100.0
        return w if w >= 0 else 0.0
    except (TypeError, ValueError):
        return default


# ─── NARASI CLIX-M ───────────────────────────────────────────────────────────
def get_detailed_explanation(feature_name, shap_value, pred_class, raw_val):
    feat_upper = feature_name.upper()

    if pred_class == 0:
        if shap_value > 0:
            direction = "memperkuat probabilitas bahwa pasien dalam kondisi Normal."
        else:
            direction = "sedikit menurunkan keyakinan sistem, karena nilai ini menyimpang dari titik rata-rata pasien sehat di dalam dataset."
    else:
        if shap_value > 0:
            direction = "mendorong/meningkatkan probabilitas diagnosis penyakit ini."
        else:
            direction = "menahan/mengurangi probabilitas diagnosis penyakit ini."

    if raw_val is None or pd.isna(raw_val):
        return f"**{feat_upper} (tidak tersedia):** Nilai diisi oleh imputasi KNN dan {direction}"

    base_text = f"**{feat_upper} ({round(float(raw_val), 2)}):** Nilai ini {direction}"

    if feat_upper == "PLT":
        if raw_val < 150:
            if pred_class == 1: return base_text + " Penurunan trombosit (trombositopenia) terisolasi adalah tanda khas ITP akibat destruksi keping darah oleh autoimun."
            elif pred_class == 2: return base_text + " Trombositopenia sangat lazim pada fase akut Dengue akibat supresi sumsum tulang dan destruksi perifer."
            else: return base_text + " Trombositopenia mengindikasikan tingginya tingkat destruksi keping darah atau kegagalan produksi."
        elif raw_val > 450:
            return base_text + " Peningkatan trombosit (trombositosis) mengonfirmasi hiperaktivitas sumsum tulang, sering muncul sebagai respons reaktif terhadap inflamasi sistemik atau infeksi."
        else:
            return base_text + " Jumlah trombosit berada dalam rentang normal, menunjukkan fungsi hemostasis (pembekuan darah) primer yang stabil."

    elif feat_upper == "WBC":
        if raw_val < 4.0:
            if pred_class == 2: return base_text + " Penurunan sel darah putih (leukopenia) merupakan penanda patognomonik awal pada infeksi virus akut seperti Dengue."
            else: return base_text + " Leukopenia dapat terjadi akibat supresi sumsum tulang atau efek toksik sistemik."
        elif raw_val > 11.0:
            return base_text + " Peningkatan sel darah putih (leukositosis) menandakan respons imun tubuh yang sangat aktif akibat infeksi bakteri atau inflamasi hebat."
        else:
            return base_text + " Jumlah leukosit dalam rentang fisiologis menandakan fungsi imunitas bawaan (innate immunity) beroperasi normal."

    elif feat_upper == "RBC":
        if raw_val < 4.0: return base_text + " Penurunan hitung eritrosit memperkuat indikasi anemia, riwayat perdarahan, atau supresi pembentukan darah merah."
        elif raw_val > 5.5: return base_text + " Peningkatan hitung eritrosit menandakan hiperaktivitas eritropoiesis atau hemokonsentrasi."
        else: return base_text + " Jumlah eritrosit berada pada ambang batas fisiologis yang sehat."

    elif feat_upper == "MCV":
        if raw_val < 80: return base_text + " MCV rendah (mikrositik) sering menjadi rujukan penyakit penyerta seperti anemia defisiensi besi."
        elif raw_val > 100: return base_text + " MCV tinggi (makrositik) mengindikasikan kemungkinan defisiensi B12/folat."
        else: return base_text + " Ukuran sel darah merah proporsional (normositik)."

    elif feat_upper == "MCH":
        if raw_val < 27: return base_text + " MCH rendah merepresentasikan sel darah merah yang hipokromik (pucat) akibat kurangnya massa hemoglobin."
        elif raw_val > 33: return base_text + " MCH tinggi (hiperkromik) umumnya sejalan dengan membesarnya ukuran sel darah merah (makrositik)."
        else: return base_text + " Kepadatan hemoglobin per sel darah merah terpantau normokromik."

    elif feat_upper == "ABS_NEU":
        if raw_val < 2.0: return base_text + " Penurunan neutrofil (neutropenia) sangat khas terjadi pada fase akut infeksi virus akibat supresi sumsum tulang."
        elif raw_val > 7.5: return base_text + " Peningkatan neutrofil (neutrofilia) adalah respons garda terdepan sistem imun terhadap bakteri piogenik atau peradangan jaringan."
        else: return base_text + " Jumlah neutrofil normal menandakan tidak ada lonjakan infeksi bakteri."

    elif feat_upper == "ABS_EOS":
        if raw_val < 0.05: return base_text + " Eosinopenia adalah temuan reaktif terhadap stres akut primer atau inflamasi sistemik."
        elif raw_val > 0.5: return base_text + " Eosinofilia umumnya merupakan penanda biologi khas untuk reaksi alergi atau infeksi parasit."
        else: return base_text + " Kadar eosinofil wajar tanpa indikasi alergi."

    elif feat_upper in ["NLR", "PLR", "MLR"]:
        korelasi = "NLR (Rasio Neutrofil/Limfosit)" if feat_upper == "NLR" else "PLR (Rasio Trombosit/Limfosit)" if feat_upper == "PLR" else "MLR (Rasio Monosit/Limfosit)"
        if raw_val > 3.0:
            return base_text + f" Nilai {korelasi} yang tinggi secara literatur digunakan sebagai biomarker prediktif kuat adanya derajat keparahan inflamasi sistemik pada pasien."
        elif raw_val < 1.0:
            return base_text + f" Nilai {korelasi} yang sangat rendah sering mengikuti pola limfositosis relatif pada infeksi virus."
        else:
            return base_text + f" {korelasi} berada dalam ekuilibrium (keseimbangan) fisiologis."

    # Umur dan LYM% tidak memiliki aturan narasi patofisiologis:
    # narasi hanya memuat arah kontribusi SHAP.
    return base_text


# ─── ENDPOINT UTAMA ──────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict_diagnosis(data: PatientInput):
    try:
        raw_input = data.model_dump(exclude={"symptoms", "weight_ml", "weight_sym", "weight_who"})
        raw_dict = {}
        for k, v in raw_input.items():
            try:
                raw_dict[k.lower()] = float(v) if v is not None and str(v).strip() not in ["", "-"] else np.nan
            except ValueError:
                raw_dict[k.lower()] = np.nan

        # 1. Fitur turunan
        engineered_df = CBCCalculatorTransformer().fit_transform(pd.DataFrame([raw_dict]))

        # 2. Susun 23 kandidat fitur sesuai urutan pelatihan (feature_list.json)
        aligned = np.array([[engineered_df[c].iloc[0] if c in engineered_df.columns else np.nan
                             for c in FEATURES_ALL]], dtype=float)
        aligned_df = pd.DataFrame(aligned, columns=FEATURES_ALL)

        # 3. Penskalaan -> imputasi -> seleksi MI -> prediksi
        scaled = scaler.transform(aligned)
        imputed = imputer.transform(scaled)
        ml_input = selector.transform(imputed)
        raw_imputed = pd.DataFrame(scaler.inverse_transform(imputed), columns=FEATURES_ALL)

        p_ml = ml_model.predict_proba(ml_input)[0]

        # Peringatan: fitur model yang kosong (diimputasi) dan nilai di luar rentang data latih
        peringatan = []
        for c in FEATURES_SEL:
            v = aligned_df[c].iloc[0]
            if np.isnan(v):
                peringatan.append(f"{c} tidak tersedia dan diisi dengan imputasi KNN.")
            elif v < TRAIN_MIN[c] or v > TRAIN_MAX[c]:
                peringatan.append(f"{c} = {round(float(v), 2)} berada di luar rentang data latih "
                                  f"({round(float(TRAIN_MIN[c]), 2)} - {round(float(TRAIN_MAX[c]), 2)}); "
                                  f"prediksi Pilar I merupakan ekstrapolasi.")

        # 4. Pilar II dan Pilar III
        hct_calc = float(engineered_df["HCT"].iloc[0])
        p_sym = pillar_ii_symptom_score(data.symptoms or [])
        p_who = pillar_iii_who_rules(raw_dict["plt"], raw_dict["wbc"], hct_calc,
                                     raw_dict.get("gender", np.nan), raw_dict.get("age", np.nan))

        # 5. Fusi berbobot (bobot dari antarmuka, bawaan 55/25/20)
        w_ml = parse_weight(data.weight_ml, 0.55)
        w_sym = parse_weight(data.weight_sym, 0.25)
        w_who = parse_weight(data.weight_who, 0.20)
        if w_ml + w_sym + w_who == 0:
            w_ml, w_sym, w_who = 0.55, 0.25, 0.20
        p_final = w_ml * p_ml + w_sym * p_sym + w_who * p_who
        p_final = p_final / p_final.sum()
        pred_class = int(np.argmax(p_final))

        # 6. SHAP lokal (LinearExplainer, background = data latih) untuk kelas terprediksi
        sv = explainer(ml_input)
        values = np.asarray(sv.values)
        base_values = np.asarray(sv.base_values)
        sv_class = values[0, :, pred_class] if values.ndim == 3 else values[0]
        base_val = base_values[0, pred_class] if base_values.ndim == 2 else base_values[pred_class]
        raw_sel = raw_imputed[FEATURES_SEL].iloc[0].values

        single_expl = shap.Explanation(values=sv_class, base_values=float(base_val),
                                       data=np.round(raw_sel, 2), feature_names=FEATURES_SEL)
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(single_expl, max_display=10, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")

        sorted_idx = np.argsort(np.abs(sv_class))[::-1][:5]
        explanations = [get_detailed_explanation(FEATURES_SEL[i], sv_class[i], pred_class,
                                                 aligned_df[FEATURES_SEL[i]].iloc[0])
                        for i in sorted_idx]

        def safe_round(val): return round(float(val), 2) if pd.notna(val) else "N/A"
        calc_results = {
            "Hematokrit (HCT)": safe_round(engineered_df["HCT"].iloc[0]),
            "Mean Corpuscular Hemoglobin (MCH)": safe_round(engineered_df["MCH"].iloc[0]),
            "MCH Concentration (MCHC)": safe_round(engineered_df["MCHC"].iloc[0]),
            "Absolut Neutrofil": safe_round(engineered_df["ABS_NEU"].iloc[0]),
            "Absolut Limfosit": safe_round(engineered_df["ABS_LYM"].iloc[0]),
            "Absolut Monosit": safe_round(engineered_df["ABS_MON"].iloc[0]),
            "Absolut Eosinofil": safe_round(engineered_df["ABS_EOS"].iloc[0]),
            "Neutrophil-Lymphocyte Ratio (NLR)": safe_round(engineered_df["NLR"].iloc[0]),
            "Platelet-Lymphocyte Ratio (PLR)": safe_round(engineered_df["PLR"].iloc[0]),
            "Monocyte-Lymphocyte Ratio (MLR)": safe_round(engineered_df["MLR"].iloc[0]),
        }

        breakdown_dict = {}
        if w_ml > 0: breakdown_dict["Pilar_1_ML"] = round(float(p_ml[pred_class] * 100), 2)
        if w_sym > 0: breakdown_dict["Pilar_2_Sym"] = round(float(p_sym[pred_class] * 100), 2)
        if w_who > 0: breakdown_dict["Pilar_3_WHO"] = round(float(p_who[pred_class] * 100), 2)

        return {
            "status": "success", "diagnosis": CLASS_NAMES[pred_class],
            "probabilitas_final": round(float(p_final[pred_class] * 100), 2),
            "breakdown": breakdown_dict, "shap_image": image_base64,
            "clix_m_text": explanations, "kalkulasi_fisiologis": calc_results,
            "peringatan": peringatan,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
