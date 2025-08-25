#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning (Keras) para prever Volume a partir de DAP, H e Idade.
- Alvo em log: ln(Volume)
- Validação hold-out + leave-one-age-out por Idade
- Regularização: BatchNorm + Dropout + L2 + EarlyStopping
- Opcional: Smearing de Duan para voltar à escala original sem viés

Requisitos: tensorflow, pandas, numpy, scikit-learn, openpyxl
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

ARQUIVO = "dados.xlsx"
PLANILHA = "Planilha1"
RANDOM_STATE = 42
EPOCHS = 1000
BATCH_SIZE = 64

def set_seeds(seed=RANDOM_STATE):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)

def carregar_dados():
    p = Path(ARQUIVO)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p.resolve()}")
    df = pd.read_excel(p, sheet_name=PLANILHA)
    esperadas = {"Idade", "DAP", "H", "Volume"}
    if not esperadas.issubset(df.columns):
        raise ValueError(f"Colunas esperadas: {sorted(esperadas)}; encontradas: {list(df.columns)}")
    return df

def preparar(df: pd.DataFrame):
    # Idade: "36 meses" -> 36 (int)
    df = df.copy()
    df["Idade_num"] = df["Idade"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    # filtros
    df = df[(df["DAP"] > 0) & (df["H"] > 0) & (df["Volume"] > 0) & df["Idade_num"].notna()].reset_index(drop=True)
    # alvo em log
    df["ln_V"] = np.log(df["Volume"])
    X = df[["DAP", "H", "Idade_num"]].values.astype("float32")
    y_log = df["ln_V"].values.astype("float32")
    grupos = df["Idade_num"].astype(int).values
    return df, X, y_log, grupos

def construir_modelo(input_dim: int, l2=1e-4, dropout=0.2):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)  # prediz ln(V)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model

def smearing_duan(residuos_log: np.ndarray) -> float:
    return float(np.mean(np.exp(residuos_log)))

def avaliar_escala_original(y_log_true, y_log_pred, smearing=None):
    # volta para a escala original; opcionalmente aplica smearing
    if smearing is None:
        y_pred = np.exp(y_log_pred)
    else:
        y_pred = smearing * np.exp(y_log_pred)
    y_true = np.exp(y_log_true)
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, rmse, y_true, y_pred

def holdout_treino_avaliacao(X, y_log):
    # split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=RANDOM_STATE)

    # padronização por treino
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_tr)
    X_ten = scaler.transform(X_te)

    # modelo
    model = construir_modelo(X_trn.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=3e-5)
    ]
    hist = model.fit(
        X_trn, y_tr,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks
    )

    # smearing calculado nos resíduos do treino (valendo para ln-escala)
    y_tr_pred = model.predict(X_trn, verbose=0).ravel()
    res_tr = y_tr - y_tr_pred
    smear = smearing_duan(res_tr)

    # avaliação no teste
    y_te_pred = model.predict(X_ten, verbose=0).ravel()
    r2, rmse, _, _ = avaliar_escala_original(y_te, y_te_pred, smearing=smear)

    print("\n" + "-"*72)
    print("DEEP LEARNING (hold-out 20%) [COM Idade: DAP, H, Idade_num]")
    print("-"*72)
    print(f"R² (orig): {r2:.6f}")
    print(f"RMSE (m³): {rmse:.6f}")
    print(f"Smearing de Duan (treino): {smear:.6f}")

    return model, scaler, smear, (r2, rmse)

def leave_one_age_out(df, X, y_log, grupos):
    idades = np.unique(grupos)
    resultados = []
    for idade in idades:
        # separa treino/teste por idade
        idx_te = (grupos == idade)
        idx_tr = ~idx_te
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y_log[idx_tr], y_log[idx_te]

        # scaler por treino
        scaler = StandardScaler()
        X_trn = scaler.fit_transform(X_tr)
        X_ten = scaler.transform(X_te)

        # modelo
        model = construir_modelo(X_trn.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=3e-5)
        ]
        model.fit(
            X_trn, y_tr,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=callbacks
        )

        # smearing por treino
        y_tr_pred = model.predict(X_trn, verbose=0).ravel()
        smear = smearing_duan(y_tr - y_tr_pred)

        # avaliação
        y_te_pred = model.predict(X_ten, verbose=0).ravel()
        r2, rmse, _, _ = avaliar_escala_original(y_te, y_te_pred, smearing=smear)
        resultados.append({"Idade_holdout_meses": int(idade), "R2": r2, "RMSE_m3": rmse, "N_teste": int(np.sum(idx_te))})

    res = pd.DataFrame(resultados).sort_values("Idade_holdout_meses").reset_index(drop=True)
    print("\n" + "#"*72)
    print("# LEAVE-ONE-AGE-OUT (DL em ln-volume + smearing)               #")
    print("#"*72)
    print(res.to_string(index=False))
    print(f" Médias -> R² médio={res['R2'].mean():.3f} | RMSE médio={res['RMSE_m3'].mean():.6f} m³")
    return res

def main():
    set_seeds()
    df = carregar_dados()
    df, X, y_log, grupos = preparar(df)

    # HOLD-OUT
    _, _, smear, (r2, rmse) = holdout_treino_avaliacao(X, y_log)

    # LOAO
    _ = leave_one_age_out(df, X, y_log, grupos)

if __name__ == "__main__":
    main()
