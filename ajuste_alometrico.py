#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensão científica do ajuste alométrico:
- Ajuste global (todos os dados) e por idade
- Comparações formais:
  (1) Critérios de informação: AIC/BIC
  (2) ANCOVA (efeito de Idade no intercepto) e ANCOVA com interações (efeito de Idade nas inclinações)
  (3) Teste de razão de verossimilhança (LR) manual entre modelos aninhados
  (4) Validação cruzada por grupos de Idade (leave-one-age-out)

Modelo base nos logs:
    ln(V) = ln(a) + b*ln(DAP) + c*ln(H) + erro
Retransformação: smearing de Duan para métricas na escala original.

Requisitos: pandas, numpy, scikit-learn, statsmodels, openpyxl, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from pathlib import Path
from typing import Dict, Tuple


# ------------------------------
# Utilidades base (ajuste alométrico)
# ------------------------------
def ajustar_modelo(df: pd.DataFrame) -> Tuple[Dict, sm.regression.linear_model.RegressionResultsWrapper]:
    """Ajusta ln(V) = ln(a) + b*ln(DAP) + c*ln(H) com OLS e calcula smearing.
    Retorna (resultados_dict, modelo_full_logs). Métricas calculadas com hold-out 80/20.
    """
    df = df.copy()
    df = df[(df["DAP"] > 0) & (df["H"] > 0) & (df["Volume"] > 0)].reset_index(drop=True)

    # Logs
    df["ln_V"] = np.log(df["Volume"])
    df["ln_DAP"] = np.log(df["DAP"])
    df["ln_H"] = np.log(df["H"])

    X = df[["ln_DAP", "ln_H"]]
    y = df["ln_V"]

    # Hold-out
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    ols_tr = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()

    # Smearing (nos resíduos de treino, escala log)
    residuos_tr = y_tr - ols_tr.predict(sm.add_constant(X_tr))
    smearing = float(np.mean(np.exp(residuos_tr)))

    # Métricas (escala original)
    y_te_orig = np.exp(y_te)
    yhat_te_orig = smearing * np.exp(ols_tr.predict(sm.add_constant(X_te)))
    r2_te = r2_score(y_te_orig, yhat_te_orig)
    rmse_te = root_mean_squared_error(y_te_orig, yhat_te_orig)

    # Ajuste final em todos os dados (para reportar a equação final)
    ols_full = sm.OLS(y, sm.add_constant(X)).fit()
    b0 = float(ols_full.params["const"])
    b_dap = float(ols_full.params["ln_DAP"])
    b_h = float(ols_full.params["ln_H"])

    # a efetivo com smearing
    a_eff = float(smearing * np.exp(b0))

    resultados = {
        "params_log": {
            "const_ln_a": b0,
            "b_dap": b_dap,
            "b_h": b_h
        },
        "smearing": smearing,
        "equacao": "V = a * DAP^b * H^c",
        "a": float(np.exp(b0)),
        "b": b_dap,
        "c": b_h,
        "a_efetivo_com_smearing": a_eff,
        "r2_teste_escala_original": r2_te,
        "rmse_teste_escala_original": rmse_te,
        "n_observacoes": int(len(df))
    }
    return resultados, ols_full


def prever_volume(dap, h, a_eff, b, c):
    """Prevê V usando V = a_eff * DAP^b * H^c (a_eff já inclui o smearing)."""
    dap = np.asarray(dap, dtype=float)
    h = np.asarray(h, dtype=float)
    return a_eff * np.power(dap, b) * np.power(h, c)


# ------------------------------
# Preparação e comparações
# ------------------------------
def preparar_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna dataframe com ln_V, ln_DAP, ln_H e Idade como categoria."""
    df = df.copy()
    df = df[(df["DAP"] > 0) & (df["H"] > 0) & (df["Volume"] > 0)].reset_index(drop=True)
    df["ln_V"] = np.log(df["Volume"])
    df["ln_DAP"] = np.log(df["DAP"])
    df["ln_H"] = np.log(df["H"])
    df["Idade"] = df["Idade"].astype("category")
    return df


def lr_test(restricted, full):
    """Teste de razão de verossimilhança (LR) para modelos aninhados."""
    lr = 2.0 * (full.llf - restricted.llf)
    df = int(full.df_model - restricted.df_model)
    p = chi2.sf(lr, df) if df > 0 else np.nan
    return lr, p, df


def ajustar_modelos_ancova(df_log: pd.DataFrame):
    """
    Ajusta três modelos nos logs:
      M0: sem Idade (global),
      M1: com Idade no intercepto,
      M2: com Idade no intercepto e interações (inclinações por Idade).
    Retorna dict com resultados e testes de comparação (LR manual).
    """
    # Modelo global (sem Idade)
    M0 = smf.ols("ln_V ~ ln_DAP + ln_H", data=df_log).fit()

    # ANCOVA: diferentes interceptos por Idade
    M1 = smf.ols("ln_V ~ ln_DAP + ln_H + C(Idade)", data=df_log).fit()

    # ANCOVA com interações: diferentes inclinações por Idade (b e c variam)
    M2 = smf.ols("ln_V ~ C(Idade) * ln_DAP + C(Idade) * ln_H", data=df_log).fit()

    # Testes LR manuais (evita bug do compare_lm_test em pandas recentes)
    lr01 = lr_test(M0, M1)  # (LR, p, df)
    lr12 = lr_test(M1, M2)

    resultados = {
        "M0_global": M0,
        "M1_intercepto_por_idade": M1,
        "M2_interacoes_por_idade": M2,
        "comparacao_M0_M1": {
            "LR_stat": float(lr01[0]),
            "p_value": float(lr01[1]),
            "df_diff": int(lr01[2]),
        },
        "comparacao_M1_M2": {
            "LR_stat": float(lr12[0]),
            "p_value": float(lr12[1]),
            "df_diff": int(lr12[2]),
        }
    }
    return resultados


def criterios_informacao(modelo):
    """Retorna AIC, BIC e R² de um modelo OLS."""
    return {"AIC": float(modelo.aic), "BIC": float(modelo.bic), "R2": float(modelo.rsquared)}


def validacao_leave_one_age_out(df: pd.DataFrame) -> pd.DataFrame:
    """Validação cruzada por grupos de Idade (leave-one-age-out) para o modelo global nos logs.
       Para cada idade k, treina M0 nas demais e avalia no grupo k (RMSE e MAE na escala original).
    """
    df_log = preparar_logs(df)
    idades = df_log["Idade"].cat.categories.tolist()
    rows = []
    for idade in idades:
        te = df_log[df_log["Idade"] == idade]
        tr = df_log[df_log["Idade"] != idade]
        if len(te) < 3 or len(tr) < 3:
            continue

        # Ajuste no treino (modelo global)
        M0_tr = smf.ols("ln_V ~ ln_DAP + ln_H", data=tr).fit()

        # Smearing com resíduos do treino
        res_tr = tr["ln_V"] - M0_tr.predict(tr)
        smear = float(np.mean(np.exp(res_tr)))

        # Predições no teste (escala original)
        yhat_log = M0_tr.predict(te)
        yhat = smear * np.exp(yhat_log)
        y_true = np.exp(te["ln_V"])

        rmse = float(root_mean_squared_error(y_true, yhat))
        mae = float(np.mean(np.abs(y_true - yhat)))
        rows.append({"Idade_holdout": str(idade), "RMSE_m3": rmse, "MAE_m3": mae, "N_teste": int(len(te))})

    return pd.DataFrame(rows)


def imprimir_resumo_modelo(label: str, resultados: dict):
    print("--------------------------------------------------------------------")
    print(f"\n=== EQUAÇÃO ALOMÉTRICA ({label}) ===")
    print("V = a * DAP^b * H^c")
    print("\n=== Parâmetros (ajuste nos logs) ===")
    print(f"ln(a) = {resultados['params_log']['const_ln_a']:.6f}")
    print(f"b (DAP) = {resultados['params_log']['b_dap']:.6f}")
    print(f"c (H)   = {resultados['params_log']['b_h']:.6f}")
    print("\n=== Parâmetros na escala original ===")
    print(f"a = exp(ln(a)) = {resultados['a']:.9f}")
    print(f"Smearing de Duan = {resultados['smearing']:.6f}")
    print(f"a_efetivo (a * smearing) = {resultados['a_efetivo_com_smearing']:.9f}")
    print(f"Equação final (recomendada):")
    print(f"V = {resultados['a_efetivo_com_smearing']:.9f} * DAP^{resultados['b']:.6f} * H^{resultados['c']:.6f}")
    print("\n=== Métricas (hold-out 20% na escala original) ===")
    print(f"R² (orig)  = {resultados['r2_teste_escala_original']:.3f}")
    print(f"RMSE (m³)  = {resultados['rmse_teste_escala_original']:.6f}")
    print(f"N (obs.)   = {resultados['n_observacoes']}")


# ------------------------------
# Fluxo principal
# ------------------------------
def main():
    arquivo = "dados.xlsx"   # ajuste o nome se necessário
    planilha = "Planilha1"

    caminho = Path(arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.resolve()}")

    df = pd.read_excel(caminho, sheet_name=planilha)

    colunas_esperadas = {"Idade", "DAP", "H", "Volume"}
    if not colunas_esperadas.issubset(df.columns):
        raise ValueError(f"O arquivo deve conter as colunas: {sorted(colunas_esperadas)}. Colunas encontradas: {list(df.columns)}")

    # Ajuste global
    resultados_global, _ = ajustar_modelo(df)
    imprimir_resumo_modelo("todos os dados (GLOBAL)", resultados_global)

    # Ajustes por idade
    for idade in df["Idade"].unique():
        df_idade = df[df["Idade"] == idade]
        resultados_idade, _ = ajustar_modelo(df_idade)
        imprimir_resumo_modelo(f"idade = {idade}", resultados_idade)

    # Comparações formais (ANCOVA / LR manual)
    print("\n####################################################################")
    print("# COMPARAÇÕES FORMAIS (ANCOVA e testes de verossimilhança)        #")
    print("####################################################################")
    df_log = preparar_logs(df)
    comps = ajustar_modelos_ancova(df_log)

    info_M0 = criterios_informacao(comps["M0_global"])
    info_M1 = criterios_informacao(comps["M1_intercepto_por_idade"])
    info_M2 = criterios_informacao(comps["M2_interacoes_por_idade"])

    print("\nAIC/BIC/R² por modelo nos logs:")
    print(f"M0 (global)                 -> AIC={info_M0['AIC']:.2f}  BIC={info_M0['BIC']:.2f}  R²={info_M0['R2']:.3f}")
    print(f"M1 (intercepto por idade)   -> AIC={info_M1['AIC']:.2f}  BIC={info_M1['BIC']:.2f}  R²={info_M1['R2']:.3f}")
    print(f"M2 (inclinações por idade)  -> AIC={info_M2['AIC']:.2f}  BIC={info_M2['BIC']:.2f}  R²={info_M2['R2']:.3f}")

    lr01 = comps["comparacao_M0_M1"]
    lr12 = comps["comparacao_M1_M2"]
    print("\nTeste de verossimilhança (LR):")
    print(f"M0 vs. M1  -> LR={lr01['LR_stat']:.3f}, p={lr01['p_value']:.4f}, df={lr01['df_diff']} (Interceptos por Idade?)")
    print(f"M1 vs. M2  -> LR={lr12['LR_stat']:.3f}, p={lr12['p_value']:.4f}, df={lr12['df_diff']} (Inclinações por Idade?)")

    # Validação leave-one-age-out
    print("\n####################################################################")
    print("# VALIDAÇÃO CRUZADA POR GRUPO DE IDADE (leave-one-age-out)        #")
    print("####################################################################")
    cv_age = validacao_leave_one_age_out(df)
    if len(cv_age):
        print(cv_age.to_string(index=False))
        print("\nMédias (sobre idades retidas):")
        print(f"RMSE médio (m³): {cv_age['RMSE_m3'].mean():.6f} | MAE médio (m³): {cv_age['MAE_m3'].mean():.6f}")
    else:
        print("Não foi possível executar a validação (grupos com poucas observações).")


if __name__ == "__main__":
    main()
