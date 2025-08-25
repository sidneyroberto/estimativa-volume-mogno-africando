#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste de equação alométrica de volume:
    V = a * DAP^b * H^c
com correção de viés de retransfomação (smearing de Duan).

Este script foi adaptado para abrir diretamente o arquivo
"Dados cubagem Sidney.xlsx" (planilha "Planilha1") sem precisar
passar argumentos na linha de comando.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from pathlib import Path


def ajustar_modelo(df: pd.DataFrame):
    """Ajusta o modelo ln(V) = ln(a) + b*ln(DAP) + c*ln(H) com OLS.
    Retorna dicionário com parâmetros e objetos auxiliares.
    """
    # Filtro (valores positivos)
    df = df.copy()
    df = df[(df["DAP"] > 0) & (df["H"] > 0) & (df["Volume"] > 0)].reset_index(drop=True)

    # Logs
    df["ln_V"] = np.log(df["Volume"])
    df["ln_DAP"] = np.log(df["DAP"])
    df["ln_H"] = np.log(df["H"])

    X = df[["ln_DAP", "ln_H"]]
    y = df["ln_V"]

    # Avaliação (hold-out) para métricas na escala original
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    ols_tr = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()

    # Predição no conjunto de teste (escala log)
    yhat_log_te = ols_tr.predict(sm.add_constant(X_te))

    # Smearing (estimado nos resíduos de treino)
    residuos_tr = y_tr - ols_tr.predict(sm.add_constant(X_tr))
    smearing = float(np.mean(np.exp(residuos_tr)))

    # Métricas no conjunto de teste, na escala original
    y_te_orig = np.exp(y_te)
    yhat_te_orig = smearing * np.exp(yhat_log_te)
    r2_te = r2_score(y_te_orig, yhat_te_orig)
    rmse_te = root_mean_squared_error(y_te_orig, yhat_te_orig)

    # Ajuste final em TODOS os dados para reportar a equação final
    ols_full = sm.OLS(y, sm.add_constant(X)).fit()
    b0 = float(ols_full.params["const"])
    b_dap = float(ols_full.params["ln_DAP"])
    b_h = float(ols_full.params["ln_H"])

    # Coeficiente a efetivo incorporando smearing: a_eff = smearing * exp(b0)
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


def main():
    arquivo = "dados.xlsx"
    planilha = "Planilha1"

    caminho = Path(arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.resolve()}")

    df = pd.read_excel(caminho, sheet_name=planilha)

    # Normalizar nomes de colunas esperadas
    colunas_esperadas = {"Idade", "DAP", "H", "Volume"}
    if not colunas_esperadas.issubset(df.columns):
        raise ValueError(f"O arquivo deve conter as colunas: {sorted(colunas_esperadas)}. Colunas encontradas: {list(df.columns)}")


    # -----------------------------------------------------------------------
    # Ajustar
    resultados, _ = ajustar_modelo(df)

    print("--------------------------------------------------------------------")

    # Exibir
    print("\n=== EQUAÇÃO ALOMÉTRICA (forma geral) para todos os dados ===")
    print("V = a * DAP^b * H^c")
    print("\n=== Parâmetros (ajuste nos logs) ===")
    print(f"ln(a) = {resultados['params_log']['const_ln_a']:.6f}")
    print(f"b (DAP) = {resultados['params_log']['b_dap']:.6f}")
    print(f"c (H)   = {resultados['params_log']['b_h']:.6f}")

    print("\n=== Parâmetros na escala original ===")
    print(f"a = exp(ln(a)) = {resultados['a']:.9f}")
    print(f"Smearing de Duan = {resultados['smearing']:.6f}")
    print(f"a_efetivo (a * smearing) = {resultados['a_efetivo_com_smearing']:.9f}")
    print(f"Equação final recomendada (com smearing):")
    print(f"V = {resultados['a_efetivo_com_smearing']:.9f} * DAP^{resultados['b']:.6f} * H^{resultados['c']:.6f}")

    print("\n=== Métricas (conjunto de teste na escala original) ===")
    print(f"R² (orig)  = {resultados['r2_teste_escala_original']:.3f}")
    print(f"RMSE (m³)  = {resultados['rmse_teste_escala_original']:.6f}")
    print(f"N (obs.)   = {resultados['n_observacoes']}")

    # Exemplo de previsão
    exemplo_dap = 10.0
    exemplo_h = 6.0
    v_prev = prever_volume(exemplo_dap, exemplo_h,
                        resultados["a_efetivo_com_smearing"],
                        resultados["b"], resultados["c"])
    print("\n=== Exemplo de previsão ===")
    print(f"Para DAP={exemplo_dap} cm e H={exemplo_h} m => V ≈ {v_prev:.6f} m³")
    # -----------------------------------------------------------------------

    idades_unicas = df["Idade"].unique()

    for idade in idades_unicas:
        df_idade = df[df["Idade"] == idade]

        # Ajustar
        resultados, _ = ajustar_modelo(df_idade)

        print("--------------------------------------------------------------------")

        # Exibir
        print("\n=== EQUAÇÃO ALOMÉTRICA (forma geral) para idade de %s ===" % idade)
        print("V = a * DAP^b * H^c")
        print("\n=== Parâmetros (ajuste nos logs) ===")
        print(f"ln(a) = {resultados['params_log']['const_ln_a']:.6f}")
        print(f"b (DAP) = {resultados['params_log']['b_dap']:.6f}")
        print(f"c (H)   = {resultados['params_log']['b_h']:.6f}")

        print("\n=== Parâmetros na escala original ===")
        print(f"a = exp(ln(a)) = {resultados['a']:.9f}")
        print(f"Smearing de Duan = {resultados['smearing']:.6f}")
        print(f"a_efetivo (a * smearing) = {resultados['a_efetivo_com_smearing']:.9f}")
        print(f"Equação final recomendada (com smearing):")
        print(f"V = {resultados['a_efetivo_com_smearing']:.9f} * DAP^{resultados['b']:.6f} * H^{resultados['c']:.6f}")

        print("\n=== Métricas (conjunto de teste na escala original) ===")
        print(f"R² (orig)  = {resultados['r2_teste_escala_original']:.3f}")
        print(f"RMSE (m³)  = {resultados['rmse_teste_escala_original']:.6f}")
        print(f"N (obs.)   = {resultados['n_observacoes']}")

        # Exemplo de previsão
        exemplo_dap = 10.0
        exemplo_h = 6.0
        v_prev = prever_volume(exemplo_dap, exemplo_h,
                            resultados["a_efetivo_com_smearing"],
                            resultados["b"], resultados["c"])
        print("\n=== Exemplo de previsão ===")
        print(f"Para DAP={exemplo_dap} cm e H={exemplo_h} m => V ≈ {v_prev:.6f} m³")

if __name__ == "__main__":
    main()
