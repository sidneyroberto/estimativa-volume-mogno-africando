#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Boosting para prever Volume de árvores (mogno africano)
usando DAP, Altura (H) e Idade extraída de strings como "36 meses".

O script:
1) Lê "dados.xlsx", planilha "Planilha1"
2) Converte a coluna "Idade" ("36 meses") -> 36 (inteiro)
3) Treina GradientBoostingRegressor COM e SEM Idade
4) Exibe R², RMSE (m³) e importâncias das variáveis (hold-out 80/20)
5) Faz validação leave-one-age-out (GroupKFold por Idade_num)

Requisitos: pandas, numpy, scikit-learn, openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

ARQUIVO = "dados.xlsx"
PLANILHA = "Planilha1"
RANDOM_STATE = 42


def carregar_dados(caminho: str, planilha: str) -> pd.DataFrame:
    p = Path(caminho)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p.resolve()}")
    df = pd.read_excel(p, sheet_name=planilha)
    colunas_necessarias = {"Idade", "DAP", "H", "Volume"}
    if not colunas_necessarias.issubset(df.columns):
        raise ValueError(
            f"O arquivo deve conter as colunas: {sorted(colunas_necessarias)}. "
            f"Colunas encontradas: {list(df.columns)}"
        )
    return df


def extrair_idade_em_meses(serie: pd.Series) -> pd.Series:
    """Extrai número de 'XX meses' -> XX (float). Mantém NaN se falhar."""
    idade_num = serie.astype(str).str.extract(r'(\d+)')[0].astype(float)
    return idade_num


def filtrar_valores_validos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["DAP"] > 0) & (df["H"] > 0) & (df["Volume"] > 0)]
    df = df[df["Idade_num"].notna()]
    return df.reset_index(drop=True)


def treino_teste_gb(X: pd.DataFrame, y: pd.Series, descricao: str = "COM IDADE"):
    """Hold-out 80/20 com GB; imprime R², RMSE e importâncias."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    gb.fit(X_tr, y_tr)
    y_pred = gb.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    rmse = root_mean_squared_error(y_te, y_pred)

    print("\n" + "-" * 72)
    print(f"GRADIENT BOOSTING (hold-out 20%) [{descricao}]")
    print("-" * 72)
    print(f"R²: {r2:.6f}")
    print(f"RMSE (m³): {rmse:.6f}")
    print("Importâncias das variáveis:")
    for col, imp in zip(X.columns, gb.feature_importances_):
        print(f"  - {col}: {imp:.6f}")
    return gb, (r2, rmse)


def leave_one_age_out_cv(df: pd.DataFrame, usar_idade: bool = True) -> pd.DataFrame:
    """Validação cruzada por grupo (uma idade por vez fora) usando GroupKFold."""
    features = ["DAP", "H"] + (["Idade_num"] if usar_idade else [])
    X = df[features].copy()
    y = df["Volume"].copy()
    groups = df["Idade_num"].astype(int).values

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    resultados = []
    for idx_tr, idx_te in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
        y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]
        idade_teste = int(df.iloc[idx_te]["Idade_num"].iloc[0])

        gb = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
        gb.fit(X_tr, y_tr)
        y_pred = gb.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        rmse = root_mean_squared_error(y_te, y_pred)
        resultados.append({
            "Idade_holdout_meses": idade_teste,
            "R2": r2,
            "RMSE_m3": rmse,
            "N_teste": int(len(y_te))
        })
    res = pd.DataFrame(resultados).sort_values("Idade_holdout_meses").reset_index(drop=True)
    return res


def main():
    # 1) Carregar e preparar
    df = carregar_dados(ARQUIVO, PLANILHA)
    df["Idade_num"] = extrair_idade_em_meses(df["Idade"])
    df = filtrar_valores_validos(df)

    # 2) Treino/teste COM Idade
    X_with_age = df[["DAP", "H", "Idade_num"]]
    y = df["Volume"]
    _, (r2_age, rmse_age) = treino_teste_gb(
        X_with_age, y, "COM Idade (DAP, H, Idade_num)"
    )

    # 3) Treino/teste SEM Idade (apenas DAP e H) para comparação
    X_no_age = df[["DAP", "H"]]
    _, (r2_no_age, rmse_no_age) = treino_teste_gb(
        X_no_age, y, "SEM Idade (DAP, H)"
    )

    # 4) Validação leave-one-age-out
    print("\n" + "#" * 72)
    print("# VALIDAÇÃO LEAVE-ONE-AGE-OUT (GroupKFold por Idade)            #")
    print("#" * 72)
    loao_com = leave_one_age_out_cv(df, usar_idade=True)
    loao_sem = leave_one_age_out_cv(df, usar_idade=False)

    # 5) Resumo
    print("\nResumo hold-out 20%:")
    print(f"  COM Idade  -> R²={r2_age:.3f} | RMSE={rmse_age:.6f} m³")
    print(f"  SEM Idade  -> R²={r2_no_age:.3f} | RMSE={rmse_no_age:.6f} m³")

    print("\nLeave-one-age-out (COM Idade):")
    print(loao_com.to_string(index=False))
    print(f" Médias -> R² médio={loao_com['R2'].mean():.3f} | "
          f"RMSE médio={loao_com['RMSE_m3'].mean():.6f} m³")

    print("\nLeave-one-age-out (SEM Idade):")
    print(loao_sem.to_string(index=False))
    print(f" Médias -> R² médio={loao_sem['R2'].mean():.3f} | "
          f"RMSE médio={loao_sem['RMSE_m3'].mean():.6f} m³")


if __name__ == "__main__":
    main()
