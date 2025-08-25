import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# Carregar dados
df = pd.read_excel("dados.xlsx")

idades_unicas = df["Idade"].unique()

for idade in idades_unicas:
    df_idade = df[df["Idade"] == idade]

    # Features e alvo
    X = df_idade[["DAP", "H"]]
    y = df_idade["Volume"]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)

    print("--------------------------------------------------------------------")
    print("Ajuste com Random Forest para a idade de %s" % idade)
    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))

    # Exemplo de previsão
    novo_dado = pd.DataFrame({"DAP": [10.0], "H": [6.0]})
    print("Volume estimado para DAP = 10 e H = 6: %.6f" % model.predict(novo_dado)[0])
