import pandas as pd

df = pd.read_csv("Database/Principale_NonAggiornato.csv")
sort = df.sort_values(by=["Country"], ascending=True)

sort.to_csv("Database/Principale_Organizzato_per_Nazione.csv")

# df_mancante = df["Country"]
# df_mancante = df[df.Country.notnull()]
nan_rows = df[df.isna().any(axis=1)]

nan_rows.to_csv("Database/Principale_Estratti_Null.csv")
print(nan_rows)

# print(df_mancante)

# print(df_mancante.isna().drop(df[df_mancante =="False"].index(), inplace=True))
