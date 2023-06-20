import numpy as np
import pandas as pd

df = pd.read_csv("Database/Principale_NonAggiornato.csv")
sort = df.sort_values(by=["Country"], ascending=True)

sort.to_csv("Database/Principale_Organizzato_per_Nazione.csv")

sort.isna();
nan_rows = df[df.isna().any(axis=1)]

nan_rows.to_csv("Database/Principale_Estratti_Null.csv")
print(nan_rows)

nan_rows_cityonly = nan_rows["City"]
nan_rows_cityonly.to_csv("Database/Null-Solo-Città.csv", index= False)

df_solo_città = pd.read_csv("Database/Null-Solo-Città.csv")
df_vuoti = pd.read_csv("Database/ris.csv", names=["Country"])

print(df_vuoti)
pd.concat([df_vuoti,df_solo_città], axis=1).to_csv("Database/joined_non_riempito.csv", index=False)

df_riempito = pd.read_csv("Database/joined.csv")
df_no_paesi = pd.read_csv("Database/Principale_Estratti_Null.csv")
pd.merge(df_riempito,df_no_paesi, on="City").to_csv("Database/joined_completo.csv")



df_finale = pd.read_csv("Database/joined_completo.csv")
df_finale = df_finale.drop(["Country_y", "Unnamed: 0.1"], axis=1)
cols = list(df_finale)
cols.insert(0, cols.pop(cols.index('Unnamed: 0')))
df_finale = df_finale.loc[:, cols]
df_finale = df_finale.rename(columns={"Unnamed: 0": "", "Country_x": "Country"})


df_finale.drop_duplicates().to_csv("Database/joined_completo.csv", index=False)
df_finale = pd.read_csv("Database/joined_completo.csv")
sort = pd.read_csv("Database/Principale_Organizzato_per_Nazione.csv")
sort["Country"].replace('', np.nan, inplace=True)
print(sort)
sort.dropna(subset=["Country"], inplace=True)
sort.to_csv("Database/Principale_Organizzato_per_Nazione_Copia_SenzaNULLI.csv", index=False)
sort = pd.read_csv("Database/Principale_Organizzato_per_Nazione_SenzaNULLI.csv")
df_completo = pd.concat([sort, df_finale])
df_completo = df_completo.rename(columns={"AQI Value": "AQI_Value", "AQI Category": "AQI_Category",
                                          "CO AQI Value": "CO_AQI_Value","CO AQI Category": "CO_AQI_Category",
                                          "Ozone AQI Value": "Ozone_AQI_Value","Ozone AQI Category": "Ozone_AQI_Category",
                                          "NO2 AQI Value": "NO2_AQI_Value","NO2 AQI Category": "NO2_AQI_Category",
                                          "PM2.5 AQI Value": "PM2_5_AQI_Value","PM2.5 AQI Category": "PM2_5_AQI_Category",
                                          "lat":"Lat", "lng": "Lng"
                                          })
df_completo.drop(columns=["Unnamed: 0"]).to_csv("Database/Dataset_Completato.csv", index=False)
#pd.merge(sort,df_finale, left_index = True, right_on="").to_csv("Database/Dataset_Completato.csv")


''' 
Joined.csv é stato riempito a mano dai ~15 valori mancanti 

'''
