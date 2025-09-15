import pandas as pd, sys

df = pd.read_csv("data/interim/limpio_sent.csv")  # viene del paso 6.1 previo
gold = pd.read_csv("data/interim/para_anotar_gold.csv")

if "sentimiento_gold" not in gold.columns:
    print("Falta 'sentimiento_gold' en el CSV de oro.")
    sys.exit(1)

m = gold[["id", "sentimiento_gold"]].dropna()
ok = {"pos", "neg", "neu"}
bad = m[~m["sentimiento_gold"].isin(ok)]
if len(bad):
    print("Hay etiquetas fuera de {pos,neg,neu}:")
    print(bad["sentimiento_gold"].value_counts())
    sys.exit(1)

df = df.merge(m, on="id", how="left")
df["sentimiento"] = df["sentimiento_gold"].fillna(df["sentimiento"])
df.drop(columns=["sentimiento_gold"], inplace=True)
df.to_csv("data/interim/limpio_final.csv", index=False)

# Resumen útil
print("Etiquetas de sentimiento listas -> data/interim/limpio_final.csv")
print(df["sentimiento"].value_counts(dropna=False))
if "dieta_heuristica" in df.columns:
    print("\nPor dieta:")
    print(
        df.groupby("dieta_heuristica")["sentimiento"]
        .value_counts()
        .unstack(fill_value=0)
    )
