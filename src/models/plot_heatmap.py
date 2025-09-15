# src/models/plot_heatmap.py
import os
import matplotlib

matplotlib.use("Agg")  # asegura backend no interactivo (guarda a PNG)
import matplotlib.pyplot as plt
import pandas as pd
from src.common.paths import pjoin


def run():
    # CSV principal (scores) generado por absa_extract
    scores_path = pjoin("reports", "matriz_dieta_aspecto.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(
            f"No existe {scores_path}. Primero corre: python -m src.models.absa_extract"
        )

    mat = pd.read_csv(scores_path, index_col=0)

    plt.figure(figsize=(7, 4))
    plt.imshow(mat.values, aspect="auto")
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=0)
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.colorbar(label="Intensidad (pos − neg)")
    plt.title("Matriz dieta × aspecto")
    plt.tight_layout()

    out = pjoin("reports", "heatmap_dieta_aspecto.png")
    plt.savefig(out, dpi=200)
    print(f"Figura -> {out}")


if __name__ == "__main__":
    run()
