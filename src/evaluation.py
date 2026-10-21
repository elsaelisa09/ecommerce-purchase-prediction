import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


def evaluate_model(pipeline, X, y, threshold=0.5):
    """Menghitung metrik pembelian pada data evaluasi."""

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold harus berada antara 0 dan 1.")

    # Mengambil skor probabilitas untuk kelas membeli.
    probabilities = pipeline.predict_proba(X)[:, 1]

    # Mengubah skor menjadi prediksi kelas berdasarkan threshold.
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "average_precision": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
    }

    return metrics, predictions, probabilities
def create_confusion_matrix(y_true, predictions):
    """Membuat grafik perbandingan label asli dan prediksi."""

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["Tidak Membeli", "Membeli"],
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax,
    )

    ax.set_title("Confusion Matrix — Validation")
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Kondisi Sebenarnya")

    fig.tight_layout()

    return fig