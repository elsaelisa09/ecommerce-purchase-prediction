from sklearn.linear_model import LogisticRegression


def create_model(params, seed):
    """Membentuk model berdasarkan parameter eksperimen."""

    model_params = params.copy()

    # Menggunakan seed jika random_state belum ditentukan.
    model_params.setdefault("random_state", seed)

    return LogisticRegression(**model_params)