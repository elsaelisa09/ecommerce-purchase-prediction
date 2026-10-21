from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from src.models import build_model
from copy import deepcopy


CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def build_preprocessor(X_train, model_name):
    """Menyiapkan pengolahan fitur numerik dan kategorikal."""

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X_train.columns
    ]

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    # Logistic Regression menggunakan scaling fitur numerik.
    if model_name == "logistic_regression":
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

def train_model(X_train, y_train, config):
    """Melatih pipeline sesuai model dan konfigurasi resampling."""

    config = resolve_training_config(y_train, config)
    model_name = config["model"]["name"]

    resampling_name = config.get(
        "resampling", {}
    ).get("name", "none")

    if model_name == "catboost":
        if resampling_name != "none":
            raise ValueError(
                "Pipeline CatBoost ini menggunakan resampling='none'."
            )

        pipeline = build_catboost_pipeline(X_train, config)

    elif resampling_name == "smotenc":
        pipeline = build_smotenc_pipeline(X_train, config)

    elif resampling_name == "none":
        preprocessor = build_preprocessor(
            X_train,
            model_name,
        )

        model = build_model(config["model"], config["seed"])

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    else:
        raise ValueError(
            f"Resampling belum tersedia: {resampling_name}"
        )

    pipeline.fit(X_train, y_train)

    return pipeline

def build_smotenc_pipeline(X_train, config):
    """Membuat pipeline preprocessing, SMOTENC, dan model."""

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X_train.columns
    ]

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    # Standardisasi numerik untuk perhitungan tetangga SMOTENC.
    # Kategori tetap dipertahankan sebagai kategori.
    sampling_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", "passthrough", categorical_columns),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    sampler_params = config["resampling"].get("params", {}).copy()
    sampler_params.setdefault("random_state", config["seed"])

    sampler = SMOTENC(
        categorical_features=categorical_columns,
        **sampler_params,
    )

    # Setelah resampling, kategori diubah menjadi fitur one-hot.
    model_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    model = build_model(config["model"], config["seed"])

    return ImbalancedPipeline(
        steps=[
            ("sampling_preprocessor", sampling_preprocessor),
            ("sampler", sampler),
            ("preprocessor", model_preprocessor),
            ("model", model),
        ]
    )

def cross_validate_model(
    X,
    y,
    cv_splits,
    config,
    on_fold_end=None,
):
    """Melatih pipeline baru dan mengevaluasinya pada setiap fold."""

    from src.evaluation import evaluate_model

    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv_splits,
        start=1,
    ):
        # Preprocessing dan model dilatih hanya pada training fold.
        pipeline = train_model(
            X.iloc[train_idx],
            y.iloc[train_idx],
            config,
        )

        metrics, _, _ = evaluate_model(
            pipeline,
            X.iloc[valid_idx],
            y.iloc[valid_idx],
            threshold=config["evaluation"]["threshold"],
        )

        result = {
            "fold": fold,
            **metrics,
        }

        if (
            config["model"]["name"] == "xgboost"
            and config["model"]["params"].get(
                "scale_pos_weight"
            ) == "auto"
        ):
            result["scale_pos_weight"] = float(
                pipeline.named_steps["model"].get_params()[
                    "scale_pos_weight"
                ]
            )
        fold_results.append(result)

        print(
            f"Fold {fold}/{len(cv_splits)} "
            f"| F1: {metrics['f1']:.4f}"
        )

        # Menjalankan pencatatan tambahan jika diberikan.
        if on_fold_end is not None:
            on_fold_end(result)

    return fold_results

def stringify_categories(X):
    """Menyeragamkan nilai kategori menjadi teks."""
    return X.astype(str)


def build_catboost_pipeline(X_train, config):
    """Membuat pipeline dengan kategori langsung untuk CatBoost."""

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X_train.columns
    ]

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                FunctionTransformer(
                    stringify_categories,
                    feature_names_out="one-to-one",
                ),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    model = build_model(config["model"], config["seed"])
    model.set_params(cat_features=categorical_columns)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

def resolve_training_config(y_train, config):
    """Menghitung bobot kelas hanya dari target training."""

    resolved_config = deepcopy(config)

    model_name = resolved_config["model"]["name"]
    params = resolved_config["model"].get("params", {})

    if (
        model_name == "xgboost"
        and params.get("scale_pos_weight") == "auto"
    ):
        resampling_name = resolved_config.get(
            "resampling", {}
        ).get("name", "none")

        if resampling_name != "none":
            raise ValueError(
                "Eksperimen pembobotan ini menggunakan resampling='none'."
            )

        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())

        if negative_count == 0 or positive_count == 0:
            raise ValueError(
                "Training harus memiliki kedua kelas target."
            )

        params["scale_pos_weight"] = (
            negative_count / positive_count
        )

    return resolved_config