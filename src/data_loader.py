from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data(data_config):
    """Membaca dataset asli dan daftar pembagian data."""

    data_path = PROJECT_ROOT / data_config["path"]
    split_path = PROJECT_ROOT / data_config["split_path"]

    df = pd.read_csv(data_path)
    split_manifest = pd.read_csv(split_path)

    return df, split_manifest


def validate_manifest(df, split_manifest):
    """Memeriksa struktur manifest dan pemisahan kelompok."""

    required_columns = {"row_id", "group_id", "split"}

    # Memeriksa kolom wajib.
    if not required_columns.issubset(split_manifest.columns):
        raise ValueError(
            "Manifest harus memiliki row_id, group_id, dan split."
        )

    # Memeriksa nilai kosong.
    if split_manifest[list(required_columns)].isna().any().any():
        raise ValueError("Ada nilai kosong pada manifest.")

    # Memeriksa tipe row_id.
    row_ids = split_manifest["row_id"]

    if not pd.api.types.is_integer_dtype(row_ids):
        raise ValueError("row_id harus berupa bilangan bulat.")

    # Memastikan setiap posisi baris tercatat tepat satu kali.
    if sorted(row_ids.tolist()) != list(range(len(df))):
        raise ValueError(
            "row_id tidak lengkap, berulang, atau di luar dataset."
        )

    # Memeriksa nama bagian dataset.
    if set(split_manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError(
            "Label split harus train, validation, dan test."
        )

    # Memastikan satu kelompok hanya berada pada satu bagian.
    splits_per_group = (
        split_manifest.groupby("group_id")["split"].nunique()
    )

    if splits_per_group.gt(1).any():
        raise ValueError("Ada kelompok yang tersebar antarbagian.")

    return True

def get_split(
    df,
    split_manifest,
    split_name,
    target_column="Revenue",
    exclude_columns=None,
):
    """Mengambil fitur dan target untuk satu bagian dataset."""

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name harus train, validation, atau test.")

    # Mengambil posisi baris yang termasuk bagian pilihan.
    row_ids = split_manifest.loc[
        split_manifest["split"] == split_name,
        "row_id",
    ]

    # Mengambil baris tersebut dari dataset asli.
    split_df = df.iloc[row_ids].copy()

    # Memisahkan fitur dan target.
    drop_columns = [target_column] + (exclude_columns or [])

    X = split_df.drop(columns=drop_columns)
    y = split_df[target_column].astype(int)

    return X, y