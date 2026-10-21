from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

SELECTION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "weighted_candidate_selection.yaml"
)


st.set_page_config(
    page_title="Prediksi Pembelian",
    page_icon="🛒",
    layout="wide",
)


# Tampilan aplikasi.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1160px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1 {
            letter-spacing: -0.035em;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.7rem;
        }

        button[kind="primary"] {
            background-color: #2563eb;
            border-color: #2563eb;
            color: white;
        }

        button[kind="primary"]:hover {
            background-color: #1d4ed8;
            border-color: #1d4ed8;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline(path, file_version):
    """Memuat ulang pipeline jika file model berubah."""
    return joblib.load(path)


def reset_inputs(defaults):
    """Mengembalikan input ke nilai awal."""
    for key, value in defaults.items():
        st.session_state[key] = value

    st.session_state.pop("purchase_result", None)


def select_default(options, preferred):
    """Memilih nilai awal yang dikenal encoder."""
    return preferred if preferred in options else options[0]


# ============================================================
# 1. MEMUAT MODEL
# ============================================================

try:
    with SELECTION_PATH.open("r", encoding="utf-8") as file:
        selection = yaml.safe_load(file)

    source_run_id = selection["source_run_path"].rsplit("/", 1)[-1]
    threshold = float(selection["threshold"])

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold harus berada di antara 0 dan 1."
        )

    pipeline_path = (
        PROJECT_ROOT
        / "outputs"
        / source_run_id
        / "pipeline.joblib"
    )

    file_version = pipeline_path.stat().st_mtime_ns

    pipeline = load_pipeline(
        str(pipeline_path),
        file_version,
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    category_options = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "categorical":
            category_options = {
                column: choices.tolist()
                for column, choices in zip(
                    columns,
                    transformer.categories_,
                )
            }

    required_categories = {
        "Month",
        "VisitorType",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "Weekend",
    }

    missing_categories = (
        required_categories - set(category_options)
    )

    if missing_categories:
        raise ValueError(
            "Kategori belum ditemukan pada pipeline: "
            f"{sorted(missing_categories)}"
        )

    positive_class_index = list(pipeline.classes_).index(1)

except Exception as error:
    st.title("Prediksi pembelian")

    st.error(
        "Model belum dapat dimuat. "
        "Periksa file konfigurasi dan pipeline."
    )

    with st.expander("Detail untuk pengelola aplikasi"):
        st.code(str(error), language="text")

    st.stop()


# ============================================================
# 2. LABEL DAN NILAI AWAL
# ============================================================

MONTH_LABELS = {
    "Jan": "Januari",
    "Feb": "Februari",
    "Mar": "Maret",
    "Apr": "April",
    "May": "Mei",
    "June": "Juni",
    "Jul": "Juli",
    "Aug": "Agustus",
    "Sep": "September",
    "Oct": "Oktober",
    "Nov": "November",
    "Dec": "Desember",
}

MONTH_ORDER = {
    month: index
    for index, month in enumerate(MONTH_LABELS)
}

VISITOR_LABELS = {
    "Returning_Visitor": "Pengunjung lama",
    "New_Visitor": "Pengunjung baru",
    "Other": "Lainnya",
}

month_options = sorted(
    category_options["Month"],
    key=lambda month: MONTH_ORDER.get(month, 99),
)

defaults = {
    "session_product_pages": 18,
    "session_product_minutes": 10.0,
    "session_bounce_pct": 2.0,
    "session_exit_pct": 3.0,
    "session_page_values": 0.0,
    "session_month": select_default(
        month_options,
        "May",
    ),
    "session_visitor": select_default(
        category_options["VisitorType"],
        "Returning_Visitor",
    ),
    "session_weekend": False,
    "session_admin_pages": 0,
    "session_admin_minutes": 0.0,
    "session_info_pages": 0,
    "session_info_minutes": 0.0,
    "session_special_day": 0.0,
    "session_os": select_default(
        category_options["OperatingSystems"],
        2,
    ),
    "session_browser": select_default(
        category_options["Browser"],
        2,
    ),
    "session_region": select_default(
        category_options["Region"],
        1,
    ),
    "session_traffic": select_default(
        category_options["TrafficType"],
        2,
    ),
}

# Menghapus hasil lama jika model atau threshold berubah.
model_signature = (
    source_run_id,
    file_version,
    threshold,
)

if (
    st.session_state.get("purchase_model_signature")
    != model_signature
):
    reset_inputs(defaults)

    st.session_state["purchase_model_signature"] = (
        model_signature
    )

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 3. FORM INPUT
# ============================================================

st.title("Prediksi pembelian")

st.caption(
    "Atur perilaku kunjungan untuk melihat prediksi model."
)

form_column, result_column = st.columns(
    [1.7, 1],
    gap="large",
)

with form_column:
    with st.form("session_form", border=True):
        st.subheader("Data sesi")

        left, right = st.columns(2)

        with left:
            product_pages = st.number_input(
                "Halaman produk dilihat",
                min_value=0,
                step=1,
                key="session_product_pages",
            )

        with right:
            product_minutes = st.number_input(
                "Durasi lihat produk (menit)",
                min_value=0.0,
                step=0.5,
                key="session_product_minutes",
                help="Total durasi pada halaman produk.",
            )

        left, right = st.columns(2)

        with left:
            bounce_pct = st.number_input(
                "Bounce rate (%)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                key="session_bounce_pct",
                help=(
                    "Masukkan dalam persen. "
                    "Rentang dataset adalah 0–20%."
                ),
            )

        with right:
            exit_pct = st.number_input(
                "Exit rate (%)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                key="session_exit_pct",
                help=(
                    "Masukkan dalam persen. "
                    "Rentang dataset adalah 0–20%."
                ),
            )

        page_values = st.number_input(
            "Nilai halaman (PageValues)",
            min_value=0.0,
            step=1.0,
            key="session_page_values",
            help=(
                "Gunakan nilai PageValues pada data sesi. "
                "Nilai 0 berarti metrik tercatat nol, "
                "bukan belum diketahui."
            ),
        )

        left, right = st.columns(2)

        with left:
            month = st.selectbox(
                "Bulan kunjungan",
                options=month_options,
                format_func=lambda value: MONTH_LABELS.get(
                    value,
                    value,
                ),
                key="session_month",
            )

        with right:
            visitor_type = st.selectbox(
                "Tipe pengunjung",
                options=category_options["VisitorType"],
                format_func=lambda value: VISITOR_LABELS.get(
                    value,
                    value,
                ),
                key="session_visitor",
            )

        weekend = st.checkbox(
            "Kunjungan saat akhir pekan",
            key="session_weekend",
        )

        with st.expander(
            "Pengaturan tambahan",
            expanded=False,
        ):
            st.caption(
                "Sesuaikan bila sesi juga membuka "
                "halaman selain produk."
            )

            left, right = st.columns(2)

            with left:
                admin_pages = st.number_input(
                    "Halaman administratif",
                    min_value=0,
                    step=1,
                    key="session_admin_pages",
                )

                info_pages = st.number_input(
                    "Halaman informasi",
                    min_value=0,
                    step=1,
                    key="session_info_pages",
                )

            with right:
                admin_minutes = st.number_input(
                    "Durasi administratif (menit)",
                    min_value=0.0,
                    step=0.5,
                    key="session_admin_minutes",
                )

                info_minutes = st.number_input(
                    "Durasi informasi (menit)",
                    min_value=0.0,
                    step=0.5,
                    key="session_info_minutes",
                )

            special_day = st.slider(
                "Kedekatan hari spesial",
                min_value=0.0,
                max_value=1.0,
                step=0.2,
                key="session_special_day",
                help=(
                    "0 berarti tidak dekat. "
                    "Nilai lebih besar berarti semakin dekat."
                ),
            )

            st.caption(
                "Kode berikut mengikuti dataset. "
                "Pemetaan nama perangkat dan wilayah "
                "tidak tersedia."
            )

            left, right = st.columns(2)

            with left:
                operating_system = st.selectbox(
                    "Sistem operasi (kode)",
                    options=category_options["OperatingSystems"],
                    key="session_os",
                )

                region = st.selectbox(
                    "Wilayah (kode)",
                    options=category_options["Region"],
                    key="session_region",
                )

            with right:
                browser = st.selectbox(
                    "Browser (kode)",
                    options=category_options["Browser"],
                    key="session_browser",
                )

                traffic_type = st.selectbox(
                    "Sumber trafik (kode)",
                    options=category_options["TrafficType"],
                    key="session_traffic",
                )

        predict_column, reset_column = st.columns([2, 1])

        with predict_column:
            submitted = st.form_submit_button(
                "Prediksi pembelian",
                type="primary",
                width="stretch",
            )

        with reset_column:
            st.form_submit_button(
                "Atur ulang",
                width="stretch",
                on_click=reset_inputs,
                args=(defaults,),
            )

    # ========================================================
    # 4. VALIDASI DAN PREDIKSI
    # ========================================================

    if submitted:
        input_errors = []

        for pages, minutes, label in [
            (product_pages, product_minutes, "produk"),
            (admin_pages, admin_minutes, "administratif"),
            (info_pages, info_minutes, "informasi"),
        ]:
            if pages == 0 and minutes > 0:
                input_errors.append(
                    f"Durasi halaman {label} harus 0 "
                    "jika jumlah halamannya 0."
                )

        if input_errors:
            st.session_state.pop("purchase_result", None)

            for message in input_errors:
                st.error(message)

        else:
            # Konversi menit menjadi detik.
            # Konversi persen menjadi rasio.
            input_data = {
                "Administrative": admin_pages,
                "Administrative_Duration": admin_minutes * 60.0,
                "Informational": info_pages,
                "Informational_Duration": info_minutes * 60.0,
                "ProductRelated": product_pages,
                "ProductRelated_Duration": product_minutes * 60.0,
                "BounceRates": bounce_pct / 100.0,
                "ExitRates": exit_pct / 100.0,
                "PageValues": page_values,
                "SpecialDay": special_day,
                "Month": month,
                "OperatingSystems": operating_system,
                "Browser": browser,
                "Region": region,
                "TrafficType": traffic_type,
                "VisitorType": visitor_type,
                "Weekend": weekend,
            }

            try:
                input_df = pd.DataFrame([input_data])

                input_df = input_df.loc[
                    :,
                    pipeline.feature_names_in_,
                ]

                score = float(
                    pipeline.predict_proba(input_df)[
                        0,
                        positive_class_index,
                    ]
                )

                st.session_state["purchase_result"] = {
                    "score": score,
                    "prediction": int(score >= threshold),
                    "input_df": input_df,
                }

            except Exception as error:
                st.session_state.pop("purchase_result", None)

                st.error(
                    "Prediksi belum dapat dihitung. "
                    "Periksa kembali input sesi."
                )

                with st.expander(
                    "Detail untuk pengelola aplikasi"
                ):
                    st.code(str(error), language="text")


# ============================================================
# 5. HASIL SIMULASI
# ============================================================

with result_column:
    with st.container(border=True):
        st.subheader("Hasil simulasi")

        result = st.session_state.get("purchase_result")

        if result is None:
            st.info(
                "Isi data sesi, lalu pilih "
                "Prediksi pembelian."
            )

            st.caption(
                "Hasil prediksi akan muncul di sini."
            )

        else:
            if result["prediction"] == 1:
                st.success("Prediksi: Membeli")
            else:
                st.info("Prediksi: Tidak membeli")

            st.metric(
                "Skor kecenderungan membeli",
                f"{result['score']:.1%}",
            )

            st.progress(result["score"])

            st.caption(
                f"Ambang keputusan: {threshold:.0%}. "
                "Skor yang mencapai ambang "
                "diklasifikasikan sebagai membeli."
            )

            st.caption(
                "Berdasarkan data terakhir yang dikirim. "
                "Skor model bukan jaminan "
                "terjadinya transaksi."
            )

            with st.expander("Lihat input terakhir"):
                input_display = (
                    result["input_df"]
                    .T
                    .rename(columns={0: "Nilai"})
                    .astype(str)
                )

                st.dataframe(
                    input_display,
                    width="stretch",
                )

    with st.expander("Tentang model"):
        st.write("**Model:** XGBoost")

        st.write(
            f"**Ambang keputusan:** {threshold:.0%}"
        )

        st.write(
            f"**Jumlah fitur:** {int(pipeline.n_features_in_)}"
        )

        st.caption(
            "Model menggunakan pembobotan kelas "
            "untuk data tidak seimbang."
        )

        st.caption(
            "Demo memakai atribut sesi, termasuk PageValues. "
            "Penggunaan fitur sebelum transaksi "
            "belum divalidasi."
        )


