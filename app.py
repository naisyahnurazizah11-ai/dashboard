from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fiscal Insight Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "dataset_fiskal_97_hipoFull.csv"
ACCENT = "#00D4AA"
PLOT_TEMPLATE = "plotly_dark"

GROUPS = {
    "Transfer & Pendapatan": [
        "DBH_PAJAK", "DBH_SDA", "DBH_Lainnya", "DAU_Grant", "DAU_Earmark",
        "DAK_Fisik", "DAK_Nonfisik", "PAD", "ICT",
    ],
    "Belanja Modal": [
        "bm_pendidikan", "bm_kesehatan", "bm_perumahan", "bm_lingkungan",
        "bm_ekonomi", "bm_infrastruktur", "bm_pariwisata", "bm_perlindungan_sosial",
    ],
    "Indikator Pembangunan": ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"],
}
LABELS = {
    "KEMISKINAN": "Kemiskinan (%)",
    "PENGANGGURAN": "Pengangguran (%)",
    "KETIMPANGAN": "Ketimpangan (Gini)",
    "IPM": "IPM",
    "PE": "Pertumbuhan Ekonomi (%)",
}
# Untuk indikator ini, nilai lebih TINGGI = lebih buruk
LOWER_IS_BETTER = {"KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN"}
OUTCOMES = GROUPS["Indikator Pembangunan"]
ALL_NUM = [c for g in GROUPS.values() for c in g]


def label(col: str) -> str:
    return LABELS.get(col, col)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Kota"] = df["Kota"].astype(str)
    df["Nama"] = df["Kota"].str.replace(r"^(Kota|Kabupaten)\s+", "", regex=True)
    return df


df_all = load_data()


# ----------------------------------------------------------------------------
# Regresi OLS (numpy)
# ----------------------------------------------------------------------------
def ols(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    resid = y - yhat
    n, k = A.shape
    sse = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
    rmse = float(np.sqrt(sse / n))
    if n > k:
        cov = sse / (n - k) * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.diag(cov))
        tstat = beta / se
    else:
        se = tstat = np.full(k, np.nan)
    return dict(beta=beta, yhat=yhat, resid=resid, r2=r2, adj=adj,
                rmse=rmse, se=se, t=tstat, n=n)


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .hero {{
        padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
        background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border: 1px solid rgba(255,255,255,.08);
    }}
    .hero h1 {{ margin: 0; font-size: 2rem; }}
    .hero p  {{ margin: .3rem 0 0; opacity: .8; }}
    [data-testid="stMetric"] {{
        background: #1A1F2B; border: 1px solid rgba(255,255,255,.07);
        padding: .9rem 1rem; border-radius: 14px;
    }}
    .eq {{
        background:#1A1F2B; border-left: 4px solid {ACCENT};
        padding:.8rem 1rem; border-radius:8px; font-family:monospace;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>📊 Fiscal Insight Indonesia</h1>
      <p>Eksplorasi data fiskal 97 kota (2014–2024): transfer daerah, PAD, belanja modal,
      dan dampaknya pada kemiskinan, pengangguran, ketimpangan, IPM &amp; pertumbuhan ekonomi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Filter")
    y_min, y_max = int(df_all["Tahun"].min()), int(df_all["Tahun"].max())
    year_range = st.slider("Rentang tahun", y_min, y_max, (y_min, y_max))

    kota_list = sorted(df_all["Kota"].unique())
    kota_sel = st.multiselect(
        "Kota", kota_list, default=kota_list,
        help="Kosongkan/ubah untuk fokus ke kota tertentu.",
    )
    st.caption(f"{len(kota_sel)} dari {len(kota_list)} kota dipilih")
    st.divider()
    st.caption("Dibuat dengan Streamlit • Plotly • NumPy")

df = df_all[
    df_all["Kota"].isin(kota_sel)
    & df_all["Tahun"].between(year_range[0], year_range[1])
].copy()

if df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah filter di sidebar.")
    st.stop()

tab_ring, tab_tren, tab_kor, tab_reg, tab_data = st.tabs(
    ["🏠 Ringkasan", "📈 Tren", "🔥 Korelasi", "🧮 Regresi", "🗂️ Data"]
)

# ----------------------------------------------------------------------------
# 1. Ringkasan
# ----------------------------------------------------------------------------
with tab_ring:
    first_y, last_y = df["Tahun"].min(), df["Tahun"].max()
    cols = st.columns(5)
    for c, col in zip(cols, OUTCOMES):
        now = df.loc[df["Tahun"] == last_y, col].mean()
        before = df.loc[df["Tahun"] == first_y, col].mean()
        delta = now - before if first_y != last_y else None
        c.metric(
            label(col), f"{now:.2f}",
            None if delta is None else f"{delta:+.2f} vs {first_y}",
            delta_color="inverse" if col in LOWER_IS_BETTER else "normal",
        )
    st.caption(f"Rata-rata seluruh kota terpilih pada tahun {last_y}.")

    st.subheader("🏆 Ranking kota")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ind = c1.selectbox("Indikator", ALL_NUM, index=ALL_NUM.index("IPM"), format_func=label)
    yr = c2.selectbox("Tahun", sorted(df["Tahun"].unique(), reverse=True))
    topn = c3.slider("Jumlah kota", 5, 30, 10)
    asc = c4.toggle("Urut terkecil", value=ind in LOWER_IS_BETTER)

    rank = (
        df[df["Tahun"] == yr][["Nama", ind]]
        .sort_values(ind, ascending=asc).head(topn).iloc[::-1]
    )
    fig = px.bar(
        rank, x=ind, y="Nama", orientation="h", text=ind,
        color=ind, color_continuous_scale="Tealgrn", template=PLOT_TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=max(320, 32 * topn), coloraxis_showscale=False,
                      yaxis_title=None, xaxis_title=label(ind),
                      margin=dict(l=0, r=20, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧩 Komposisi belanja modal")
    bm_cols = GROUPS["Belanja Modal"]
    comp = df[bm_cols].mean().rename(lambda s: s.replace("bm_", "")).reset_index()
    comp.columns = ["Sektor", "Rata-rata"]
    fig = px.pie(comp, names="Sektor", values="Rata-rata", hole=0.55,
                 template=PLOT_TEMPLATE, color_discrete_sequence=px.colors.sequential.Tealgrn)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# 2. Tren
# ----------------------------------------------------------------------------
with tab_tren:
    c1, c2 = st.columns([1, 2])
    ind = c1.selectbox("Variabel", ALL_NUM, index=ALL_NUM.index("IPM"),
                       format_func=label, key="tren_var")
    default_cmp = [k for k in ["Kota Denpasar"] if k in kota_sel] or kota_sel[:1]
    cmp_kota = c2.multiselect("Bandingkan kota", kota_sel, default=default_cmp)

    avg = df.groupby("Tahun", as_index=False)[ind].mean().assign(Kota="Rata-rata kota terpilih")
    plot_df = pd.concat([avg, df[df["Kota"].isin(cmp_kota)][["Tahun", ind, "Kota"]]])
    fig = px.line(plot_df, x="Tahun", y=ind, color="Kota", markers=True,
                  template=PLOT_TEMPLATE)
    for tr in fig.data:
        if tr.name == "Rata-rata kota terpilih":
            tr.line.update(dash="dash", color="#AAAAAA")
    fig.update_layout(height=480, yaxis_title=label(ind),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sebaran per tahun")
    fig = px.box(df, x="Tahun", y=ind, template=PLOT_TEMPLATE,
                 color_discrete_sequence=[ACCENT])
    fig.update_layout(height=380, yaxis_title=label(ind))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# 3. Korelasi
# ----------------------------------------------------------------------------
with tab_kor:
    grp = st.multiselect(
        "Kelompok variabel", list(GROUPS), default=list(GROUPS),
    )
    vars_ = [c for g in grp for c in GROUPS[g]]
    if len(vars_) >= 2:
        corr = df[vars_].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", template=PLOT_TEMPLATE)
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        pairs = (
            corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            .stack().rename("r").reset_index()
        )
        pairs.columns = ["Variabel A", "Variabel B", "r"]
        pairs["|r|"] = pairs["r"].abs()
        top = pairs.sort_values("|r|", ascending=False).head(10).drop(columns="|r|")
        st.subheader("Top 10 pasangan berkorelasi")
        st.dataframe(top.round(3), hide_index=True, use_container_width=True)
    else:
        st.info("Pilih minimal dua variabel.")

    st.subheader("Scatter antar variabel")
    c1, c2, c3 = st.columns(3)
    xv = c1.selectbox("Sumbu X", ALL_NUM, index=ALL_NUM.index("PAD"), format_func=label, key="sx")
    yv = c2.selectbox("Sumbu Y", ALL_NUM, index=ALL_NUM.index("IPM"), format_func=label, key="sy")
    colour = c3.selectbox("Warna", ["Tahun", "Kota"], key="sc")
    fig = px.scatter(df, x=xv, y=yv, color=df[colour].astype(str) if colour == "Kota" else df[colour],
                     hover_name="Kota", opacity=0.75, template=PLOT_TEMPLATE)
    if colour == "Kota":
        fig.update_layout(showlegend=False)
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# 4. Regresi
# ----------------------------------------------------------------------------
with tab_reg:
    st.markdown("Regresi linier **OLS** (dihitung dengan NumPy). Model 2D = 1 variabel bebas, "
                "model 3D = 2 variabel bebas.")

    c1, c2 = st.columns([1, 2])
    scope_default = ["Kota Denpasar"] if "Kota Denpasar" in kota_list else kota_list[:1]
    reg_kota = c2.multiselect(
        "Data yang dipakai (kota)", kota_list, default=scope_default,
        help="Default: Kota Denpasar. Pilih beberapa kota / semua kota untuk sampel lebih besar.",
    )
    mode = c1.radio("Model", ["2D (1 variabel bebas)", "3D (2 variabel bebas)"], horizontal=False)

    d = df_all[df_all["Kota"].isin(reg_kota)
               & df_all["Tahun"].between(year_range[0], year_range[1])]

    y_col = st.selectbox("Variabel terikat (Y)", ALL_NUM,
                         index=ALL_NUM.index("IPM"), format_func=label)
    x_opts = [c for c in ALL_NUM if c != y_col]
    n_x = 1 if mode.startswith("2D") else 2
    if n_x == 1:
        x_cols = [st.selectbox("Variabel bebas (X)", x_opts,
                               index=x_opts.index("PAD"), format_func=label)]
    else:
        cc1, cc2 = st.columns(2)
        x1 = cc1.selectbox("Variabel bebas X1", x_opts, index=x_opts.index("PAD"),
                           format_func=label)
        rest = [c for c in x_opts if c != x1]
        x2 = cc2.selectbox("Variabel bebas X2", rest, index=rest.index("DAU_Grant")
                           if "DAU_Grant" in rest else 0, format_func=label)
        x_cols = [x1, x2]

    if len(d) < n_x + 2:
        st.warning("Data terlalu sedikit untuk regresi. Tambah kota atau perlebar rentang tahun.")
    else:
        X = d[x_cols].to_numpy(float)
        y = d[y_col].to_numpy(float)
        m = ols(X, y)
        b = m["beta"]

        eq = f"{y_col} = {b[0]:.4f}" + "".join(
            f" {'+' if bi >= 0 else '-'} {abs(bi):.4f}·{xc}" for bi, xc in zip(b[1:], x_cols)
        )
        st.markdown(f'<div class="eq">{eq}</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("R²", f"{m['r2']:.4f}")
        k2.metric("Adj. R²", "-" if np.isnan(m["adj"]) else f"{m['adj']:.4f}")
        k3.metric("RMSE", f"{m['rmse']:.4f}")
        k4.metric("Jumlah observasi", m["n"])

        coef_tbl = pd.DataFrame({
            "Variabel": ["Intercept"] + x_cols,
            "Koefisien": b, "Std. Error": m["se"], "t-stat": m["t"],
        }).round(4)
        st.dataframe(coef_tbl, hide_index=True, use_container_width=True)

        if n_x == 1:
            xs = np.linspace(X.min(), X.max(), 100)
            fig = go.Figure()
            fig.add_scatter(x=X[:, 0], y=y, mode="markers", name="Observasi",
                            marker=dict(color=ACCENT, size=9),
                            text=d["Kota"] + " (" + d["Tahun"].astype(str) + ")")
            fig.add_scatter(x=xs, y=b[0] + b[1] * xs, mode="lines", name="Garis regresi",
                            line=dict(color="#FF6B6B", width=3))
            fig.update_layout(template=PLOT_TEMPLATE, height=500,
                              xaxis_title=label(x_cols[0]), yaxis_title=label(y_col))
        else:
            g1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
            g2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
            G1, G2 = np.meshgrid(g1, g2)
            Z = b[0] + b[1] * G1 + b[2] * G2
            fig = go.Figure()
            fig.add_surface(x=G1, y=G2, z=Z, opacity=0.6, colorscale="Tealgrn",
                            showscale=False, name="Bidang regresi")
            fig.add_scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode="markers", name="Observasi",
                              marker=dict(size=5, color="#FF6B6B"),
                              text=d["Kota"] + " (" + d["Tahun"].astype(str) + ")")
            fig.update_layout(
                template=PLOT_TEMPLATE, height=620,
                scene=dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], zaxis_title=y_col),
                margin=dict(l=0, r=0, t=10, b=0),
            )
        st.plotly_chart(fig, use_container_width=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Aktual vs Prediksi**")
            fig = px.scatter(x=m["yhat"], y=y, template=PLOT_TEMPLATE,
                             labels={"x": "Prediksi", "y": "Aktual"},
                             color_discrete_sequence=[ACCENT])
            lo, hi = float(min(y.min(), m["yhat"].min())), float(max(y.max(), m["yhat"].max()))
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                            line=dict(dash="dash", color="#AAAAAA"), name="y = x")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            st.markdown("**Plot residual**")
            fig = px.scatter(x=m["yhat"], y=m["resid"], template=PLOT_TEMPLATE,
                             labels={"x": "Prediksi", "y": "Residual"},
                             color_discrete_sequence=["#FFB347"])
            fig.add_hline(y=0, line_dash="dash", line_color="#AAAAAA")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔮 Kalkulator prediksi")
        pc = st.columns(len(x_cols))
        vals = [
            col.number_input(xc, value=float(d[xc].mean()), key=f"pred_{xc}")
            for col, xc in zip(pc, x_cols)
        ]
        pred = b[0] + sum(bi * v for bi, v in zip(b[1:], vals))
        st.success(f"Prediksi **{label(y_col)}** = **{pred:.3f}**")

# ----------------------------------------------------------------------------
# 5. Data
# ----------------------------------------------------------------------------
with tab_data:
    st.caption(f"{len(df):,} baris • {df['Kota'].nunique()} kota • "
               f"{df['Tahun'].min()}–{df['Tahun'].max()}")
    shown = df.drop(columns="Nama")
    st.dataframe(shown, use_container_width=True, hide_index=True, height=520)
    st.download_button(
        "⬇️ Unduh data terfilter (CSV)",
        shown.to_csv(index=False).encode("utf-8"),
        file_name="data_fiskal_terfilter.csv",
        mime="text/csv",
    )
    with st.expander("Statistik deskriptif"):
        st.dataframe(shown[ALL_NUM].describe().T.round(3), use_container_width=True)
