import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt, detrend

import plotly.graph_objects as go

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False


# =========================
# Configurações gerais
# =========================
st.set_page_config(page_title="TUG Gyro Segmentation", page_icon="🌀", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

files = [
        'Pct 01_GYR.txt',
        'Pct 02_GYR.txt',
        'Pct 03_GYR.txt',
        'Pct 04_GYR.txt',
        'Pct 112_GYR.txt',
        'Pct 118_GYR.txt',
        'Pct 131_GYR.txt',
        'Pct 178_GYR.txt',
        'Pct 52_GYR.txt',
        'Pct 185_GYR.txt',
        'Pct 29_GYR.txt',
        'Pct 31_GYR.txt',
        'Pct 33_GYR.txt',
        'Pct 42_GYR.txt',
        'Pct 43_GYR.txt',
        'Pct 92_GYR.txt',
        'Pct 102_GYR.txt',
        'Pct 109_GYR.txt',
        'Pct 08_GYR.txt',
        'Pct 04_GYR.txt',
        'Pct 182_GYR.txt',
        'Pct 192_GYR.txt',
        'Pct 37_GYR.txt',
        'Pct 119_GYR.txt',
        'Pct 172_GYR.txt',
        'Pct 36_GYR.txt',
        'Pct 50_GYR.txt',
        'Pct 11_GYR.txt',
        'Pct 12_GYR.txt',
        'Pct 49_GYR.txt',
        'Pct 142_GYR.txt',
        'Pct 57_GYR.txt',
        'Pct 145_GYR.txt',
        'Pct 171_GYR.txt',
        'Pct 103_GYR.txt',
        'Pct 107_GYR.txt',
        'Pct 166_GYR.txt',
        'Pct 190_GYR.txt',
        'Pct 62_GYR.txt',
        'Pct 68_GYR.txt',
        'Pct 108_GYR.txt',
        'Pct 130_GYR.txt',
        'Pct 18_GYR.txt',
        'Pct 26_GYR.txt',
        'Pct 134_GYR.txt',
        'Pct 199_GYR.txt',
        'Pct 58_GYR.txt',
        'Pct 143_GYR.txt',
        'Pct 24_GYR.txt',
        'Pct 53_GYR.txt',
        'Pct 86_GYR.txt',
        'Pct 163_GYR.txt',
        'Pct 76_GYR.txt',
        'Pct 69_GYR.txt',
        'Pct 99_GYR.txt',
        'Pct 133_GYR.txt',
        'Pct 160_GYR.txt',
        'Pct 165_GYR.txt',
        'Pct 194_GYR.txt',
        'Pct 100_GYR.txt',
        'Pct 82_GYR.txt',
        'Pct 91_GYR.txt',
        'Pct 126_GYR.txt',
        'Pct 164_GYR.txt',
        'pct 71_GYR.txt',
        'Pct 77_GYR.txt',
        'Pct 79_GYR.txt',
        'Pct 96_GYR.txt',
        'Pct 155_GYR.txt',
        'Pct 81_GYR.txt',
        'Pct 63_GYR.txt',
        'Pct 65_GYR.txt',
        'Pct 78_GYR.txt',
        'Pct 90_GYR.txt',
        'Pct 113_GYR.txt',
        'Pct 93_GYR.txt',
        'Pct 80_GYR.txt',
        'Pct 161_GYR.txt',
        'Pct 168_GYR.txt',
        'Pct 88_GYR.txt',
    ]
EVENTS = [
    ("t0_start", "Início do sinal"),
    ("t1_turn3m_start", "Início do giro em 3 m"),
    ("t2_turn3m_peak", "Pico do giro em 3 m"),
    ("t3_turn3m_end", "Final do giro em 3 m"),
    ("t4_turnchair_start", "Início do giro na cadeira"),
    ("t5_turnchair_peak", "Pico do giro na cadeira"),
    ("t6_turnchair_end", "Final do giro na cadeira"),
    ("t7_end", "Final da atividade"),
]

FS_TARGET = 100.0
FC = 1.5
FILTER_ORDER = 4


# =========================
# Funções auxiliares
# =========================
def seed_from_text(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _read_semicolon_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

def _ensure_time_seconds(df: pd.DataFrame, time_col: str) -> np.ndarray:
    t = df[time_col].astype(float).to_numpy()
    if np.nanmax(t) > 200:
        t = t / 1000.0
    return t

def preprocess_gyro_norm(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    x = detrend(x, type="linear")
    y = detrend(y, type="linear")
    z = detrend(z, type="linear")

    idx = np.argsort(t)
    t = t[idx]
    x, y, z = x[idx], y[idx], z[idx]

    t_u = np.arange(t[0], t[-1], 1 / FS_TARGET)
    x_i = np.interp(t_u, t, x)
    y_i = np.interp(t_u, t, y)
    z_i = np.interp(t_u, t, z)

    norm = np.sqrt(x_i**2 + y_i**2 + z_i**2)

    b, a = butter(FILTER_ORDER, FC / (0.5 * FS_TARGET), btype="low")
    norm_f = filtfilt(b, a, norm)

    return t_u, norm_f

def make_fig(t, y, events, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="black")))
    ymax = float(np.max(y))
    for k, label in EVENTS:
        if events.get(k) is not None:
            fig.add_vline(x=events[k], line_dash="dash", line_color="gray")
            fig.add_annotation(
                x=events[k], y=ymax, text=label,
                textangle=-90, showarrow=False, yanchor="top",
                font=dict(color="red", size=11)
            )
    fig.update_layout(height=350, title=title)
    return fig


# =========================
# Estado
# =========================
ss = st.session_state
ss.setdefault("uploaded_records", [])
ss.setdefault("record_index", 0)
ss.setdefault("annotations", {})
ss.setdefault("evaluator_id", "")
ss.setdefault("files_order", [])
ss.setdefault("order_seed", None)
ss.setdefault("cursor_time", None)
ss.setdefault("cursor_event", EVENTS[0][0])


# =========================
# App
# =========================
st.title("🌀 Segmentação do Giroscópio no TUG")

# -------------------------
# Identificação do avaliador
# -------------------------
st.subheader("Identificação do avaliador")

evaluator_id = st.text_input(
    "Identidade do avaliador (ex.: AVAL_01)",
    value=ss.evaluator_id,
    max_chars=40
).strip()

if not evaluator_id:
    st.warning("Informe a identidade do avaliador para iniciar.")
    st.stop()

ss.evaluator_id = evaluator_id

# -------------------------
# Geração da ordem específica
# -------------------------
if not ss.files_order:
    seed = seed_from_text(evaluator_id)
    rng = np.random.default_rng(seed)
    order = FILES_BASE.copy()
    rng.shuffle(order)

    ss.files_order = order
    ss.order_seed = seed

st.success(f"Seed do avaliador: {ss.order_seed}")
st.caption("Sequência de apresentação:")
st.code("\n".join(ss.files_order))

# -------------------------
# Carregamento dos arquivos
# -------------------------
if not ss.uploaded_records:
    for fname in ss.files_order:
        path = BASE_DIR / fname
        if not path.exists():
            st.error(f"Arquivo não encontrado: {path}")
            st.stop()

        df = _read_semicolon_txt(path)
        time_col = "DURACAO" if "DURACAO" in df.columns else df.columns[0]

        x_col, y_col, z_col = "AVL EIXO X", "AVL EIXO Y", "AVL EIXO Z"
        if not all(c in df.columns for c in [x_col, y_col, z_col]):
            x_col, y_col, z_col = df.columns[-3:]

        t = _ensure_time_seconds(df, time_col)
        t_u, norm_f = preprocess_gyro_norm(
            t, df[x_col].values, df[y_col].values, df[z_col].values
        )

        ss.uploaded_records.append({
            "name": fname,
            "t": t_u,
            "norm": norm_f
        })

# -------------------------
# Registro atual
# -------------------------
i = ss.record_index
rec = ss.uploaded_records[i]

st.subheader(f"Registro {i+1}/{len(ss.uploaded_records)} — {rec['name']}")

event_times = ss.annotations.get(rec["name"], {k: None for k, _ in EVENTS})

fig = make_fig(rec["t"], rec["norm"], event_times, title="Clique para marcar eventos")

if PLOTLY_EVENTS_AVAILABLE:
    clicked = plotly_events(fig, click_event=True)
    if clicked:
        ss.cursor_time = clicked[0]["x"]

    ss.cursor_event = st.selectbox(
        "Evento selecionado",
        options=[k for k, _ in EVENTS],
        format_func=lambda k: dict(EVENTS)[k]
    )

    if st.button("Atribuir cursor ao evento") and ss.cursor_time is not None:
        event_times[ss.cursor_event] = float(ss.cursor_time)

# Sliders
for k, label in EVENTS:
    event_times[k] = st.slider(
        label,
        float(rec["t"][0]),
        float(rec["t"][-1]),
        float(event_times[k]) if event_times[k] else float(rec["t"][0]),
        0.01,
        key=f"{rec['name']}_{k}"
    )

st.plotly_chart(make_fig(rec["t"], rec["norm"], event_times))

# -------------------------
# Salvar e navegar
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Salvar"):
        ss.annotations[rec["name"]] = event_times
        st.success("Salvo.")

with col2:
    if st.button("⬅️ Anterior", disabled=i == 0):
        ss.record_index -= 1
        st.rerun()

with col3:
    if st.button("➡️ Próximo", disabled=i == len(ss.uploaded_records) - 1):
        ss.record_index += 1
        st.rerun()

# -------------------------
# Exportação
# -------------------------
if ss.annotations:
    rows = []
    for name, ev in ss.annotations.items():
        row = {
            "evaluator_id": ss.evaluator_id,
            "order_seed": ss.order_seed,
            "record_name": name,
        }
        row.update(ev)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    st.download_button(
        "⬇️ Baixar CSV",
        data=df_out.to_csv(index=False).encode(),
        file_name="tug_segmentation_results.csv"
    )
