### IMPORTS ###
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter
import numpy as np
import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import streamlit as st
import lib as lb
import graficos as gf
import data as dt
from data import load_emplacamento
import filtros as flt

st.subheader("Share por Categoria")

df = load_emplacamento('arquivos/Emplacamento/*.xlsx')

row = st.columns([5,6,7])  # 2 selects compactos + “respiro”

with row[0]:
    df_flt = df[df['Ano']>= 2024]
    ym_ini, ym_fim, meta =  flt.filtro_periodo(df_flt)
with row[2]:
    # tipos = ["Todos"] + sorted(df["Tipo"].dropna().astype(str).unique())
    # tipo_sel = st.selectbox("Tipo", tipos, index=0, label_visibility="collapsed")
    tipo_sel = flt.filtro_tipo_segmented(df)

# --- tabela: top10 + OUTROS, % do total ---
tab = gf.tabela_share_mix(
    df,
    top_n=10,
    tipo=tipo_sel,
    ym_ini=ym_ini,
    ym_fim=ym_fim,
)

# --- exibição ---
if tab.empty:
    st.info("Sem dados para os filtros selecionados.")
else:
    st.dataframe(
        tab.style.format(lambda v: "" if pd.isna(v) else f"{v:.2f}%".replace(".", ",")),
        use_container_width=True,
        height=(len(tab) + 2) * 35,
    )