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
from data import load_anfavea
from data import load_emplacamento

st.set_page_config(page_title="Dashboard", layout="wide")
col1, col2, col3 = st.columns([1,6,1])

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


df_anfavea = dt.load_anfavea('arquivos/SeriesTemporais_Autoveiculos (1).xlsm')
df_emplacamento = dt.load_emplacamento('arquivos/Emplacamento/*.xlsx')

df = df_emplacamento.copy()
df_sc = df[df['Tipo'] == 'SOBRE CHASSI']
df_sr = df[df['Tipo'] == 'SEMIRREBOQUE']
ano_atual = datetime.now().year

tabela = lb.prepara_emplacamento(df, ano_atual)
tabela_sc = lb.prepara_emplacamento(df_sc, ano_atual)
tabela_sr = lb.prepara_emplacamento(df_sr, ano_atual)

# === depois de criar: tabela, tabela_sc, tabela_sr, e ano_atual ===

LOGO_PATH       = "arquivos/logoprincipal.png"
ARROW_UP_PATH   = "arquivos/seta-verde-para-cima.png"
ARROW_DOWN_PATH = "arquivos/seta-vermelha-para-baixo.png"

### AGRUPA PRODUÇÃO DE CAMINHÕES POR ANO
valores_prod_ano = df_anfavea.groupby('Ano')['Produção'].sum()
variacao_prod_ano = valores_prod_ano.pct_change() * 100

# === 3 GRÁFICOS EMPILHADOS (Total, SC e SR) ===

# 1) TOTAL
fig_total = gf.grafico_share_facchini_generico(
    tabela,
    ano_atual=ano_atual,
    logo_path=LOGO_PATH,
    seta_up_path=ARROW_UP_PATH,
    seta_down_path=ARROW_DOWN_PATH,
    label_fac="FACCHINI",
    label_outros="Mercado Geral",
)

# supondo que você já tenha valores_prod_ano (Series) e variacao_prod_ano (Series)
#st.header("Produção de Caminhões – 2014 / 2025")
fig_prod, _ = gf.grafico_producao_caminhao(
    valores_prod_ano,
    variacao_prod_ano,
    titulo="Produção Caminhões 2013 a 2025",
    y_max=200000,
    y_step=20000,
    renomear_previsao_ano=2025,
    mostrar_gradiente=True,
    df=df_anfavea
)

# 3) SEMIRREBOQUE (SR)
#st.header("Market Share SC FACCHINI – 2014 / 2025")
fig_sr = gf.grafico_share_facchini_generico(
    tabela_sr,
    ano_atual=ano_atual,
    logo_path=LOGO_PATH,
    seta_up_path=ARROW_UP_PATH,
    seta_down_path=ARROW_DOWN_PATH,
    label_fac="FACCHINI (SR)",
    label_outros="Mercado (SR)",
)

# 2) SOBRE CHASSI (SC)
#st.header("Market Share SR FACCHINI – 2014 / 2025")
fig_sc = gf.grafico_share_facchini_generico(
    tabela_sc,
    ano_atual=ano_atual,
    logo_path=LOGO_PATH,
    seta_up_path=ARROW_UP_PATH,
    seta_down_path=ARROW_DOWN_PATH,
    label_fac="FACCHINI (SC)",
    label_outros="Mercado (SC)",
)

# total
#st.header("Distribuição Market Share SC + SR / 2014 - 2025")
fig_ms_total = gf.plot_share_area(
    df,
    ano_atual, 
    previsao_media3m=lb.previsao_media3m
)

# segmentado
#st.header("Distribuição Market Share SR – 2014 / 2024")
fig_ms_sc = gf.plot_share_area(
    df, 
    ano_atual, 
    tipo="SOBRE CHASSI", 
    col_tipo="Tipo",
    previsao_media3m=lb.previsao_media3m
)

#st.header("Distribuição Market Share SR – 2014 / 2024")
fig_ms_sr = gf.plot_share_area(
    df, 
    ano_atual, 
    tipo="SEMIRREBOQUE", 
    col_tipo="Tipo",
    previsao_media3m=lb.previsao_media3m
)

with col2:
    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Market Share FACCHINI – 2014 / 2025</h1>", unsafe_allow_html=True)
    st.pyplot(fig_total, use_container_width=True, clear_figure=True)

    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Produção de Caminhões – 2014 / 2025</h1>", unsafe_allow_html=True)
    st.pyplot(fig_prod, use_container_width=True, clear_figure=True)

    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Market Share SR FACCHINI – 2014 / 2025</h1>", unsafe_allow_html=True)
    st.pyplot(fig_sr, use_container_width=True, clear_figure=True)

    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Market Share SC FACCHINI – 2014 / 2025</h1>", unsafe_allow_html=True)
    st.pyplot(fig_sc, use_container_width=True, clear_figure=True)

    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Distribuição Market Share SC + SR / 2014 - 2025</h1>", unsafe_allow_html=True)
    st.pyplot(fig_ms_total, use_container_width=True, clear_figure=True)

    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Distribuição Market Share SC – 2014 / 2024</h1>", unsafe_allow_html=True)
    st.pyplot(fig_ms_sc, use_container_width=True, clear_figure=True)

    st.markdown("<h1 style='font-size:45px; font-weight:bold; color:black;'>Distribuição Market Share SC – 2014 / 2024</h1>", unsafe_allow_html=True)
    st.pyplot(fig_ms_sr, use_container_width=True, clear_figure=True)
