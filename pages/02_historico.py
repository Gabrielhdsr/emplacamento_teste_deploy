import streamlit as st
import pandas as pd
import altair as alt

import data as dt
import lib as lb
import ui

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(layout="wide", page_title="Histórico de Mercado", page_icon="📈")

# ============================================================
# DADOS E CONSTANTES
# ============================================================
df = dt.carregar_emplacamento('arquivos/Emplacamento/*.xlsx')

SC = "SOBRE CHASSI"
SR = "SEMIRREBOQUE"
FACCHINI = "FACCHINI"

# Cores
COR_FACCHINI = "#b91c1c"
COR_MERCADO = "#3b82f6"
COR_OUTROS = "#94a3b8"

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def get_delta_info(pct):
    COR_POS_LIGHT = "#4ade80"
    COR_NEG_LIGHT = "#fca5a5"
    if pd.isna(pct):
        return "", "white"
    sinal = "▲" if pct >= 0 else "▼"
    delta_txt = f"{sinal} {abs(pct):.1%}".replace('.', ',')
    cor = COR_POS_LIGHT if pct >= 0 else COR_NEG_LIGHT
    return delta_txt, cor

# ============================================================
# PREPARAÇÃO DE DADOS
# ============================================================
def preparar_dados_volume_anual(df_raw, tipo=None):
    df_f = df_raw.copy()
    if tipo:
        df_f = df_f[df_f["Tipo"] == tipo]

    # Totais por ano
    df_totais = (
        df_f.groupby("Ano")["Qtde"].sum()
        .reset_index()
        .rename(columns={"Qtde": "Total_Ano"})
        .sort_values("Ano")
    )
    df_totais["Delta_Pct"] = df_totais["Total_Ano"].pct_change()

    # Facchini por ano
    df_fac = (
        df_f[df_f["Implementadora"] == FACCHINI]
        .groupby("Ano")["Qtde"].sum()
        .reset_index()
        .rename(columns={"Qtde": "Vol_Facchini"})
    )

    df_chart = pd.merge(df_totais, df_fac, on="Ano", how="left").fillna(0)
    df_chart["Vol_Resto"] = df_chart["Total_Ano"] - df_chart["Vol_Facchini"]
    df_chart["Share_Facchini"] = df_chart["Vol_Facchini"] / df_chart["Total_Ano"]

    # Heurística visual (tamanho do vermelho vs altura do chart)
    CHART_HEIGHT_PX = 380
    MIN_2LIN_PX = 34
    MIN_1LIN_PX = 18

    max_total = float(df_chart["Total_Ano"].max()) if len(df_chart) else 1.0
    limiar_2lin = max_total * (MIN_2LIN_PX / CHART_HEIGHT_PX)
    limiar_1lin = max_total * (MIN_1LIN_PX / CHART_HEIGHT_PX)

    rows = []
    for _, r in df_chart.iterrows():
        delta_txt, delta_cor = get_delta_info(r["Delta_Pct"])

        # --- Azul (Mercado) ---
        y0_m = 0.0
        y1_m = float(r["Vol_Resto"])
        rows.append({
            "Ano": r["Ano"],
            "Segmento": "Mercado",
            "Y0": y0_m,
            "Y1": y1_m,
            "Y_Texto": (y0_m + y1_m) / 2,
            "Txt_Cima": lb.formatar_br(r["Total_Ano"]),
            "Txt_Baixo": delta_txt,
            "Cor_Txt_Cima": "white",
            "Cor_Txt_Baixo": delta_cor,
            "Cor_Barra": COR_MERCADO,
            "Font_Top": 13,
            "Font_Bot": 11,
            "Ordem": 1
        })

        # --- Vermelho (Facchini) ---
        altura_vermelho = float(r["Vol_Facchini"])
        if altura_vermelho <= limiar_1lin:
            font_top, font_bot = 10, 0
            txt_baixo = ""
        elif altura_vermelho <= limiar_2lin:
            font_top, font_bot = 11, 9
            txt_baixo = f"{r['Share_Facchini']*100:.1f}%".replace('.', ',')
        else:
            font_top, font_bot = 13, 11
            txt_baixo = f"{r['Share_Facchini']*100:.1f}%".replace('.', ',')

        y0_f = float(r["Vol_Resto"])
        y1_f = float(r["Total_Ano"])
        rows.append({
            "Ano": r["Ano"],
            "Segmento": "Facchini",
            "Y0": y0_f,
            "Y1": y1_f,
            "Y_Texto": (y0_f + y1_f) / 2,
            "Txt_Cima": lb.formatar_br(r["Vol_Facchini"]),
            "Txt_Baixo": txt_baixo,
            "Cor_Txt_Cima": "white",
            "Cor_Txt_Baixo": "white",
            "Cor_Barra": COR_FACCHINI,
            "Font_Top": font_top,
            "Font_Bot": font_bot,
            "Ordem": 2
        })

    return pd.DataFrame(rows)

def preparar_dados_share_evolution(df_raw, tipo=None):
    df_f = df_raw.copy()
    if tipo:
        df_f = df_f[df_f["Tipo"] == tipo]

    ranking_geral = (
        df_f[df_f["Implementadora"] != FACCHINI]
        .groupby("Implementadora")["Qtde"].sum()
        .sort_values(ascending=False)
    )
    top_concorrentes = ranking_geral.head(4).index.tolist()

    def categorizar(imp):
        if imp == FACCHINI:
            return FACCHINI
        if imp in top_concorrentes:
            return imp
        return "OUTROS"

    df_f["Player_Chart"] = df_f["Implementadora"].apply(categorizar)
    df_chart = df_f.groupby(["Ano", "Player_Chart"])["Qtde"].sum().reset_index()

    df_totais = df_f.groupby("Ano")["Qtde"].sum().reset_index().rename(columns={"Qtde": "Total"})
    df_chart = pd.merge(df_chart, df_totais, on="Ano")
    df_chart["Share"] = df_chart["Qtde"] / df_chart["Total"]

    return df_chart

# ============================================================
# GRÁFICOS (ALTAIR)
# ============================================================
def plot_historico_volume(df_dados: pd.DataFrame):
    if df_dados.empty:
        return alt.Chart(pd.DataFrame({"Ano": [], "Qtde": []})).mark_text(text="Sem dados")

    y_max = float(df_dados["Y1"].max())
    y_domain_max = y_max * 1.03

    base = alt.Chart(df_dados).encode(
        x=alt.X(
            "Ano:O",
            axis=alt.Axis(labelAngle=0, title=None, labelFontSize=12, labelFontWeight="bold"),
            scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2)
        )
    )

    # AZUL: aqui fica o eixo Y com GRID (garante que aparece)
    bars_azul = base.transform_filter(
        alt.datum.Segmento == "Mercado"
    ).mark_bar(size=52).encode(
        y=alt.Y(
            "Y1:Q",
            scale=alt.Scale(domain=[0, y_domain_max]),
            axis=alt.Axis(
                title=None,
                labels=False,
                ticks=False,
                grid=True,
                gridColor="#e2e8f0",
                gridDash=[4, 4],
                domain=False
            )
        ),
        y2="Y0:Q",
        color=alt.Color("Cor_Barra:N", scale=None, legend=None)
    )

    # VERMELHO: topo arredondado
    bars_vermelho = base.transform_filter(
        alt.datum.Segmento == "Facchini"
    ).mark_bar(
        size=52,
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6
    ).encode(
        y=alt.Y("Y1:Q", scale=alt.Scale(domain=[0, y_domain_max]), axis=None),
        y2="Y0:Q",
        color=alt.Color("Cor_Barra:N", scale=None, legend=None)
    )

    # Texto linha 1
    txt_top = base.mark_text(dy=-6, fontWeight="bold").encode(
        y=alt.Y("Y_Texto:Q", scale=alt.Scale(domain=[0, y_domain_max])),
        text="Txt_Cima:N",
        color=alt.Color("Cor_Txt_Cima:N", scale=None),
        size=alt.Size("Font_Top:Q", scale=None, legend=None)
    )

    # Texto linha 2 (só quando Font_Bot > 0)
    txt_bot = base.transform_filter(
        alt.datum.Font_Bot > 0
    ).mark_text(dy=10, fontWeight="bold").encode(
        y=alt.Y("Y_Texto:Q", scale=alt.Scale(domain=[0, y_domain_max])),
        text="Txt_Baixo:N",
        color=alt.Color("Cor_Txt_Baixo:N", scale=None),
        size=alt.Size("Font_Bot:Q", scale=None, legend=None)
    )

    return (bars_azul + bars_vermelho + txt_top + txt_bot).properties(
        height=380,
        background="white"
    ).configure_view(
        strokeWidth=0,
        fill="white"
    )

def plot_evolucao_share(df_dados):
    top_players = (
        df_dados.groupby("Player_Chart")["Qtde"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if FACCHINI in top_players:
        top_players.remove(FACCHINI)
    if "OUTROS" in top_players:
        top_players.remove("OUTROS")

    stack_order = [FACCHINI] + top_players + ["OUTROS"]

    colors = (
        [COR_FACCHINI] +
        ["#1e40af", "#3b82f6", "#60a5fa", "#93c5fd"][:len(top_players)] +
        [COR_OUTROS]
    )

    base = alt.Chart(df_dados).encode(
        x=alt.X("Ano:O", axis=alt.Axis(labelAngle=0, title=None, labelFontSize=12)),
        y=alt.Y(
            "Qtde:Q",
            stack="normalize",
            axis=alt.Axis(format=".0%", title=None, grid=False)
        ),
        order=alt.Order("Player_Chart"),
        color=alt.Color(
            "Player_Chart",
            scale=alt.Scale(domain=stack_order, range=colors),
            legend=alt.Legend(title="Players", orient="bottom", columns=3)
        ),
        tooltip=["Ano", "Player_Chart", alt.Tooltip("Share", format=".1%")]
    )

    return base.mark_area(opacity=0.9).properties(
        height=380,
        background="white"
    ).configure_view(
        strokeWidth=0,
        fill="white"
    )

# ============================================================
# LAYOUT
# ============================================================
ui.header("Histórico & Tendências", "Evolução anual de <b>Volume</b> e <b>Market Share</b>")
ui.apply_style()

# (opcional) reforça branco no svg em alguns temas
st.markdown("""
<style>
div[data-testid="stAltairChart"] svg { background: white !important; }
</style>
""", unsafe_allow_html=True)

# --- SEÇÃO 1: CONSOLIDADO ---
ui.section("1. Visão Consolidada (Mercado Total)")
c1_vol, c1_share = st.columns(2, gap="large")

with c1_vol:
    st.markdown("##### 📦 Evolução de Volume")
    df_vol_total = preparar_dados_volume_anual(df)
    st.altair_chart(plot_historico_volume(df_vol_total), use_container_width=True)

with c1_share:
    st.markdown("##### 🥧 Evolução de Share (%)")
    df_share_total = preparar_dados_share_evolution(df)
    st.altair_chart(plot_evolucao_share(df_share_total), use_container_width=True)

# --- SEÇÃO 2: SOBRE CHASSI ---
st.write("---")
ui.section(f"2. Segmento: {SC}")
c2_vol, c2_share = st.columns(2, gap="large")

with c2_vol:
    st.markdown(f"##### 📦 Volume: {SC}")
    df_vol_sc = preparar_dados_volume_anual(df, SC)
    st.altair_chart(plot_historico_volume(df_vol_sc), use_container_width=True)

with c2_share:
    st.markdown(f"##### 🥧 Share: {SC}")
    df_share_sc = preparar_dados_share_evolution(df, SC)
    st.altair_chart(plot_evolucao_share(df_share_sc), use_container_width=True)

# --- SEÇÃO 3: SEMIRREBOQUE ---
st.write("---")
ui.section(f"3. Segmento: {SR}")
c3_vol, c3_share = st.columns(2, gap="large")

with c3_vol:
    st.markdown(f"##### 📦 Volume: {SR}")
    df_vol_sr = preparar_dados_volume_anual(df, SR)
    st.altair_chart(plot_historico_volume(df_vol_sr), use_container_width=True)

with c3_share:
    st.markdown(f"##### 🥧 Share: {SR}")
    df_share_sr = preparar_dados_share_evolution(df, SR)
    st.altair_chart(plot_evolucao_share(df_share_sr), use_container_width=True)
