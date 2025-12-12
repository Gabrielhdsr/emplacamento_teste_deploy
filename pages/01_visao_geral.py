import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import numpy as np

import data as dt
import lib as lb
import ui 

# ============================================================
# CONFIGURAÇÃO E ESTADO
# ============================================================
st.set_page_config(layout="wide", page_title="Market Panorama", page_icon="📊")

if 'view_quarter' not in st.session_state:
    st.session_state['view_quarter'] = 'cards'
if 'view_month' not in st.session_state:
    st.session_state['view_month'] = 'cards'

def set_quarter_view(view):
    st.session_state['view_quarter'] = view
    
def set_month_view(view):
    st.session_state['view_month'] = view

# ============================================================
# DADOS E CONSTANTES
# ============================================================
df = dt.carregar_emplacamento('arquivos/Emplacamento/*.xlsx')

ano_atual = datetime.now().year
ano_anterior = ano_atual - 1

SC = "SOBRE CHASSI"
SR = "SEMIRREBOQUE"
FACCHINI = "FACCHINI"

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def fmt_pct(valor):
    """Formata float para porcentagem BR (ex: 18.5 -> 18,5%)"""
    return f"{valor:.1f}%".replace('.', ',')

def calcular_total_acumulado(df, ano, tipo=None):
    df_f = df[df["Ano"] == ano]
    if tipo: df_f = df_f[df_f["Tipo"] == tipo]
    return int(df_f["Qtde"].sum())

def calcular_media_recente(df, ano, tipo=None):
    mes_atual = datetime.now().month
    if mes_atual <= 1: return 0 
    inicio = max(1, mes_atual - 3) 
    lista_meses = list(range(inicio, mes_atual))
    df_f = df[(df["Ano"] == ano) & (df["Mes"].isin(lista_meses))]
    if tipo: df_f = df_f[df_f["Tipo"] == tipo]
    soma = df_f["Qtde"].sum()
    qtd = len(lista_meses)
    return soma / qtd if qtd > 0 else 0

def calcular_share_periodo(df, ano, meses, tipo):
    df_f = df[(df["Ano"]==ano) & (df["Mes"].isin(meses))]
    if tipo: df_f = df_f[df_f["Tipo"]==tipo]
    total = df_f["Qtde"].sum()
    if total == 0: return 0.0
    facchini = df_f[df_f["Implementadora"] == FACCHINI]["Qtde"].sum()
    return (facchini / total) * 100

def calcular_vol_facchini(df, ano, meses, tipo):
    df_f = df[(df["Ano"]==ano) & (df["Mes"].isin(meses)) & (df["Implementadora"]==FACCHINI)]
    if tipo: df_f = df_f[df_f["Tipo"]==tipo]
    return int(df_f["Qtde"].sum())

def variacao(atual, anterior):
    if anterior == 0: return 0
    return (atual / anterior) - 1

# ============================================================
# FUNÇÕES DE GRÁFICO (REFINADO: TOTAL NA BASE, SHARE EMBAIXO DO VOL)
# ============================================================
def plot_comparativo_trimestral(dados_lista):
    chart_data = []
    
    COR_MERCADO = "#3b82f6"
    COR_FACCHINI = "#b91c1c"

    for d in dados_lista:
        trim = d['Trimestre']
        
        # --- LY ---
        vol_fac_ly = d['Vol_Fac_LY_Media']
        vol_total_ly = d['Vol_LY']

        chart_data.append({
            "Trimestre": trim, "Grupo": "LY", "SubGrupo": "Mercado",
            "Valor_Barra": vol_total_ly - vol_fac_ly,
            "Posicao_Texto": (vol_total_ly - vol_fac_ly) / 2,
            "Label_Texto": lb.formatar_br(vol_total_ly),
            "Label_Ano": str(ano_anterior),
            "Cor": COR_MERCADO,
            "StackOrdem": 1,
            "Label_Cor": "white"
        })

        chart_data.append({
            "Trimestre": trim, "Grupo": "LY", "SubGrupo": "Facchini",
            "Valor_Barra": vol_fac_ly,
            "Posicao_Texto": (vol_total_ly - vol_fac_ly) + vol_fac_ly / 2,
            "Label_Texto": f"{lb.formatar_br(vol_fac_ly)}\n{fmt_pct(d['Share_Facchini_LY_Pct'])}",
            "Label_Ano": str(ano_anterior),
            "Cor": COR_FACCHINI,
            "StackOrdem": 2,
            "Label_Cor": "white"
        })

        # --- ATUAL ---
        if d['Has_Data']:
            vol_fac_atual = d['Vol_Fac_Media']
            vol_total_atual = d['Vol_Atual']

            chart_data.append({
                "Trimestre": trim, "Grupo": "Atual", "SubGrupo": "Mercado",
                "Valor_Barra": vol_total_atual - vol_fac_atual,
                "Posicao_Texto": (vol_total_atual - vol_fac_atual) / 2,
                "Label_Texto": lb.formatar_br(vol_total_atual),
                "Label_Ano": str(ano_atual),
                "Cor": COR_MERCADO,
                "StackOrdem": 1,
                "Label_Cor": "white"
            })

            chart_data.append({
                "Trimestre": trim, "Grupo": "Atual", "SubGrupo": "Facchini",
                "Valor_Barra": vol_fac_atual,
                "Posicao_Texto": (vol_total_atual - vol_fac_atual) + vol_fac_atual / 2,
                "Label_Texto": f"{lb.formatar_br(vol_fac_atual)}\n{fmt_pct(d['Share_Facchini_Pct'])}",
                "Label_Ano": str(ano_atual),
                "Cor": COR_FACCHINI,
                "StackOrdem": 2,
                "Label_Cor": "white"
            })

    df_chart = pd.DataFrame(chart_data)

    grupo_offset = alt.XOffset('Grupo:N', sort=['LY', 'Atual'])

    base = alt.Chart(df_chart).encode(
        x=alt.X(
            'Trimestre:O',
            axis=alt.Axis(
                labelAngle=0,
                title=None,
                labelFontSize=14,
                labelFontWeight=900,
                tickSize=0
            ),
            scale=alt.Scale(
                paddingOuter=0.35,   # <<< MARGEM IGUAL DOS DOIS LADOS
                paddingInner=0.25    # <<< ESPAÇO ENTRE TRIMESTRES
            )
        ),
        xOffset=grupo_offset,
        order=alt.Order('StackOrdem:Q', sort='ascending')
    )

    bars = base.mark_bar(
        size=55,
        cornerRadiusTopLeft=8,
        cornerRadiusTopRight=8
    ).encode(
        y=alt.Y(
            'Valor_Barra:Q',
            axis=alt.Axis(
                title=None,
                labels=False,
                grid=True,
                gridColor='#e2e8f0',
                gridDash=[4, 4],
                domain=False
            )
        ),
        color=alt.Color('Cor:N', scale=None, legend=None)
    )

    text_labels = base.mark_text(
        baseline='middle',
        lineBreak='\n',
        fontSize=12,
        fontWeight='bold'
    ).encode(
        y='Posicao_Texto:Q',
        text='Label_Texto:N',
        color=alt.Color('Label_Cor:N', scale=None)
    ).transform_filter(
        alt.datum.Valor_Barra > 100
    )

    text_years = base.mark_text(
        align='center',
        baseline='top',
        dy=10,
        fontSize=11,
        fontWeight='bold',
        color='#64748b'
    ).encode(
        y=alt.datum(0),
        text='Label_Ano:N'
    ).transform_filter(
        alt.datum.StackOrdem == 1
    )

    delta_data = []

    for d in dados_lista:
        if not d['Has_Data']:
            continue

        var_vol = variacao(d['Vol_Atual'], d['Vol_LY']) * 100
        var_share = d['Share_Facchini_Pct'] - d['Share_Facchini_LY_Pct']

        texto = (
            f"{'▲' if var_vol >= 0 else '▼'} {var_vol:+.1f}%\n"
            f"{'▲' if var_share >= 0 else '▼'} {var_share:+.1f} p.p."
        )

        delta_data.append({
            "Trimestre": d['Trimestre'],
            "Texto": texto,
            "Cor": "#16a34a" if (var_vol >= 0 and var_share >= 0) else "#dc2626"
        })

    df_delta = pd.DataFrame(delta_data)

    deltas = alt.Chart(df_delta).mark_text(
        align='center',
        baseline='middle',
        fontSize=12,
        fontWeight=900,
        lineBreak='\n'
    ).encode(
        x=alt.X('Trimestre:O'),
        xOffset=alt.value(0),   # exatamente ENTRE LY e Atual
        y=alt.value(20),        # posição vertical fixa
        text='Texto:N',
        color=alt.Color('Cor:N', scale=None)
    )

 # =========================
    # VARIAÇÕES (CORES SEPARADAS) + SHARE EM CIMA
    # =========================
    delta_rows = []

    for d in dados_lista:
        if not d["Has_Data"]:
            continue

        var_vol_pct = variacao(d["Vol_Atual"], d["Vol_LY"]) * 100
        var_share_pp = d["Share_Facchini_Pct"] - d["Share_Facchini_LY_Pct"]

        topo = max(d["Vol_LY"], d["Vol_Atual"])

        # Linha de VOLUME (cor só do volume)
        delta_rows.append({
            "Trimestre": d["Trimestre"],
            "Texto": f"{'▲' if var_vol_pct >= 0 else '▼'} {var_vol_pct:+.1f} %",
            "YPos": topo * 0.48,  # fica abaixo do share
            "Cor": "#16a34a" if var_vol_pct >= 0 else "#dc2626"
        })

        # Linha de SHARE (cor só do share) — EM CIMA
        delta_rows.append({
            "Trimestre": d["Trimestre"],
            "Texto": f"{'▲' if var_share_pp >= 0 else '▼'} {var_share_pp:+.1f} p.p.",
            "YPos": topo * 0.53,  # mais alto
            "Cor": "#16a34a" if var_share_pp >= 0 else "#dc2626"
        })

    df_delta = pd.DataFrame(delta_rows)

    deltas = alt.Chart(df_delta).mark_text(
        align="center",
        baseline="middle",
        fontSize=12,
        fontWeight=900
    ).encode(
        x=alt.X("Trimestre:O"),     # sem xOffset = centralizado entre LY e Atual
        y=alt.Y("YPos:Q"),
        text="Texto:N",
        color=alt.Color("Cor:N", scale=None)
    )

    return (
        bars + text_labels + text_years + deltas   # <<< adiciona deltas aqui
    ).properties(
        height=400,
        padding={"left": 40, "right": 40, "top": 10, "bottom": 10}
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )
        
def plot_evolucao_mensal(dados_lista):
    df_chart = pd.DataFrame(dados_lista).copy()
    df_chart = df_chart[df_chart["Has_Data"] == True]

    COR_MERCADO = "#3b82f6"
    COR_FACCHINI = "#b91c1c"

    ordem_meses = ["JAN","FEV","MAR","ABR","MAI","JUN","JUL","AGO","SET","OUT","NOV","DEZ"]

    df_chart["Mes_ord"] = df_chart["Mes"].apply(lambda m: ordem_meses.index(m) if m in ordem_meses else 999)
    df_chart = df_chart.sort_values("Mes_ord")  
    
    chart_data = []
    prev_total = None

    for _, d in df_chart.iterrows():
        vol_total = int(d["Vol_Atual"])
        vol_fac = int(d["Vol_Fac"])
        vol_mkt = max(0, vol_total - vol_fac)
        share = float(d["Share_Raw"])

        # crescimento do mercado vs mês anterior
        if prev_total is None or prev_total == 0:
            mom_str = "—"
        else:
            mom = (vol_total / prev_total - 1) * 100
            mom_str = f"{'▲' if mom >= 0 else '▼'} {mom:+.1f}%".replace(".", ",")

        # Mercado (azul): TOTAL + MoM embaixo
        chart_data.append({
            "Mes": d["Mes"],
            "SubGrupo": "Mercado",
            "Valor_Barra": vol_mkt,
            "Posicao_Texto": vol_mkt / 2 if vol_mkt > 0 else 0,
            "Label_Texto": f"{lb.formatar_br(vol_total)}\n{mom_str}",
            "Cor": COR_MERCADO,
            "StackOrdem": 1,
            "Label_Cor": "white"
        })

        # Facchini (vermelho): VOL + SHARE
        chart_data.append({
            "Mes": d["Mes"],
            "SubGrupo": "Facchini",
            "Valor_Barra": vol_fac,
            "Posicao_Texto": vol_mkt + (vol_fac / 2 if vol_fac > 0 else 0),
            "Label_Texto": f"{lb.formatar_br(vol_fac)}\n{fmt_pct(share)}",
            "Cor": COR_FACCHINI,
            "StackOrdem": 2,
            "Label_Cor": "white"
        })

        prev_total = vol_total

    df_plot = pd.DataFrame(chart_data)

    base = alt.Chart(df_plot).encode(
        x=alt.X(
            "Mes:O",
            sort=ordem_meses,
            axis=alt.Axis(
                title=None,
                labelAngle=0,
                labelFontSize=12,
                labelFontWeight=700,
                tickSize=0
            ),
            scale=alt.Scale(paddingOuter=0.35, paddingInner=0.25)
        ),
        order=alt.Order("StackOrdem:Q", sort="ascending")
    )

    bars = base.mark_bar(
        size=55,
        cornerRadiusTopLeft=8,
        cornerRadiusTopRight=8
    ).encode(
        y=alt.Y(
            "Valor_Barra:Q",
            axis=alt.Axis(
                title=None,
                labels=False,
                grid=True,
                gridColor="#e2e8f0",
                gridDash=[4, 4],
                domain=False
            )
        ),
        color=alt.Color("Cor:N", scale=None, legend=None)
    )

    labels = base.mark_text(
        baseline="middle",
        lineBreak="\n",
        fontSize=11,
        fontWeight="bold"
    ).encode(
        y="Posicao_Texto:Q",
        text="Label_Texto:N",
        color=alt.Color("Label_Cor:N", scale=None)
    ).transform_filter(
        alt.datum.Valor_Barra > 0
    )

    return (bars + labels).properties(
        height=350
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )

# ============================================================
# HEADER E FILTROS
# ============================================================
c_head, c_filt_area = st.columns([1.2, 1])

with c_head:
    ui.header("Panorama de Mercado", f"Visão Estratégica • <b>{ano_atual}</b> vs {ano_anterior} (LY)")

with c_filt_area:
    st.write("") 
    c_mix, c_seg = st.columns([1.5, 2])
    with c_seg:
        st.write("**Segmento**")
        filtro_segmento = st.radio("Segmento", ["Consolidado", "Sobre Chassi", "Semirreboque"], horizontal=True, label_visibility="collapsed")
    
    tipo_selecionado = None
    if filtro_segmento == "Sobre Chassi": tipo_selecionado = SC
    elif filtro_segmento == "Semirreboque": tipo_selecionado = SR

    with c_mix:
        if "Mix Produto" in df.columns:
            st.write("**Mix Produto**")
            df_prods = df if not tipo_selecionado else df[df["Tipo"] == tipo_selecionado]
            opcoes_mix = sorted(df_prods["Mix Produto"].dropna().unique())
            sel_mix = st.multiselect("Mix", options=opcoes_mix, placeholder="Filtrar Mix...", label_visibility="collapsed")
            if sel_mix: df = df[df["Mix Produto"].isin(sel_mix)]

ui.apply_style()

st.markdown("""
<style>
/* Card do gráfico Altair */
div[data-testid="stVegaLiteChart"] > div {
    background-color: white;
    border-radius: 16px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);

    width: 100%;
    padding: 32px !important;
    box-sizing: border-box;
    overflow: hidden;

    /* 🔥 ISSO RESOLVE */
    display: flex;
    justify-content: center;   /* centraliza horizontal */
    align-items: center;       /* centraliza vertical */
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CÁLCULOS KPI
# ============================================================
mes_atual = datetime.now().month
meses_restantes = 12 - (mes_atual - 1)

vol_acum = calcular_total_acumulado(df, ano_atual, tipo_selecionado)
vol_LY = calcular_total_acumulado(df, ano_anterior, tipo_selecionado)
vol_fac_acum = calcular_vol_facchini(df, ano_atual, list(range(1, 13)), tipo_selecionado)
media_3m = calcular_media_recente(df, ano_atual, tipo_selecionado)
forecast_vol = vol_acum + (media_3m * meses_restantes)
var_forecast = variacao(forecast_vol, vol_LY)
share_acum = calcular_share_periodo(df, ano_atual, list(range(1, mes_atual)), tipo_selecionado)
share_acum_LY = calcular_share_periodo(df, ano_anterior, list(range(1, 13)), tipo_selecionado)
media_3m_fac = calcular_media_recente(df[df["Implementadora"]==FACCHINI], ano_atual, tipo_selecionado)
forecast_fac = vol_fac_acum + (media_3m_fac * meses_restantes)
share_forecast = (forecast_fac / forecast_vol * 100) if forecast_vol > 0 else 0
var_share_forecast = share_forecast - share_acum_LY
media_LY = vol_LY / 12
var_media = variacao(media_3m, media_LY)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    html_ctx = f"Mkt YTD: <b>{lb.formatar_br(vol_acum)}</b> | LY: {lb.formatar_br(vol_LY)}<br>Facchini YTD: <b>{lb.formatar_br(vol_fac_acum)}</b>"
    st.markdown(ui.kpi_card(f"Previsão Fechamento {ano_atual}", lb.formatar_br(forecast_vol), var_forecast, f"{var_forecast:+.1%} vs LY", html_ctx, color="dark"), unsafe_allow_html=True)
with col2:
    html_ctx = f"Média LY (12m): <b>{lb.formatar_br(media_LY)}</b><br>Base cálculo: Últimos 3 meses"
    st.markdown(ui.kpi_card("Média Mensal (Mercado)", lb.formatar_br(media_3m), var_media, f"{var_media:+.1%} vs LY", html_ctx, color="blue"), unsafe_allow_html=True)
with col3:
    html_ctx = f"Share YTD: <b>{share_acum:.2f}%</b> (LY: {share_acum_LY:.2f}%)<br>Vol. Facchini Est.: <b>{lb.formatar_br(forecast_fac)}</b>"
    st.markdown(ui.kpi_card(f"Previsão Share {FACCHINI}", f"{share_forecast:.2f}%", var_share_forecast, f"{var_share_forecast:+.2f} p.p. vs LY", html_ctx, is_percent=True, color="red"), unsafe_allow_html=True)

# ============================================================
# TRIMESTRES
# ============================================================
st.write("")
c_t_title, c_t_btn = st.columns([6, 1])
with c_t_title:
    ui.section("Performance por Trimestre")
with c_t_btn:
    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True) 
    st.radio(
        "Visão Trimestre",
        ('cards', 'chart'),
        index=0 if st.session_state['view_quarter'] == 'cards' else 1,
        format_func=lambda x: 'Cards' if x == 'cards' else 'Gráfico',
        key='radio_quarter_view',
        on_change=lambda: set_quarter_view(st.session_state['radio_quarter_view']),
        horizontal=True,
        label_visibility='collapsed'
    )
    st.markdown("</div>", unsafe_allow_html=True)

data_quarter = []
curr_trim = (datetime.now().month - 1) // 3 + 1

for t in [1, 2, 3, 4]:
    meses = {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[t]
    
    # ATUAL
    df_t = df[(df["Ano"]==ano_atual) & (df["Mes"].isin(meses))]
    if tipo_selecionado: df_t = df_t[df_t["Tipo"]==tipo_selecionado]
    has_data = not df_t.empty
    media_val = int(df_t["Qtde"].sum() / df_t["Mes"].nunique()) if has_data else 0
    
    # LY (LAST YEAR)
    df_ly = df[(df["Ano"]==ano_anterior) & (df["Mes"].isin(meses))]
    if tipo_selecionado: df_ly = df_ly[df_ly["Tipo"]==tipo_selecionado]
    media_ly = int(df_ly["Qtde"].sum() / 3) # Média sempre divide por 3
    
    # FACCHINI LY
    vol_fac_ly_total = df_ly[df_ly["Implementadora"]==FACCHINI]["Qtde"].sum()
    vol_ly_total = df_ly["Qtde"].sum()
    share_ly = (vol_fac_ly_total / vol_ly_total * 100) if vol_ly_total > 0 else 0.0
    vol_fac_ly_media = int(vol_fac_ly_total / 3)
    
    d_vol = variacao(media_val, media_ly)
    
    # FACCHINI ATUAL
    share_atual = 0.0
    vol_fac_media = 0
    if has_data and df_t["Qtde"].sum() > 0:
        vol_fac = df_t[df_t["Implementadora"]==FACCHINI]["Qtde"].sum()
        vol_fac_media = int(vol_fac / df_t["Mes"].nunique())
        share_atual = (vol_fac / df_t["Qtde"].sum()) * 100
    
    data_quarter.append({
        "Trimestre": f"{t}º Trim",
        "Vol_Atual": media_val,
        "Vol_LY": media_ly,
        "Delta": d_vol,
        "Share_Facchini_Pct": share_atual,
        "Share_Facchini_LY_Pct": share_ly,
        "Vol_Fac_Media": vol_fac_media,
        "Vol_Fac_LY_Media": vol_fac_ly_media,
        "Has_Data": has_data,
        "Is_Future": (t > curr_trim)
    })

if st.session_state['view_quarter'] == 'chart':
    st.altair_chart(plot_comparativo_trimestral(data_quarter), use_container_width=True)
else:
    cols = st.columns(4, gap="medium")
    for i, d in enumerate(data_quarter):
        share_str = f"{lb.formatar_br(d['Vol_Fac_Media'])} un ({fmt_pct(d['Share_Facchini_Pct'])})" if d['Has_Data'] else "—"
        val_str = lb.formatar_br(d['Vol_Atual']) if d['Has_Data'] else "—"
        with cols[i]:
            st.markdown(ui.quarter_card(
                d['Trimestre'], val_str, d['Delta'], f"{d['Delta']:+.1%} vs LY".replace('.', ','), 
                share_str, lb.formatar_br(d['Vol_LY']), d['Is_Future']
            ), unsafe_allow_html=True)

# ============================================================
# MENSAL
# ============================================================
st.write("")
c_m_title, c_m_btn = st.columns([6, 1])
with c_m_title:
    ui.section("Detalhamento Mensal")
with c_m_btn:
    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    st.radio(
        "Visão Mensal",
        ('cards', 'chart'),
        index=0 if st.session_state['view_month'] == 'cards' else 1,
        format_func=lambda x: 'Cards' if x == 'cards' else 'Gráfico',
        key='radio_month_view',
        on_change=lambda: set_month_view(st.session_state['radio_month_view']),
        horizontal=True,
        label_visibility='collapsed'
    )
    st.markdown("</div>", unsafe_allow_html=True)

data_month = []
meses_nomes = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]

for i in range(12):
    mes = i+1
    val = calcular_total_acumulado(df[(df["Mes"]==mes)], ano_atual, tipo_selecionado)
    val_ly = calcular_total_acumulado(df[(df["Mes"]==mes)], ano_anterior, tipo_selecionado)
    
    has_data = val > 0
    vol_fac = calcular_vol_facchini(df, ano_atual, [mes], tipo_selecionado)
    share_atual = (vol_fac / val * 100) if val > 0 else 0
    
    data_month.append({
        "Mes": meses_nomes[i],
        "Vol_Atual": val,
        "Vol_LY": val_ly,
        "Delta": variacao(val, val_ly),
        "Share_Raw": share_atual,
        "Vol_Fac": vol_fac,
        "Has_Data": has_data
    })

if st.session_state['view_month'] == 'chart':
    data_month_chart = [d for d in data_month if d['Has_Data']]
    st.altair_chart(plot_evolucao_mensal(data_month_chart), use_container_width=True)
else:
    rows = [st.columns(6, gap="small"), st.columns(6, gap="small")]
    for i, d in enumerate(data_month):
        c = rows[0 if i < 6 else 1][i if i < 6 else i - 6]
        share_str = f"{lb.formatar_br(d['Vol_Fac'])} un ({fmt_pct(d['Share_Raw'])})"
        with c:
            st.markdown(ui.month_card(
                d['Mes'], lb.formatar_br(d['Vol_Atual']), d['Delta'], 
                f"{d['Delta']:+.1%} vs LY".replace('.', ','), share_str, d['Has_Data']
            ), unsafe_allow_html=True)