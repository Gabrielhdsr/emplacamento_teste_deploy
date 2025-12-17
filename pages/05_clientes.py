import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import data as dt
import lib as lb
import ui

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="Clientes", page_icon="🎯")
ui.apply_style()

ui.header(
    "Clientes (Onde Atirar)",
    "Macro da carteira → Relevantes (+/−) → Potenciais → Mapa → Playbook do cliente → Histórico"
)

FACCHINI = "FACCHINI"

# Paleta
C_FAC   = "#dc2626"
C_COMP  = "#2563eb"
C_GRAY  = "#9ca3af"
C_GREEN = "#16a34a"
C_AMB   = "#f59e0b"
C_RISK  = "#7c3aed"

# ============================================================
# CSS (acabamento + sticky context)
# ============================================================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
hr { margin: 1.2rem 0; opacity: .25; }

.kpi-wrap{
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(148,163,184,.18);
  border-radius: 16px;
  padding: 14px 14px;
}

.metric-card{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(148,163,184,.16);
  border-radius: 16px;
  padding: 16px 16px;
  min-height: 118px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  box-shadow: 0 8px 18px rgba(0,0,0,.10);
}

.metric-label{
  font-size: .82rem;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: rgba(148,163,184,.95);
}
.metric-value--num{
  font-size: 2.15rem;
  letter-spacing: -0.02em;
  line-height: 1.05;
  white-space: nowrap;
}
.metric-value--title{
  font-size: 1.35rem;
  letter-spacing: -0.01em;
  line-height: 1.15;
  white-space: normal;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.metric-sub{
  font-size: .88rem;
  color: rgba(148,163,184,.90);
  margin-top: 8px;
}

.kpi-accent{
  border-left: 6px solid rgba(148,163,184,.55);
  padding-left: 12px;
}
.kpi-red{ border-left-color: rgba(220,38,38,.95); }
.kpi-blue{ border-left-color: rgba(37,99,235,.95); }
.kpi-green{ border-left-color: rgba(22,163,74,.95); }
.kpi-purple{ border-left-color: rgba(124,58,237,.95); }
.kpi-gray{ border-left-color: rgba(156,163,175,.80); }

.sticky-focus{
  position: sticky;
  top: 0.5rem;
  z-index: 100;
  backdrop-filter: blur(10px);
  background: rgba(15, 23, 42, .55);
  border: 1px solid rgba(148,163,184,.18);
  border-radius: 16px;
  padding: 10px 12px;
  margin-bottom: 14px;
}

.vega-embed summary { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("cli_focus", None)
st.session_state.setdefault("cli_uf_focus", None)
st.session_state.setdefault("cli_mix_focus", None)

# Keys únicas (NUNCA repetir)
K_BEST  = "cli_rank_best"
K_WORST = "cli_rank_worst"
K_OPP   = "cli_rank_opp"
K_WS    = "cli_rank_ws"
K_MAP   = "cli_map_carteira"
K_UF    = "cli_uf_opp"
K_MIX   = "cli_mix_opp"

# ============================================================
# HELPERS
# ============================================================
def _normalize_keys(d: dict) -> dict:
    return {str(k).replace("\\", ""): v for k, v in d.items()}

def selected_value_from_chart_state(chart_key: str, selection_name: str, field_name: str):
    state = st.session_state.get(chart_key)
    if not isinstance(state, dict):
        return None
    sel = state.get("selection", {}).get(selection_name)
    if not sel:
        return None

    def extract(d):
        if not isinstance(d, dict):
            return None
        dn = _normalize_keys(d)
        return dn.get(field_name)

    if isinstance(sel, list) and len(sel) > 0:
        return extract(sel[0])
    if isinstance(sel, dict):
        return extract(sel)
    return None

def is_nan(x) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False

def fmt_int(x) -> str:
    try:
        if pd.isna(x):
            return "0"
        return lb.formatar_br(int(round(float(x))))
    except Exception:
        return "0"

def fmt_pct(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"

def fmt_pp(x_frac) -> str:
    try:
        if pd.isna(x_frac):
            return "—"
        return f"{float(x_frac)*100:+.1f} pp"
    except Exception:
        return "—"

def small_metric(label: str, value: str, sub: str = "", accent="kpi-gray", value_class="metric-value--num"):
    html = f"""
    <div class="metric-card kpi-accent {accent}">
        <div>
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
        </div>
        <div class="metric-sub">{sub}</div>
    </div>
    """
    return " ".join(html.split())

# ============================================================
# LOAD + PADRONIZAÇÃO
# ============================================================
@st.cache_data(show_spinner=False)
def load_df():
    df = dt.carregar_emplacamento("arquivos/Emplacamento/*.xlsx")
    df = df.rename(columns={"Cidade": "Municipio", "Município": "Municipio"})

    required = [
        "Ano", "Tipo", "UF", "Municipio", "Qtde",
        "Implementadora", "Mix Produto", "Cliente",
        "Representante", "Modelo"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = "N/I"

    df["UF"] = df["UF"].astype(str).str.strip().str.upper()
    df["Municipio"] = df["Municipio"].astype(str).str.strip().str.upper()
    df["Implementadora"] = df["Implementadora"].astype(str).str.strip().str.upper()

    for c in ["Mix Produto", "Cliente", "Representante", "Modelo", "Tipo"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
        df.loc[df[c].eq(""), c] = "N/I"

    df["Qtde"] = pd.to_numeric(df["Qtde"], errors="coerce").fillna(0).astype(int)
    df["Ano"]  = pd.to_numeric(df["Ano"], errors="coerce").fillna(0).astype(int)
    return df

df = load_df()

# ============================================================
# FILTROS TOPO
# ============================================================
with st.container():
    f1, f2, f3, f4, f5, f6 = st.columns([1.0, 1.2, 1.6, 2.1, 1.6, 1.1], gap="medium")

    with f1:
        anos = sorted([a for a in df["Ano"].unique() if a > 0], reverse=True)
        ano_sel = st.selectbox("Ano base", anos if anos else [0])

    with f2:
        ano_prev = (ano_sel - 1) if (ano_sel - 1) in anos else None
        comp_opts = [a for a in anos if a != ano_sel]
        idx = 0
        if ano_prev in comp_opts:
            idx = 1 + comp_opts.index(ano_prev)
        ano_comp = st.selectbox("Comparar com", ["—"] + comp_opts, index=idx)

    with f3:
        tipos = sorted([t for t in df["Tipo"].unique() if t not in ("", "N/I")])
        tipo_sel = st.selectbox("Tipo", ["Consolidado"] + tipos)

    df_scope_all = df.copy()
    if tipo_sel != "Consolidado":
        df_scope_all = df_scope_all[df_scope_all["Tipo"].eq(tipo_sel)]

    with f4:
        mix_opts = sorted([m for m in df_scope_all["Mix Produto"].unique() if m not in ("", "N/I")])
        mix_sel = st.multiselect("Mix Produto (opcional)", mix_opts, default=[])

    if mix_sel:
        df_scope_all = df_scope_all[df_scope_all["Mix Produto"].isin(mix_sel)]

    with f5:
        uf_opts = sorted([u for u in df_scope_all["UF"].unique() if u not in ("", "N/I")])
        uf_filter = st.multiselect("UF (opcional)", uf_opts, default=[])

    if uf_filter:
        df_scope_all = df_scope_all[df_scope_all["UF"].isin(uf_filter)]

    with f6:
        ignore_ni = st.radio("Ignorar N/I", ["Não", "Sim"], horizontal=True, index=1)

if ignore_ni == "Sim":
    for col in ["Cliente", "Mix Produto", "Municipio", "Implementadora"]:
        df_scope_all = df_scope_all[~df_scope_all[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

df_scope = df_scope_all[df_scope_all["Ano"].eq(ano_sel)].copy()
if df_scope.empty:
    st.warning("Sem dados no recorte atual.")
    st.stop()

# ============================================================
# MODELO DE CARTEIRA (cliente + YoY + concorrente #1)
# ============================================================
def agg_client_year(df_in: pd.DataFrame) -> pd.DataFrame:
    g = df_in.groupby(["Ano", "Cliente"], as_index=False).agg(Market=("Qtde", "sum"))
    f = (df_in[df_in["Implementadora"].eq(FACCHINI)]
         .groupby(["Ano", "Cliente"], as_index=False).agg(Facchini=("Qtde", "sum")))
    out = g.merge(f, on=["Ano", "Cliente"], how="left")
    out["Facchini"] = out["Facchini"].fillna(0)
    out["Share_Fac"] = (out["Facchini"] / out["Market"]).fillna(0)
    return out

def add_yoy(g: pd.DataFrame, year_base: int, year_comp):
    cur = g[g["Ano"].eq(year_base)].copy()

    # IMPORTANTe: sem ano comparável -> NaN (não inventa YoY)
    if year_comp in (None, "—"):
        cur["Share_Fac_prev"] = np.nan
        cur["dShareFac"] = np.nan
    else:
        prev = g[g["Ano"].eq(int(year_comp))].copy()
        prev = prev.rename(columns={"Share_Fac": "Share_Fac_prev"})[["Cliente", "Share_Fac_prev"]]
        cur = cur.merge(prev, on="Cliente", how="left")
        cur["Share_Fac_prev"] = cur["Share_Fac_prev"].astype(float)
        cur["dShareFac"] = cur["Share_Fac"].astype(float) - cur["Share_Fac_prev"].astype(float)

    return cur[["Cliente", "dShareFac"]]

def top_competitor_per_client(df_year: pd.DataFrame) -> pd.DataFrame:
    comp = (df_year[df_year["Implementadora"].ne(FACCHINI)]
            .groupby(["Cliente", "Implementadora"], as_index=False)
            .agg(Comp1Vol=("Qtde", "sum"))
            .sort_values(["Cliente", "Comp1Vol"], ascending=[True, False]))
    topc = comp.drop_duplicates("Cliente", keep="first").rename(columns={"Implementadora": "Comp1"})
    return topc

def build_clients_model(df_year: pd.DataFrame, yoy_cur: pd.DataFrame) -> pd.DataFrame:
    m = df_year.groupby("Cliente", as_index=False).agg(Market=("Qtde", "sum"))
    f = (df_year[df_year["Implementadora"].eq(FACCHINI)]
         .groupby("Cliente", as_index=False).agg(Facchini=("Qtde", "sum")))
    topc = top_competitor_per_client(df_year)

    out = (m.merge(f, on="Cliente", how="left")
           .merge(topc, on="Cliente", how="left")
           .merge(yoy_cur, on="Cliente", how="left"))

    out["Facchini"] = out["Facchini"].fillna(0)
    out["Comp1"] = out["Comp1"].fillna("—")
    out["Comp1Vol"] = out["Comp1Vol"].fillna(0)
    out["dShareFac"] = out["dShareFac"].astype(float)

    out["Share_Fac"] = (out["Facchini"] / out["Market"]).fillna(0)
    out["Share_Comp1"] = (out["Comp1Vol"] / out["Market"]).fillna(0)

    out["OppTake"] = ((out["Share_Comp1"] - out["Share_Fac"]).clip(lower=0) * out["Market"]).fillna(0)
    out["WhiteSpace"] = (out["Market"] - out["Facchini"]).clip(lower=0)

    # impacto em "qtde equivalente" (só quando há YoY)
    out["ImpactShare"] = np.where(
        pd.isna(out["dShareFac"]),
        np.nan,
        (out["dShareFac"] * out["Market"]).astype(float)
    )

    return out

years_needed = [ano_sel] + ([] if ano_comp == "—" else [int(ano_comp)])
df_years = df_scope_all[df_scope_all["Ano"].isin(years_needed)].copy()

g_cy = agg_client_year(df_years)
yoy_cur = add_yoy(g_cy, ano_sel, ano_comp)

df_clients = build_clients_model(df_scope, yoy_cur)
df_clients = df_clients[df_clients["Market"] > 0].copy()

if df_clients.empty:
    st.warning("Sem clientes com volume no recorte.")
    st.stop()

# ============================================================
# CLUSTER (macro decisão)
# ============================================================
m_med = float(df_clients["Market"].median())
s_med = float(df_clients["Share_Fac"].median())

def classify_cluster(row):
    if row["Market"] >= m_med and row["Share_Fac"] < s_med:
        base = "Atacar"
    elif row["Market"] >= m_med and row["Share_Fac"] >= s_med:
        base = "Defender"
    elif row["Market"] < m_med and row["Share_Fac"] < s_med:
        base = "Cultivar"
    else:
        base = "Manter"

    # Risco só faz sentido se houver YoY
    if base == "Defender" and (not pd.isna(row.get("dShareFac", np.nan))) and float(row.get("dShareFac", 0)) < 0:
        return "Risco"
    return base

df_clients["Cluster"] = df_clients.apply(classify_cluster, axis=1)

# foco default
if st.session_state.get("cli_focus") not in set(df_clients["Cliente"].astype(str)):
    st.session_state["cli_focus"] = str(df_clients.sort_values("Market", ascending=False).iloc[0]["Cliente"])
    st.session_state["cli_uf_focus"] = None
    st.session_state["cli_mix_focus"] = None

# ============================================================
# MACRO KPIs
# ============================================================
market_total = int(df_scope["Qtde"].sum())
fac_total = int(df_scope.loc[df_scope["Implementadora"].eq(FACCHINI), "Qtde"].sum())
fac_share = (fac_total / market_total) if market_total > 0 else 0

if ano_comp != "—":
    df_prev = df_scope_all[df_scope_all["Ano"].eq(int(ano_comp))]
    m_prev = int(df_prev["Qtde"].sum())
    f_prev = int(df_prev.loc[df_prev["Implementadora"].eq(FACCHINI), "Qtde"].sum())
    s_prev = (f_prev / m_prev) if m_prev > 0 else 0
    d_share_total = fac_share - s_prev
else:
    d_share_total = np.nan

opp_total = float(df_clients["OppTake"].sum())
white_total = float(df_clients["WhiteSpace"].sum())

top10 = df_clients.sort_values("Market", ascending=False).head(10)
conc_top10 = float(top10["Market"].sum() / df_clients["Market"].sum()) if df_clients["Market"].sum() > 0 else 0

n_attack = int((df_clients["Cluster"] == "Atacar").sum())
n_risk  = int((df_clients["Cluster"] == "Risco").sum())

# ============================================================
# CHART HELPERS
# ============================================================
def bar_rank(df_in: pd.DataFrame, dim: str, val: str, sel, title: str, color: str, fmt: str, height: int):
    dfp = df_in.copy()

    if dfp.empty:
        return alt.Chart(pd.DataFrame({dim: [], val: []})).mark_bar().properties(height=height, title=title)

    # 1) Formata o valor
    if fmt == "pp":
        dfp["_lab"] = dfp[val].apply(lambda x: "—" if pd.isna(x) else f"{float(x):+.1f} pp")
        x_format = None
    elif fmt == "pct":
        dfp["_lab"] = dfp[val].apply(lambda x: "—" if pd.isna(x) else f"{float(x)*100:.1f}%")
        x_format = "%"
    else:
        dfp["_lab"] = dfp[val].apply(fmt_int)
        x_format = ",.0f"

    # 2) Nome com reticências (mais elegante que cortar seco)
    def _ellipsis(s: str, n: int = 28) -> str:
        s = "" if s is None else str(s)
        return (s[: n - 1] + "…") if len(s) > n else s

    dfp["_dim_short"] = dfp[dim].apply(_ellipsis)

    # 3) Rótulo único (1 camada) com separador clean
    # espaços finos: \u2009
    dfp["__y"] = dfp["_dim_short"].astype(str) + "\u2009\u2009·\u2009" + dfp["_lab"].astype(str)

    axis_right = alt.Axis(
        orient="right",
        title=None,
        labelFont="Segoe UI, Roboto, Arial, sans-serif",
        labelFontSize=12,
        labelFontWeight=500,     # leve (não tão escuro)
        labelColor="#3a3a3a",    # só um pouco mais escuro
        ticks=False,
        domain=False,
        labelPadding=10,
        labelLimit=320,
        grid=False
    )

    return (
        alt.Chart(dfp)
        .mark_bar(color=color, cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{val}:Q", title=None, axis=None),
            y=alt.Y("__y:N", sort="-x", axis=axis_right),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
            tooltip=[
                alt.Tooltip(f"{dim}:N", title=dim),
                alt.Tooltip(f"{val}:Q", title="Valor", format=x_format if x_format else ""),
                alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
                alt.Tooltip("Share_Fac:Q", title="Share FAC", format=".1%"),
            ],
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )

def scatter_carteira(df_in: pd.DataFrame, sel):
    color_scale = alt.Scale(
        domain=["Atacar", "Risco", "Defender", "Cultivar", "Manter"],
        range=[C_COMP, C_RISK, C_GREEN, C_AMB, C_GRAY],
    )
    return (
        alt.Chart(df_in)
        .mark_circle()
        .encode(
            x=alt.X("Market:Q", title="Tamanho do cliente (Qtde)", scale=alt.Scale(type="log")),
            y=alt.Y("Share_Fac:Q", title="Share FACCHINI", axis=alt.Axis(format="%")),
            size=alt.Size("Market:Q", legend=None),
            color=alt.Color("Cluster:N", scale=color_scale, legend=alt.Legend(title="Macro decisão")),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("Cliente:N"),
                alt.Tooltip("Cluster:N"),
                alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
                alt.Tooltip("Facchini:Q", title="FACCHINI", format=",.0f"),
                alt.Tooltip("Share_Fac:Q", title="Share FAC", format=".1%"),
                alt.Tooltip("dShareFac:Q", title="Δ Share YoY", format="+.1%"),
                alt.Tooltip("Comp1:N", title="Conc. #1"),
                alt.Tooltip("OppTake:Q", title="Opp (tomar do líder)", format=",.0f"),
                alt.Tooltip("WhiteSpace:Q", title="WhiteSpace", format=",.0f"),
            ],
        )
        .add_params(sel)
        .properties(height=520, title="Mapa da carteira (clique em um cliente para focar)")
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )

# ============================================================
# 1) VISÃO MACRO: LISTA COMPLETA (SEM TEXTO LATERAL PARA HABILITAR CLIQUE)
# ============================================================
ui.section("🏆 1) Ranking de Clientes (Role para ver todos, Clique para filtrar)")

# 1. PREPARAÇÃO DOS DADOS (Base completa ordenada)
df_all_sorted = df_clients.sort_values("Market", ascending=False).copy()

# KPIs DA BASE INTEIRA
all_vol = df_all_sorted["Market"].sum()
all_fac = df_all_sorted["Facchini"].sum()
all_share = (all_fac / all_vol) if all_vol > 0 else 0
all_ws = df_all_sorted["WhiteSpace"].sum()

# Cálculo da altura dinâmica
n_clientes = len(df_all_sorted)
altura_por_barra = 30 
altura_total_grafico = max(500, n_clientes * altura_por_barra)

# 2. CARD ROBUSTO
def card_robust_dark(label, value, sub, color_border="#94a3b8"):
    return f"""
    <div style="
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid {color_border};
        border-radius: 8px;
        padding: 15px 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        display: flex; flex-direction: column; gap: 4px;
    ">
        <div style="font-size: 0.7rem; text-transform: uppercase; font-weight: 700; color: #64748b; letter-spacing: 0.05em;">{label}</div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #1e293b; line-height: 1.0;">{value}</div>
        <div style="font-size: 0.85rem; color: #334155; font-weight: 500;">{sub}</div>
    </div>
    """

# 3. LAYOUT
c_chart_scroll, c_kpi_fixed = st.columns([3.8, 1], gap="small")

with c_chart_scroll:
    st.markdown("##### 📊 Ranking Completo (Role a lista 👇)")
    
    sel_top = alt.selection_point(fields=["Cliente"], name="SEL_TOP_MKT", clear="dblclick")
    nova_paleta = ["#ff7777", "#fd2727", "#ff0000", "#be0000", "#990000"]

    # GRÁFICO DE CAMADA ÚNICA (BARRAS) - Essencial para o clique funcionar
    ch_full = alt.Chart(df_all_sorted).mark_bar(
        cornerRadiusEnd=4,
        height=22
    ).encode(
        x=alt.X("Market:Q", title=None, axis=None), # Sem eixo X
        y=alt.Y("Cliente:N", sort="-x", axis=alt.Axis(
            title=None, 
            labelLimit=250, 
            labelFontSize=12, 
            labelFontWeight=600, 
            labelColor="#334155",
            grid=False # Sem linhas de grade
        )),
        color=alt.condition(
            alt.datum.Share_Fac == 0,
            alt.value("#cbd5e1"),
            alt.Color("Share_Fac:Q", 
                      scale=alt.Scale(domain=[0, 0.25, 0.5, 0.75, 1], range=nova_paleta), 
                      legend=None)
        ),
        opacity=alt.condition(sel_top, alt.value(1), alt.value(0.4)),
        tooltip=[
            alt.Tooltip("Cliente"), 
            alt.Tooltip("Market", title="Mercado Total", format=",.0f"),
            alt.Tooltip("Facchini", title="Vol. Facchini", format=",.0f"),
            alt.Tooltip("Share_Fac", title="Share Atual", format=".1%")
        ]
    ).add_params(sel_top).properties(
        height=altura_total_grafico
    ).configure_view(
        stroke=None 
    )

    with st.container(height=520, border=False):
        # Agora funciona o on_select pois é um gráfico simples (só barras), sem camadas extras
        st.altair_chart(ch_full, use_container_width=True, on_select="rerun", key="chart_top_mkt_full")

with c_kpi_fixed:
    st.markdown("##### 🔎 Raio-X (Total)")
    
    st.markdown(card_robust_dark("Potencial Total", fmt_int(all_vol), "Soma da Carteira", "#94a3b8"), unsafe_allow_html=True)
    st.markdown(card_robust_dark("Carteira Facchini", fmt_int(all_fac), f"Share Global: <b>{all_share:.1%}</b>", "#ef4444"), unsafe_allow_html=True)
    st.markdown(card_robust_dark("Dinheiro na Mesa", fmt_int(all_ws), "Oportunidade Total", "#3b82f6"), unsafe_allow_html=True)
    
    st.caption("🎨 **Intensidade:**")
    legenda_html = f"""
    <div style="font-size: 0.75rem; color: #475569; line-height: 1.5; display: flex; flex-direction: column; gap: 3px;">
        <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; background:#cbd5e1; border-radius:2px;"></div> 0%</div>
        <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; background:{nova_paleta[0]}; border-radius:2px;"></div> < 25%</div>
        <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; background:{nova_paleta[2]}; border-radius:2px;"></div> ~ 50%</div>
        <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; background:{nova_paleta[4]}; border-radius:2px;"></div> 100%</div>
    </div>
    """
    st.markdown(legenda_html, unsafe_allow_html=True)

    click_top = selected_value_from_chart_state("chart_top_mkt_full", "SEL_TOP_MKT", "Cliente")
    if click_top:
         st.session_state["cli_focus"] = str(click_top)
         st.session_state["cli_uf_focus"] = None
         st.session_state["cli_mix_focus"] = None

st.markdown("<hr style='margin: 10px 0; opacity: 0.1;'>", unsafe_allow_html=True)

# ============================================================
# 2) TRIPÉ TÁTICO: ONDE AGIR
# ============================================================
ui.section("📊 2) Tripé Tático (Filtre por Oportunidade)")

# Prepara os DataFrames
df_dom = df_clients[df_clients["Share_Fac"] >= 0.50].sort_values("Facchini", ascending=False).head(10)
df_ws  = df_clients.sort_values("WhiteSpace", ascending=False).head(10)

# Tratamento para Perda (precisa ter a coluna ImpactShare)
if "ImpactShare" in df_clients.columns:
    # Filtra negativos significativos (< -0.5 para evitar ruído de zero)
    df_loss = df_clients[df_clients["ImpactShare"] < -0.5].sort_values("ImpactShare", ascending=True).head(10)
    # Truque visual: converte para positivo para o gráfico, mas mantemos a cor de alerta
    df_loss["AbsLoss"] = df_loss["ImpactShare"].abs()
else:
    df_loss = pd.DataFrame()

c_dom, c_conc, c_loss = st.columns(3, gap="large")

# --- 1. DOMÍNIO (Manter) ---
with c_dom:
    vol_dom = int(df_dom["Facchini"].sum())
    st.markdown(f"##### 🛡️ Fortaleza ({len(df_dom)})")
    st.caption(f"Clientes com Share > 50%. Vol: **{fmt_int(vol_dom)}**")
    
    if not df_dom.empty:
        sel_dom = alt.selection_point(fields=["Cliente"], name="SEL_DOM", clear="dblclick")
        # Note que passamos sel_dom para o helper bar_rank
        ch_dom = bar_rank(df_dom, "Cliente", "Facchini", sel_dom, "", C_GREEN, "int", 350)
        
        st.altair_chart(ch_dom, use_container_width=True, on_select="rerun", key="chart_dom")
        
        # Captura clique
        click_dom = selected_value_from_chart_state("chart_dom", "SEL_DOM", "Cliente")
        if click_dom:
             st.session_state["cli_focus"] = str(click_dom)
    else:
        st.info("Nenhum cliente dominado neste recorte.")

# --- 2. ATAQUE (Conquistar) ---
with c_conc:
    vol_ws = int(df_ws["WhiteSpace"].sum())
    st.markdown(f"##### ⚔️ Ataque ({len(df_ws)})")
    st.caption(f"Maior espaço em branco. Potencial: **{fmt_int(vol_ws)}**")
    
    if not df_ws.empty:
        sel_ws = alt.selection_point(fields=["Cliente"], name="SEL_WS", clear="dblclick")
        # Usamos WhiteSpace como métrica
        ch_ws = bar_rank(df_ws, "Cliente", "WhiteSpace", sel_ws, "", C_AMB, "int", 350) 
        
        st.altair_chart(ch_ws, use_container_width=True, on_select="rerun", key="chart_ws")
        
        # Captura clique
        click_ws = selected_value_from_chart_state("chart_ws", "SEL_WS", "Cliente")
        if click_ws:
             st.session_state["cli_focus"] = str(click_ws)
    else:
        st.info("Sem oportunidades claras de ataque.")

# --- 3. PERDA (Risco) ---
with c_loss:
    # Soma das perdas (que estão negativas)
    loss_val = df_loss["ImpactShare"].sum() if not df_loss.empty else 0
    st.markdown(f"##### 🚨 Sangramento ({len(df_loss)})")
    st.caption(f"Perda de volume vs Ano anterior: **{fmt_int(loss_val)}**")
    
    if not df_loss.empty:
        sel_loss = alt.selection_point(fields=["Cliente"], name="SEL_LOSS", clear="dblclick")
        # Usamos AbsLoss para a barra crescer pra direita, mas cor de RISCO
        ch_loss = bar_rank(df_loss, "Cliente", "AbsLoss", sel_loss, "", C_RISK, "int", 350)
        
        st.altair_chart(ch_loss, use_container_width=True, on_select="rerun", key="chart_loss")
        
        # Captura clique
        click_loss = selected_value_from_chart_state("chart_loss", "SEL_LOSS", "Cliente")
        if click_loss:
             st.session_state["cli_focus"] = str(click_loss)
    else:
        if ano_comp == "—":
            st.warning("Selecione um ano comparativo no topo.")
        else:
            st.success("Nenhuma perda relevante de share.")

st.write("---")

