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
        labelLimit=320
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
# 1) VISÃO MACRO
# ============================================================
ui.section("🏆 1) Top Clientes do Mercado (Quem movimenta o ponteiro)")

# Filtra os Top 15 Clientes por Volume Total de Mercado
df_top_market = df_clients.sort_values("Market", ascending=False).head(15).copy()

# KPIs laterais baseados nesse Top 15
top_vol = df_top_market["Market"].sum()
top_fac = df_top_market["Facchini"].sum()
top_share = (top_fac / top_vol) if top_vol > 0 else 0
top_ws = df_top_market["WhiteSpace"].sum()

c_top_chart, c_top_kpi = st.columns([2.5, 1], gap="large")

with c_top_chart:
    st.markdown("##### Ranking por Volume Total de Compras")
    sel_top = alt.selection_point(fields=["Cliente"], name="SEL_TOP_MKT", clear="dblclick")
    
    # Gráfico de Barras Horizontais (Mercado)
    # Usamos o helper bar_rank mas customizamos para mostrar mercado
    ch_top = bar_rank(
        df_top_market, 
        "Cliente", 
        "Market", 
        sel_top, 
        "", 
        C_RISK, 
        "int", 
        500
    )
    st.altair_chart(ch_top, use_container_width=True, on_select="rerun", key="chart_top_mkt")
    
    # Lógica de clique
    click_top = selected_value_from_chart_state("chart_top_mkt", "SEL_TOP_MKT", "Cliente")
    if click_top:
         st.session_state.update({"cli_focus": str(click_top), "cli_uf_focus": None, "cli_mix_focus": None})

with c_top_kpi:
    st.markdown("##### Raio-X do Top 15")
    st.markdown(small_metric("Volume Total", fmt_int(top_vol), "Soma do Top 15", "kpi-gray"), unsafe_allow_html=True)
    st.write("")
    st.markdown(small_metric("Vol. FACCHINI", fmt_int(top_fac), f"Share no Top 15: {top_share:.1%}", "kpi-red"), unsafe_allow_html=True)
    st.write("")
    st.markdown(small_metric("Espaço (WhiteSpace)", fmt_int(top_ws), "Volume na concorrência", "kpi-blue"), unsafe_allow_html=True)
    st.write("")
    
    # Mini lista dos nomes
    st.caption("**Principais nomes:**")
    st.markdown(
        f"<div style='font-size:0.85em; color:#64748b; line-height:1.4;'>{', '.join(df_top_market['Cliente'].head(5).tolist())}...</div>", 
        unsafe_allow_html=True
    )

st.write("---")

# ============================================================
# 2) TRIPÉ DE ANÁLISE: DOMÍNIO, CONCORRÊNCIA E PERDA
# ============================================================
ui.section("📊 2) Análise de Posicionamento e Risco")

col_dom, col_conc, col_loss = st.columns(3, gap="large")

# --- COLUNA 1: DOMÍNIO FACCHINI (Share > 50%) ---
with col_dom:
    st.markdown("##### 🛡️ Fortaleza FACCHINI")
    st.caption("Maiores clientes onde **somos líderes** (>50% Share).")
    
    # Filtra Share > 50% e ordena por Volume FACCHINI (para pegar os grandes parceiros)
    df_dom = df_clients[df_clients["Share_Fac"] >= 0.50].sort_values("Facchini", ascending=False).head(10)
    
    if df_dom.empty:
        st.info("Nenhum cliente com share > 50% no recorte.")
    else:
        sel_dom = alt.selection_point(fields=["Cliente"], name="SEL_DOM", clear="dblclick")
        ch_dom = bar_rank(df_dom, "Cliente", "Facchini", sel_dom, "Volume FACCHINI (Share > 50%)", C_FAC, "int", 400)
        st.altair_chart(ch_dom, use_container_width=True, on_select="rerun", key="chart_dom")
        
        click_dom = selected_value_from_chart_state("chart_dom", "SEL_DOM", "Cliente")
        if click_dom:
             st.session_state.update({"cli_focus": str(click_dom), "cli_uf_focus": None, "cli_mix_focus": None})

# --- COLUNA 2: DOMÍNIO CONCORRÊNCIA (Maior WhiteSpace) ---
with col_conc:
    st.markdown("##### ⚔️ Terreno da Concorrência")
    st.caption("Onde a **concorrência vende mais** (Maior WhiteSpace).")
    
    # Ordena por WhiteSpace (Mercado - Facchini)
    df_ws = df_clients.sort_values("WhiteSpace", ascending=False).head(10)
    
    sel_ws = alt.selection_point(fields=["Cliente"], name="SEL_WS", clear="dblclick")
    ch_ws = bar_rank(df_ws, "Cliente", "WhiteSpace", sel_ws, "Volume da Concorrência", C_COMP, "int", 400)
    st.altair_chart(ch_ws, use_container_width=True, on_select="rerun", key="chart_ws")
    
    click_ws = selected_value_from_chart_state("chart_ws", "SEL_WS", "Cliente")
    if click_ws:
         st.session_state.update({"cli_focus": str(click_ws), "cli_uf_focus": None, "cli_mix_focus": None})

# --- COLUNA 3: MAIOR PERDA DE VOLUME (ImpactShare Negativo) ---
with col_loss:
    st.markdown("##### 🚨 Alerta de Perda")
    st.caption("Onde mais **perdemos volume** (vs Ano Anterior).")
    
    # Verifica se temos dados de YoY
    if "ImpactShare" in df_clients.columns and df_clients["ImpactShare"].notna().any():
        # Filtra apenas quem perdeu (negativo) e ordena pelo mais negativo (ascending=True)
        df_loss = df_clients[df_clients["ImpactShare"] < 0].sort_values("ImpactShare", ascending=True).head(10)
        
        # Transformamos em positivo apenas para o gráfico ficar visualmente compreensível (barra de tamanho de perda)
        # Ou mantemos negativo para mostrar retração. Vamos manter negativo e pintar de roxo/risco.
        
        if df_loss.empty:
            st.success("Sem perdas relevantes de share/volume neste recorte.")
        else:
            sel_loss = alt.selection_point(fields=["Cliente"], name="SEL_LOSS", clear="dblclick")
            # Usamos ImpactShare que é (DeltaShare * Mercado), uma aproximação fiel da perda de volume por performance
            ch_loss = bar_rank(df_loss, "Cliente", "ImpactShare", sel_loss, "Volume Perdido (Estimado)", C_RISK, "int", 400)
            st.altair_chart(ch_loss, use_container_width=True, on_select="rerun", key="chart_loss")
            
            click_loss = selected_value_from_chart_state("chart_loss", "SEL_LOSS", "Cliente")
            if click_loss:
                 st.session_state.update({"cli_focus": str(click_loss), "cli_uf_focus": None, "cli_mix_focus": None})
    else:
        st.warning("Selecione um 'Ano Comparativo' no topo para ver perdas.")

st.write("---")

# ============================================================
# 2) RELEVANTES (+/−) + POTENCIAIS
# ============================================================
ui.section("⭐ 2) Relevância (+/−) e 🎯 Potenciais")

TOP_RELEVANTES = 300  # ajuste aqui
df_rel = df_clients.sort_values("Market", ascending=False).head(TOP_RELEVANTES).copy()

rep_market = float(df_rel["Market"].sum())
rep_share  = (rep_market / float(df_clients["Market"].sum())) if float(df_clients["Market"].sum()) > 0 else 0

# Card de cobertura (clientes + mercado representado)
c_cov1, c_cov2, c_cov3 = st.columns([2.3, 1.2, 1.2], gap="medium")
with c_cov1:
    st.caption(f"Rankings e análises usando os {len(df_rel)} clientes com maior Mercado (Qtde).")
with c_cov2:
    st.markdown(
        small_metric("Clientes analisados", fmt_int(len(df_rel)), f"de {fmt_int(len(df_clients))} no recorte", "kpi-gray"),
        unsafe_allow_html=True
    )
with c_cov3:
    st.markdown(
        small_metric("Mercado representado", fmt_int(rep_market), f"{fmt_pct(rep_share)} do mercado do recorte", "kpi-gray"),
        unsafe_allow_html=True
    )

TOP = 12
colA, colB, colC, colD = st.columns(4, gap="large")

# Se não tem YoY, troca "melhores/piores" por share atual
has_yoy = (ano_comp != "—") and df_rel["ImpactShare"].notna().any()

with colA:
    if has_yoy:
        st.markdown("##### ✅ Melhores (ganho relevante)")
        df_best = df_rel.sort_values("ImpactShare", ascending=False).head(TOP)
        sel_best = alt.selection_point(fields=["Cliente"], name="SEL_BEST", clear="dblclick")
        ch = bar_rank(df_best, "Cliente", "ImpactShare", sel_best, "Maior ganho (Qtde) via share", C_GREEN, "int", 420)
        with st.container(height=480, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_BEST)
        click = selected_value_from_chart_state(K_BEST, "SEL_BEST", "Cliente")
    else:
        st.markdown("##### ✅ Maior Share FACCHINI")
        df_best = df_rel.sort_values("Share_Fac", ascending=False).head(TOP)
        sel_best = alt.selection_point(fields=["Cliente"], name="SEL_BEST", clear="dblclick")
        ch = bar_rank(df_best, "Cliente", "Share_Fac", sel_best, "Maior Share FAC (YoY indisponível)", C_GREEN, "pct", 420)
        with st.container(height=480, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_BEST)
        click = selected_value_from_chart_state(K_BEST, "SEL_BEST", "Cliente")

    if click:
        st.session_state.update({"cli_focus": str(click), "cli_uf_focus": None, "cli_mix_focus": None})

with colB:
    if has_yoy:
        st.markdown("##### ⚠️ Piores (perda relevante)")
        df_worst = df_rel.sort_values("ImpactShare", ascending=True).head(TOP)
        sel_worst = alt.selection_point(fields=["Cliente"], name="SEL_WORST", clear="dblclick")
        ch = bar_rank(df_worst, "Cliente", "ImpactShare", sel_worst, "Maior perda (Qtde) via share", C_FAC, "int", 420)
        with st.container(height=480, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_WORST)
        click = selected_value_from_chart_state(K_WORST, "SEL_WORST", "Cliente")
    else:
        st.markdown("##### ⚠️ Menor Share FACCHINI")
        df_worst = df_rel.sort_values("Share_Fac", ascending=True).head(TOP)
        sel_worst = alt.selection_point(fields=["Cliente"], name="SEL_WORST", clear="dblclick")
        ch = bar_rank(df_worst, "Cliente", "Share_Fac", sel_worst, "Menor Share FAC (YoY indisponível)", C_FAC, "pct", 420)
        with st.container(height=480, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_WORST)
        click = selected_value_from_chart_state(K_WORST, "SEL_WORST", "Cliente")

    if click:
        st.session_state.update({"cli_focus": str(click), "cli_uf_focus": None, "cli_mix_focus": None})

with colC:
    st.markdown("##### 🎯 Maior Opp (tomar do líder)")
    df_opp = df_rel.sort_values("OppTake", ascending=False).head(TOP)
    sel_opp = alt.selection_point(fields=["Cliente"], name="SEL_OPP", clear="dblclick")
    ch = bar_rank(df_opp, "Cliente", "OppTake", sel_opp, "Oportunidade (Qtde)", C_COMP, "int", 420)
    with st.container(height=480, border=False):
        st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_OPP)

    click = selected_value_from_chart_state(K_OPP, "SEL_OPP", "Cliente")
    if click:
        st.session_state.update({"cli_focus": str(click), "cli_uf_focus": None, "cli_mix_focus": None})

with colD:
    st.markdown("##### 🧱 Maior WhiteSpace")
    df_ws = df_rel.sort_values("WhiteSpace", ascending=False).head(TOP)
    sel_ws = alt.selection_point(fields=["Cliente"], name="SEL_WS", clear="dblclick")
    ch = bar_rank(df_ws, "Cliente", "WhiteSpace", sel_ws, "Espaço total", C_GRAY, "int", 420)
    with st.container(height=480, border=False):
        st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_WS)

    click = selected_value_from_chart_state(K_WS, "SEL_WS", "Cliente")
    if click:
        st.session_state.update({"cli_focus": str(click), "cli_uf_focus": None, "cli_mix_focus": None})

st.write("---")

# ============================================================
# 3) MAPA DA CARTEIRA
# ============================================================
ui.section("🗺️ 3) Mapa da carteira (macro decisão)")

df_map = df_clients.sort_values("Market", ascending=False).head(700).copy()
sel_map = alt.selection_point(fields=["Cliente"], name="SEL_MAP", clear="dblclick")
ch_map = scatter_carteira(df_map, sel_map)

with st.container(height=560, border=False):
    st.altair_chart(ch_map, use_container_width=True, on_select="rerun", key=K_MAP)

clicked = selected_value_from_chart_state(K_MAP, "SEL_MAP", "Cliente")
if clicked:
    st.session_state.update({"cli_focus": str(clicked), "cli_uf_focus": None, "cli_mix_focus": None})

st.caption(
    "Definições: **Atacar** (grande + share baixo), **Defender** (grande + share alto), "
    "**Risco** (defender, mas perdendo share), **Cultivar** (pequeno + share baixo), **Manter** (pequeno + share ok)."
)

st.write("---")

# ============================================================
# 4) SELEÇÃO EXPLÍCITA
# ============================================================
ui.section("🎯 4) Selecionar cliente para aprofundar")

clientes_sorted = df_clients.sort_values("Market", ascending=False)["Cliente"].astype(str).tolist()
focus_default = st.session_state.get("cli_focus") if st.session_state.get("cli_focus") in clientes_sorted else clientes_sorted[0]

csel1, csel2, csel3 = st.columns([2.3, 1.0, 1.0], gap="medium")
with csel1:
    picked = st.selectbox("Cliente em foco", clientes_sorted, index=clientes_sorted.index(focus_default))
    if picked != st.session_state.get("cli_focus"):
        st.session_state.update({"cli_focus": str(picked), "cli_uf_focus": None, "cli_mix_focus": None})

with csel2:
    if st.button("Limpar UF/Mix", use_container_width=True):
        st.session_state["cli_uf_focus"] = None
        st.session_state["cli_mix_focus"] = None
        st.rerun()

with csel3:
    if st.button("Reset foco", use_container_width=True):
        st.session_state.update({"cli_focus": clientes_sorted[0], "cli_uf_focus": None, "cli_mix_focus": None})
        st.rerun()

# ============================================================
# Barra sticky de contexto do foco (melhora muito navegação)
# ============================================================
focus_client = st.session_state.get("cli_focus")
row_cli = df_clients[df_clients["Cliente"].astype(str).eq(str(focus_client))].iloc[0]

uf_focus = st.session_state.get("cli_uf_focus") or "Todas"
mix_focus = st.session_state.get("cli_mix_focus") or "Todos"

st.markdown(f"""
<div class="sticky-focus">
  <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:center;">
    <div style="color: rgba(148,163,184,.95); font-size:.85rem; letter-spacing:.04em; text-transform:uppercase;">Foco</div>
    <div style="font-weight:700;">{focus_client}</div>
    <div style="opacity:.75;">|</div>
    <div><span style="opacity:.75;">UF:</span> <b>{uf_focus}</b></div>
    <div><span style="opacity:.75;">Mix:</span> <b>{mix_focus}</b></div>
    <div style="opacity:.75;">|</div>
    <div><span style="opacity:.75;">Macro:</span> <b>{row_cli["Cluster"]}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# ============================================================
# Helpers: líder do slice (UF/Mix) vs FAC
# ============================================================
def calc_vs_fac_leader_dim(df_base: pd.DataFrame, dim: str) -> pd.DataFrame:
    """
    Para cada categoria do dim:
      - Market
      - Facchini
      - Leader (implementadora líder do slice, excluindo FACCHINI)
      - OppVol = max(share_leader - share_fac, 0) * market
    """
    if df_base.empty:
        return pd.DataFrame(columns=[
            dim, "Market", "Facchini", "Leader", "LeaderVol",
            "Share_Fac", "Share_Leader", "Gap_pp", "OppVol", "Sign"
        ])

    m = df_base.groupby(dim, as_index=False).agg(Market=("Qtde", "sum"))
    f = (df_base[df_base["Implementadora"].eq(FACCHINI)]
         .groupby(dim, as_index=False).agg(Facchini=("Qtde", "sum")))

    # líder por slice (exclui FACCHINI)
    comp = (df_base[df_base["Implementadora"].ne(FACCHINI)]
            .groupby([dim, "Implementadora"], as_index=False)
            .agg(LeaderVol=("Qtde", "sum"))
            .sort_values([dim, "LeaderVol"], ascending=[True, False]))

    topc = comp.drop_duplicates(dim, keep="first").rename(columns={"Implementadora": "Leader"})

    out = (m.merge(f, on=dim, how="left")
             .merge(topc, on=dim, how="left"))

    out["Facchini"] = out["Facchini"].fillna(0)
    out["Leader"] = out["Leader"].fillna("—")
    out["LeaderVol"] = out["LeaderVol"].fillna(0)

    out["Share_Fac"] = (out["Facchini"] / out["Market"]).fillna(0)
    out["Share_Leader"] = (out["LeaderVol"] / out["Market"]).fillna(0)

    # se não existe líder (Leader == "—"), share_leader = 0
    out.loc[out["Leader"].eq("—"), "Share_Leader"] = 0.0

    out["Gap_pp"] = (out["Share_Leader"] - out["Share_Fac"]) * 100.0
    out["OppVol"] = ((out["Share_Leader"] - out["Share_Fac"]).clip(lower=0) * out["Market"]).fillna(0)

    # quem está na frente no slice
    out["Sign"] = np.where(out["Share_Fac"] >= out["Share_Leader"], "FAC", "LEADER")

    return out

def leader_in_slice(df_slice: pd.DataFrame) -> str:
    if df_slice.empty:
        return "—"
    tmp = (df_slice[df_slice["Implementadora"].ne(FACCHINI)]
           .groupby("Implementadora", as_index=False)["Qtde"].sum()
           .sort_values("Qtde", ascending=False))
    if tmp.empty:
        return "—"
    return str(tmp.iloc[0]["Implementadora"])

# ============================================================
# 5/6/7) PLAYBOOK + HISTÓRICO + RAIO-X em TABS (menos scroll)
# ============================================================
ui.section("🔎 5) Playbook do cliente • 📈 Histórico • 📂 Raio-X")

tab_play, tab_hist, tab_rx = st.tabs(["🔎 Playbook", "📈 Histórico", "📂 Raio-X"])

# Base do cliente
df_c_all = df_scope[df_scope["Cliente"].astype(str).eq(str(focus_client))].copy()

# aplica filtros de foco (UF/Mix) quando existirem
def apply_focus_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    if st.session_state.get("cli_uf_focus"):
        df_out = df_out[df_out["UF"].eq(st.session_state["cli_uf_focus"])]
    if st.session_state.get("cli_mix_focus"):
        df_out = df_out[df_out["Mix Produto"].eq(st.session_state["cli_mix_focus"])]
    return df_out

with tab_play:
    if df_c_all.empty:
        st.info("Sem dados para o cliente no recorte atual.")
        st.stop()

    # Concorrente #1 "macro" (do modelo)
    comp1_macro = str(row_cli["Comp1"])

    df_c = df_c_all.copy()
    market_c = int(df_c["Qtde"].sum())
    fac_c = int(df_c.loc[df_c["Implementadora"].eq(FACCHINI), "Qtde"].sum())
    share_fac_c = (fac_c / market_c) if market_c > 0 else 0

    comp_c = int(df_c.loc[df_c["Implementadora"].eq(comp1_macro), "Qtde"].sum()) if comp1_macro != "—" else 0
    share_comp_c = (comp_c / market_c) if market_c > 0 else 0

    # líder no recorte atual (considerando UF/Mix foco)
    df_slice = apply_focus_filters(df_c_all)
    leader_slice = leader_in_slice(df_slice)

    k1, k2, k3, k4, k5 = st.columns([1.6, 1, 1, 1, 1], gap="large")
    with k1:
        st.markdown(
            small_metric("Cliente", str(focus_client), f"Macro: {row_cli['Cluster']}", "kpi-gray", value_class="metric-value--title"),
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(small_metric("Mercado", fmt_int(market_c), "Qtde no recorte", "kpi-gray"), unsafe_allow_html=True)
    with k3:
        sub = f"Share: {share_fac_c*100:.1f}%"
        sub += f" • ΔYoY: {fmt_pp(row_cli['dShareFac'])}"
        st.markdown(small_metric("FACCHINI", fmt_int(fac_c), sub, "kpi-red"), unsafe_allow_html=True)
    with k4:
        st.markdown(
            small_metric("Conc. #1 (macro)", comp1_macro, f"Share: {share_comp_c*100:.1f}%", "kpi-blue", value_class="metric-value--title"),
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            small_metric("Líder no recorte (UF/Mix)", leader_slice, "dinâmico por slice", "kpi-blue", value_class="metric-value--title"),
            unsafe_allow_html=True
        )

    st.write("")

    cA, cB, cC = st.columns([1, 1, 1], gap="large")

    with cA:
        st.markdown("##### 📍 Onde (UF) — Opp (Qtde) vs líder do UF")
        df_uf = calc_vs_fac_leader_dim(df_c_all, "UF")
        df_uf = df_uf[df_uf["Market"] > 0].sort_values("OppVol", ascending=False).head(30).copy()

        sel_uf = alt.selection_point(fields=["UF"], name="SEL_UF", clear="dblclick")

        dfp = df_uf.copy()
        dfp["_lab"] = dfp["OppVol"].apply(fmt_int)
        dfp["__y"] = dfp["UF"].astype(str) + " · " + dfp["_lab"].astype(str)
        axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)

        ch = (
            alt.Chart(dfp).mark_bar()
            .encode(
                x=alt.X("OppVol:Q", title=None),
                y=alt.Y("__y:N", sort="-x", axis=axis_right),
                color=alt.Color("Sign:N", scale=alt.Scale(domain=["FAC", "LEADER"], range=[C_FAC, C_COMP]), legend=None),
                opacity=alt.condition(sel_uf, alt.value(1), alt.value(0.55)),
                tooltip=[
                    alt.Tooltip("UF:N"),
                    alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
                    alt.Tooltip("Leader:N", title="Líder (UF)"),
                    alt.Tooltip("Gap_pp:Q", title="Gap vs líder (pp)", format="+.1f"),
                    alt.Tooltip("OppVol:Q", title="Opp (Qtde)", format=",.0f"),
                ],
            )
            .add_params(sel_uf)
            .properties(height=max(420, len(dfp) * 22), title="Clique para focar UF")
            .configure_view(stroke=None)
            .configure(background="transparent")
            .configure_axis(grid=False)
        )

        with st.container(height=520, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_UF)

        uf_clicked = selected_value_from_chart_state(K_UF, "SEL_UF", "UF")
        if uf_clicked != st.session_state.get("cli_uf_focus"):
            st.session_state["cli_mix_focus"] = None
        st.session_state["cli_uf_focus"] = uf_clicked

    with cB:
        st.markdown("##### 🧩 Em que (Mix) — Opp (Qtde) vs líder do Mix")
        df_mix_base = df_c_all.copy()
        if st.session_state.get("cli_uf_focus"):
            df_mix_base = df_mix_base[df_mix_base["UF"].eq(st.session_state["cli_uf_focus"])]

        df_mix = calc_vs_fac_leader_dim(df_mix_base, "Mix Produto")
        df_mix = df_mix[df_mix["Market"] > 0].sort_values("OppVol", ascending=False).head(30).copy()

        sel_mix = alt.selection_point(fields=["Mix Produto"], name="SEL_MIX", clear="dblclick")

        dfp = df_mix.copy()
        dfp["_lab"] = dfp["OppVol"].apply(fmt_int)
        dfp["__y"] = dfp["Mix Produto"].astype(str) + " · " + dfp["_lab"].astype(str)
        axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)

        ch = (
            alt.Chart(dfp).mark_bar()
            .encode(
                x=alt.X("OppVol:Q", title=None),
                y=alt.Y("__y:N", sort="-x", axis=axis_right),
                color=alt.Color("Sign:N", scale=alt.Scale(domain=["FAC", "LEADER"], range=[C_FAC, C_COMP]), legend=None),
                opacity=alt.condition(sel_mix, alt.value(1), alt.value(0.55)),
                tooltip=[
                    alt.Tooltip("Mix Produto:N", title="Mix"),
                    alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
                    alt.Tooltip("Leader:N", title="Líder (Mix)"),
                    alt.Tooltip("Gap_pp:Q", title="Gap vs líder (pp)", format="+.1f"),
                    alt.Tooltip("OppVol:Q", title="Opp (Qtde)", format=",.0f"),
                ],
            )
            .add_params(sel_mix)
            .properties(height=max(420, len(dfp) * 22), title="Clique para focar Mix")
            .configure_view(stroke=None)
            .configure(background="transparent")
            .configure_axis(grid=False)
        )

        with st.container(height=520, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=K_MIX)

        mix_clicked = selected_value_from_chart_state(K_MIX, "SEL_MIX", "Mix Produto")
        st.session_state["cli_mix_focus"] = mix_clicked

    with cC:
        st.markdown("##### 🏭 Contra quem (Implementadoras) — no recorte selecionado")

        df_impl = apply_focus_filters(df_c_all)
        leader_now = leader_in_slice(df_impl)

        impl = (df_impl.groupby("Implementadora", as_index=False)["Qtde"].sum()
                .rename(columns={"Qtde": "Volume"})
                .sort_values("Volume", ascending=False)
                .head(25).copy())

        if impl.empty:
            st.info("Sem dados no recorte.")
        else:
            def role(x: str) -> str:
                x = str(x)
                if x == FACCHINI:
                    return "FAC"
                if x == leader_now:
                    return "LEADER"
                return "OTH"

            impl["Role"] = impl["Implementadora"].apply(role)
            impl["_lab"] = impl["Volume"].apply(fmt_int)
            impl["__y"] = impl["Implementadora"].astype(str) + " · " + impl["_lab"].astype(str)

            axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)
            color_scale = alt.Scale(domain=["FAC", "LEADER", "OTH"], range=[C_FAC, C_COMP, C_GRAY])

            ch = (
                alt.Chart(impl).mark_bar()
                .encode(
                    x=alt.X("Volume:Q", title=None),
                    y=alt.Y("__y:N", sort="-x", axis=axis_right),
                    color=alt.Color("Role:N", scale=color_scale, legend=None),
                    tooltip=[alt.Tooltip("Implementadora:N"), alt.Tooltip("Volume:Q", format=",.0f")],
                )
                .properties(height=max(420, len(impl) * 22), title=f"FACCHINI (vermelho) | Líder do recorte (azul): {leader_now}")
                .configure_view(stroke=None)
                .configure(background="transparent")
                .configure_axis(grid=False)
            )

            with st.container(height=520, border=False):
                st.altair_chart(ch, use_container_width=True)

with tab_hist:
    df_hist = df_scope_all.copy()
    df_hist = df_hist[df_hist["Cliente"].astype(str).eq(str(focus_client))]
    df_hist = apply_focus_filters(df_hist)

    if df_hist.empty:
        st.info("Sem histórico para o recorte atual.")
    else:
        # líder do histórico no slice (para série concorrente)
        leader_hist = leader_in_slice(df_hist)

        hist_market = df_hist.groupby("Ano", as_index=False).agg(Mercado=("Qtde", "sum"))
        hist_fac = (df_hist[df_hist["Implementadora"].eq(FACCHINI)]
                    .groupby("Ano", as_index=False).agg(FACCHINI=("Qtde", "sum")))
        hist_lead = (df_hist[df_hist["Implementadora"].eq(leader_hist)]
                     .groupby("Ano", as_index=False).agg(Lider=("Qtde", "sum"))) if leader_hist != "—" else pd.DataFrame({"Ano": [], "Lider": []})

        hist = hist_market.merge(hist_fac, on="Ano", how="left").merge(hist_lead, on="Ano", how="left")
        hist["FACCHINI"] = hist["FACCHINI"].fillna(0)
        hist["Lider"] = hist["Lider"].fillna(0)

        melt = hist.melt(
            id_vars=["Ano"],
            value_vars=["Mercado", "FACCHINI", "Lider"],
            var_name="Série",
            value_name="Qtde"
        ).sort_values("Ano")

        color_scale = alt.Scale(domain=["Mercado", "FACCHINI", "Lider"], range=[C_GRAY, C_FAC, C_COMP])

        line = (
            alt.Chart(melt).mark_line(point=True)
            .encode(
                x=alt.X("Ano:O", title=None),
                y=alt.Y("Qtde:Q", title=None),
                color=alt.Color("Série:N", scale=color_scale, legend=alt.Legend(title=None)),
                tooltip=[
                    alt.Tooltip("Ano:O", title="Ano"),
                    alt.Tooltip("Série:N", title="Série"),
                    alt.Tooltip("Qtde:Q", title="Qtde", format=",.0f"),
                ],
            )
            .properties(height=380, title=f"Histórico no recorte (Líder: {leader_hist})")
            .configure_view(stroke=None)
            .configure(background="transparent")
            .configure_axis(grid=False)
        )

        st.altair_chart(line, use_container_width=True)

with tab_rx:
    df_rx = apply_focus_filters(df_c_all)

    cols = ["Ano", "Tipo", "UF", "Municipio", "Cliente", "Implementadora", "Mix Produto", "Modelo", "Representante", "Qtde"]
    cols = [c for c in cols if c in df_rx.columns]

    st.dataframe(
        df_rx[cols].sort_values("Qtde", ascending=False).head(2000),
        use_container_width=True,
        hide_index=True
    )
