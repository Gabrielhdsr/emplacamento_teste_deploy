import streamlit as st
import pandas as pd
import numpy as np

import data as dt
import lib as lb
import ui

# Engines (opcionais)
import altair as alt

try:
    import plotly.express as px
except Exception:
    px = None

try:
    from streamlit_echarts import st_echarts
except Exception:
    st_echarts = None

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="Top Clientes • Gráficos", page_icon="📊")
ui.apply_style()

ui.header(
    "Top Clientes • Gráficos",
    "Top do Mercado + 3 análises (Fortaleza / WhiteSpace / Perda) com Altair, Plotly ou ECharts"
)

FACCHINI = "FACCHINI"

# Paleta (igual sua)
C_FAC   = "#dc2626"
C_COMP  = "#2563eb"
C_GRAY  = "#9ca3af"
C_GREEN = "#16a34a"
C_AMB   = "#f59e0b"
C_RISK  = "#7c3aed"


# ============================================================
# CSS (minimalista)
# ============================================================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }
hr { margin: 1.2rem 0; opacity: .25; }

.metric-card{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(148,163,184,.16);
  border-radius: 16px;
  padding: 14px 14px;
  min-height: 110px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  box-shadow: 0 8px 18px rgba(0,0,0,.10);
}
.metric-label{
  font-size: .80rem;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: rgba(148,163,184,.95);
}
.metric-value{
  font-size: 2.0rem;
  letter-spacing: -0.02em;
  line-height: 1.05;
  white-space: nowrap;
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
.kpi-gray{ border-left-color: rgba(156,163,175,.80); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("cli_focus", None)

# keys (únicas)
K_TOP   = "pg2_top_mkt"
K_DOM   = "pg2_dom"
K_WS    = "pg2_ws"
K_LOSS  = "pg2_loss"


# ============================================================
# HELPERS
# ============================================================
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

def small_metric(label: str, value: str, sub: str = "", accent="kpi-gray"):
    html = f"""
    <div class="metric-card kpi-accent {accent}">
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        <div class="metric-sub">{sub}</div>
    </div>
    """
    return " ".join(html.split())

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


# ============================================================
# LOAD
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
# FILTROS + ENGINE
# ============================================================
with st.container():
    c0, c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1.7, 2.0, 1.6, 1.0, 1.4], gap="medium")

    with c0:
        engine = st.selectbox("Motor do gráfico", ["Altair (recomendado)", "Plotly", "ECharts"])

    with c1:
        anos = sorted([a for a in df["Ano"].unique() if a > 0], reverse=True)
        ano_sel = st.selectbox("Ano base", anos if anos else [0])

    with c2:
        ano_prev = (ano_sel - 1) if (ano_sel - 1) in anos else None
        comp_opts = [a for a in anos if a != ano_sel]
        idx = 0
        if ano_prev in comp_opts:
            idx = 1 + comp_opts.index(ano_prev)
        ano_comp = st.selectbox("Comparar com", ["—"] + comp_opts, index=idx)

    with c3:
        tipos = sorted([t for t in df["Tipo"].unique() if t not in ("", "N/I")])
        tipo_sel = st.selectbox("Tipo", ["Consolidado"] + tipos)

    df_scope_all = df.copy()
    if tipo_sel != "Consolidado":
        df_scope_all = df_scope_all[df_scope_all["Tipo"].eq(tipo_sel)]

    with c4:
        mix_opts = sorted([m for m in df_scope_all["Mix Produto"].unique() if m not in ("", "N/I")])
        mix_sel = st.multiselect("Mix Produto (opcional)", mix_opts, default=[])

    if mix_sel:
        df_scope_all = df_scope_all[df_scope_all["Mix Produto"].isin(mix_sel)]

    with c5:
        uf_opts = sorted([u for u in df_scope_all["UF"].unique() if u not in ("", "N/I")])
        uf_filter = st.multiselect("UF (opcional)", uf_opts, default=[])

    if uf_filter:
        df_scope_all = df_scope_all[df_scope_all["UF"].isin(uf_filter)]

    with c6:
        ignore_ni = st.radio("Ignorar N/I", ["Não", "Sim"], horizontal=True, index=1)

if ignore_ni == "Sim":
    for col in ["Cliente", "Mix Produto", "Municipio", "Implementadora"]:
        df_scope_all = df_scope_all[~df_scope_all[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

df_scope = df_scope_all[df_scope_all["Ano"].eq(ano_sel)].copy()
if df_scope.empty:
    st.warning("Sem dados no recorte atual.")
    st.stop()


# ============================================================
# MODELO (igual sua lógica)
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

# foco default
if st.session_state.get("cli_focus") not in set(df_clients["Cliente"].astype(str)):
    st.session_state["cli_focus"] = str(df_clients.sort_values("Market", ascending=False).iloc[0]["Cliente"])


# ============================================================
# RENDERERS (Altair / Plotly / ECharts)
# ============================================================
def bar_altair(df_in, dim, val, title, color, fmt, height, key, sel_name="SEL"):
    dfp = df_in.copy()
    if dfp.empty:
        st.info("Sem dados.")
        return None

    # label formatado
    if fmt == "pct":
        dfp["_lab"] = dfp[val].apply(lambda x: "—" if pd.isna(x) else f"{float(x)*100:.1f}%")
        x_format = "%"
    elif fmt == "pp":
        dfp["_lab"] = dfp[val].apply(lambda x: "—" if pd.isna(x) else f"{float(x):+.1f} pp")
        x_format = None
    else:
        dfp["_lab"] = dfp[val].apply(fmt_int)
        x_format = ",.0f"

    # “rótulo” embutido no eixo Y (single-view!)
    dfp["__y"] = dfp[dim].astype(str).str.slice(0, 28) + "  [" + dfp["_lab"].astype(str) + "]"

    sel = alt.selection_point(fields=[dim], name=sel_name, clear="dblclick")

    axis_right = alt.Axis(
        orient="right", title=None, labelFontSize=12,
        ticks=False, domain=False, labelPadding=10, labelLimit=340
    )

    ch = (
        alt.Chart(dfp)
        .mark_bar(color=color, cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{val}:Q", title=None, axis=None),
            y=alt.Y("__y:N", sort="-x", axis=axis_right),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
            tooltip=[
                alt.Tooltip(f"{dim}:N", title=dim),
                alt.Tooltip(f"{val}:Q", title="Valor", format=x_format if x_format else ""),
            ],
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )

    st.altair_chart(ch, use_container_width=True, on_select="rerun", key=key)

    # clique funciona porque a seleção é pelo dim (Cliente, etc.)
    clicked = selected_value_from_chart_state(key, sel_name, dim)
    return clicked

def bar_plotly(df_in, dim, val, title, fmt, height, key):
    if px is None:
        st.warning("Plotly não está disponível neste ambiente.")
        return None
    dfp = df_in.copy()
    if dfp.empty:
        st.info("Sem dados.")
        return None

    # rótulo
    if fmt == "pct":
        dfp["_lab"] = dfp[val].apply(lambda x: "" if pd.isna(x) else f"{float(x)*100:.1f}%")
    else:
        dfp["_lab"] = dfp[val].apply(fmt_int)

    fig = px.bar(
        dfp.sort_values(val, ascending=True),
        x=val, y=dim, orientation="h",
        text="_lab",
        title=title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(side="right"),
    )

    if plotly_events is not None:
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=key)
        # clicked -> lista de eventos; o y vem em 'y' ou 'pointNumber' dependendo
        if clicked:
            # tenta pegar o nome diretamente
            if "y" in clicked[0]:
                return str(clicked[0]["y"])
            # fallback: index
            idx = clicked[0].get("pointNumber")
            if idx is not None:
                # cuidado: como ordenamos ascending=True, idx é nessa ordem
                ordered = dfp.sort_values(val, ascending=True)[dim].astype(str).tolist()
                if 0 <= idx < len(ordered):
                    return ordered[idx]
        return None

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Clique não está ligado no Plotly (instale `streamlit-plotly-events` se quiser clique).")
    return None

def bar_echarts(df_in, dim, val, title, fmt, height, key):
    if st_echarts is None:
        st.warning("ECharts não está disponível neste ambiente (instale `streamlit-echarts`).")
        return None
    dfp = df_in.copy()
    if dfp.empty:
        st.info("Sem dados.")
        return None

    dfp = dfp.sort_values(val, ascending=True)
    cats = dfp[dim].astype(str).tolist()
    vals = dfp[val].astype(float).fillna(0).tolist()

    if fmt == "pct":
        labels = [("" if pd.isna(x) else f"{float(x)*100:.1f}%") for x in dfp[val].tolist()]
    else:
        labels = [fmt_int(x) for x in dfp[val].tolist()]

    option = {
        "backgroundColor": "rgba(0,0,0,0)",
        "title": {"text": title, "left": "0", "top": "0", "textStyle": {"color": "#cbd5e1"}},
        "grid": {"left": 0, "right": 10, "top": 45, "bottom": 0, "containLabel": True},
        "xAxis": {"type": "value", "axisLine": {"show": False}, "axisTick": {"show": False}, "axisLabel": {"show": False}, "splitLine": {"show": False}},
        "yAxis": {"type": "category", "data": cats, "axisLine": {"show": False}, "axisTick": {"show": False}, "axisLabel": {"color": "#cbd5e1"}, "inverse": False},
        "series": [{
            "type": "bar",
            "data": [{"value": v, "label": {"show": True, "position": "right", "formatter": labels[i]}} for i, v in enumerate(vals)],
            "barWidth": 16,
            "itemStyle": {"borderRadius": [0, 8, 8, 0]},
        }]
    }

    events = {"click": "function(params){ return params.name; }"}
    ev = st_echarts(option, height=f"{height}px", key=key, events=events)

    if isinstance(ev, dict) and ev.get("click"):
        return str(ev["click"])
    return None


def bar_any(engine_name, df_in, dim, val, title, color, fmt, height, key):
    if engine_name.startswith("Altair"):
        return bar_altair(df_in, dim, val, title, color, fmt, height, key, sel_name="SEL")
    if engine_name == "Plotly":
        return bar_plotly(df_in, dim, val, title, fmt, height, key)
    return bar_echarts(df_in, dim, val, title, fmt, height, key)


# ============================================================
# TOP 15 (Mercado)
# ============================================================
ui.section("🏆 Top 15 Clientes do Mercado")

df_top = df_clients.sort_values("Market", ascending=False).head(15).copy()
top_vol = int(df_top["Market"].sum())
top_fac = int(df_top["Facchini"].sum())
top_share = (top_fac / top_vol) if top_vol > 0 else 0
top_ws = int(df_top["WhiteSpace"].sum())

cA, cB = st.columns([2.4, 1], gap="large")
with cA:
    clicked = bar_any(
        engine, df_top,
        dim="Cliente", val="Market",
        title="Ranking por Volume Total de Compras (Top 15)",
        color=C_GRAY, fmt="int", height=520,
        key=K_TOP
    )
    if clicked:
        st.session_state["cli_focus"] = str(clicked)

with cB:
    st.markdown(small_metric("Volume Total", fmt_int(top_vol), "Soma do Top 15", "kpi-gray"), unsafe_allow_html=True)
    st.write("")
    st.markdown(small_metric("Vol. FACCHINI", fmt_int(top_fac), f"Share no Top 15: {top_share:.1%}", "kpi-red"), unsafe_allow_html=True)
    st.write("")
    st.markdown(small_metric("WhiteSpace", fmt_int(top_ws), "Volume na concorrência", "kpi-blue"), unsafe_allow_html=True)

st.write("---")


# ============================================================
# 3 ANÁLISES (Fortaleza / WhiteSpace / Perda)
# ============================================================
ui.section("📊 3 Análises (Fortaleza / WhiteSpace / Perda)")

col1, col2, col3 = st.columns(3, gap="large")

# Fortaleza: share >= 50% (ordena por Facchini)
with col1:
    st.markdown("##### 🛡️ Fortaleza FACCHINI")
    df_dom = df_clients[df_clients["Share_Fac"] >= 0.50].sort_values("Facchini", ascending=False).head(10).copy()
    if df_dom.empty:
        st.info("Nenhum cliente com share > 50% no recorte.")
    else:
        clicked = bar_any(
            engine, df_dom,
            dim="Cliente", val="Facchini",
            title="Volume FACCHINI (Share > 50%)",
            color=C_FAC, fmt="int", height=420,
            key=K_DOM
        )
        if clicked:
            st.session_state["cli_focus"] = str(clicked)

# WhiteSpace: maior espaço
with col2:
    st.markdown("##### ⚔️ Terreno da Concorrência")
    df_ws = df_clients.sort_values("WhiteSpace", ascending=False).head(10).copy()
    clicked = bar_any(
        engine, df_ws,
        dim="Cliente", val="WhiteSpace",
        title="Volume da Concorrência (WhiteSpace)",
        color=C_COMP, fmt="int", height=420,
        key=K_WS
    )
    if clicked:
        st.session_state["cli_focus"] = str(clicked)

# Perda: ImpactShare < 0
with col3:
    st.markdown("##### 🚨 Alerta de Perda (YoY)")
    has_yoy = (ano_comp != "—") and df_clients["ImpactShare"].notna().any()
    if not has_yoy:
        st.warning("Selecione um 'Ano Comparar' para ver perdas.")
    else:
        df_loss = df_clients[df_clients["ImpactShare"] < 0].sort_values("ImpactShare", ascending=True).head(10).copy()
        if df_loss.empty:
            st.success("Sem perdas relevantes neste recorte.")
        else:
            # Mostra magnitude da perda (valor negativo): ainda dá pra manter e rotular
            clicked = bar_any(
                engine, df_loss,
                dim="Cliente", val="ImpactShare",
                title="Volume Perdido (estimado via share)",
                color=C_RISK, fmt="int", height=420,
                key=K_LOSS
            )
            if clicked:
                st.session_state["cli_focus"] = str(clicked)

st.write("---")


# ============================================================
# FOCO (sempre disponível)
# ============================================================
focus = st.session_state.get("cli_focus")
st.caption(f"Foco atual: **{focus}** (clique nas barras para mudar — Altair/ECharts suportam bem; Plotly depende do componente de eventos).")
