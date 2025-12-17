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
# CSS
# ============================================================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
hr { margin: 1.2rem 0; opacity: .25; }
.vega-embed summary { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("cli_focus", None)
st.session_state.setdefault("cli_uf_focus", None)
st.session_state.setdefault("cli_mix_focus", None)

# Keys (não repetir)
K_TOP  = "chart_top200"
K_DOM  = "chart_dom"
K_WS   = "chart_ws"
K_LOSS = "chart_loss"
K_MAP  = "chart_map"

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

def apply_click_to_focus(chart_key: str, selection_name: str, field_name: str = "Cliente"):
    """Se houve clique nesse gráfico, atualiza o foco e rerun (para toda a página responder)."""
    clicked = selected_value_from_chart_state(chart_key, selection_name, field_name)
    if clicked is None:
        return
    clicked = str(clicked)
    if clicked != str(st.session_state.get("cli_focus")):
        st.session_state["cli_focus"] = clicked
        st.session_state["cli_uf_focus"] = None
        st.session_state["cli_mix_focus"] = None
        st.rerun()

def fmt_int(x) -> str:
    try:
        if pd.isna(x):
            return "0"
        return lb.formatar_br(int(round(float(x))))
    except Exception:
        return "0"

def card_robust(label, value, sub, color_border="#94a3b8"):
    return f"""
    <div style="
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid {color_border};
        border-radius: 10px;
        padding: 14px 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        display: flex; flex-direction: column; gap: 4px;
    ">
        <div style="font-size: 0.70rem; text-transform: uppercase; font-weight: 800; color: #64748b; letter-spacing: 0.06em;">{label}</div>
        <div style="font-size: 1.55rem; font-weight: 900; color: #0f172a; line-height: 1.0;">{value}</div>
        <div style="font-size: 0.86rem; color: #334155; font-weight: 600;">{sub}</div>
    </div>
    """

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

df0 = load_df()

# ============================================================
# MODELO (carteira por cliente)
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
        cur["dShareFac"] = np.nan
    else:
        prev = g[g["Ano"].eq(int(year_comp))].copy()
        prev = prev.rename(columns={"Share_Fac": "Share_Fac_prev"})[["Cliente", "Share_Fac_prev"]]
        cur = cur.merge(prev, on="Cliente", how="left")
        cur["dShareFac"] = cur["Share_Fac"].astype(float) - cur["Share_Fac_prev"].astype(float)
    return cur[["Cliente", "dShareFac"]]

def top_competitor_per_client(df_year: pd.DataFrame) -> pd.DataFrame:
    comp = (df_year[df_year["Implementadora"].ne(FACCHINI)]
            .groupby(["Cliente", "Implementadora"], as_index=False)
            .agg(Comp1Vol=("Qtde", "sum"))
            .sort_values(["Cliente", "Comp1Vol"], ascending=[True, False]))
    return comp.drop_duplicates("Cliente", keep="first").rename(columns={"Implementadora": "Comp1"})

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

def add_cluster(df_clients: pd.DataFrame) -> pd.DataFrame:
    m_med = float(df_clients["Market"].median())
    s_med = float(df_clients["Share_Fac"].median())

    def _cls(r):
        if r["Market"] >= m_med and r["Share_Fac"] < s_med:
            base = "Atacar"
        elif r["Market"] >= m_med and r["Share_Fac"] >= s_med:
            base = "Defender"
        elif r["Market"] < m_med and r["Share_Fac"] < s_med:
            base = "Cultivar"
        else:
            base = "Manter"
        if base == "Defender" and (not pd.isna(r.get("dShareFac", np.nan))) and float(r.get("dShareFac", 0)) < 0:
            return "Risco"
        return base

    out = df_clients.copy()
    out["Cluster"] = out.apply(_cls, axis=1)
    return out

# ============================================================
# COMPUTE (cacheado) -> clique rápido
# ============================================================
@st.cache_data(show_spinner=False)
def compute_all(ano_sel, ano_comp, tipo_sel, mix_sel_t, uf_filter_t, ignore_ni):
    df = load_df().copy()

    df_scope_all = df
    if tipo_sel != "Consolidado":
        df_scope_all = df_scope_all[df_scope_all["Tipo"].eq(tipo_sel)]
    if mix_sel_t:
        df_scope_all = df_scope_all[df_scope_all["Mix Produto"].isin(list(mix_sel_t))]
    if uf_filter_t:
        df_scope_all = df_scope_all[df_scope_all["UF"].isin(list(uf_filter_t))]

    if ignore_ni == "Sim":
        for col in ["Cliente", "Mix Produto", "Municipio", "Implementadora"]:
            df_scope_all = df_scope_all[~df_scope_all[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

    df_base = df_scope_all[df_scope_all["Ano"].eq(int(ano_sel))].copy()

    years_needed = [int(ano_sel)] + ([] if ano_comp == "—" else [int(ano_comp)])
    df_years = df_scope_all[df_scope_all["Ano"].isin(years_needed)].copy()
    g_cy = agg_client_year(df_years)
    yoy_cur = add_yoy(g_cy, int(ano_sel), ano_comp)

    df_clients = build_clients_model(df_base, yoy_cur)
    df_clients = df_clients[df_clients["Market"] > 0].copy()
    df_clients = add_cluster(df_clients)

    return df_scope_all, df_base, df_clients

# ============================================================
# FILTROS TOPO
# ============================================================
with st.container():
    c1, c2, c3, c4, c5, c6 = st.columns([1.0, 1.2, 1.6, 2.1, 1.6, 1.1], gap="medium")

    with c1:
        anos = sorted([a for a in df0["Ano"].unique() if a > 0], reverse=True)
        ano_sel = st.selectbox("Ano base", anos if anos else [0])

    with c2:
        ano_prev = (ano_sel - 1) if (ano_sel - 1) in anos else None
        comp_opts = [a for a in anos if a != ano_sel]
        idx = 0
        if ano_prev in comp_opts:
            idx = 1 + comp_opts.index(ano_prev)
        ano_comp = st.selectbox("Comparar com", ["—"] + comp_opts, index=idx)

    with c3:
        tipos = sorted([t for t in df0["Tipo"].unique() if t not in ("", "N/I")])
        tipo_sel = st.selectbox("Tipo", ["Consolidado"] + tipos)

    df_opt = df0.copy()
    if tipo_sel != "Consolidado":
        df_opt = df_opt[df_opt["Tipo"].eq(tipo_sel)]

    with c4:
        mix_opts = sorted([m for m in df_opt["Mix Produto"].unique() if m not in ("", "N/I")])
        mix_sel = st.multiselect("Mix Produto (opcional)", mix_opts, default=[])

    if mix_sel:
        df_opt = df_opt[df_opt["Mix Produto"].isin(mix_sel)]

    with c5:
        uf_opts = sorted([u for u in df_opt["UF"].unique() if u not in ("", "N/I")])
        uf_filter = st.multiselect("UF (opcional)", uf_opts, default=[])

    with c6:
        ignore_ni = st.radio("Ignorar N/I", ["Não", "Sim"], horizontal=True, index=1)

df_scope_all, df_base, df_clients = compute_all(
    ano_sel, ano_comp, tipo_sel, tuple(mix_sel), tuple(uf_filter), ignore_ni
)

if df_base.empty:
    st.warning("Sem dados no recorte atual.")
    st.stop()

if df_clients.empty:
    st.warning("Sem clientes com volume no recorte.")
    st.stop()

# ============================================================
# CONTEXTO GLOBAL vs CONTEXTO FILTRADO (FOCO)
# ============================================================
focus = st.session_state.get("cli_focus")
if focus is not None and str(focus) not in set(df_clients["Cliente"].astype(str)):
    # se o foco ficou inválido após mudar filtros
    st.session_state["cli_focus"] = None
    focus = None

df_ctx = df_base if not focus else df_base[df_base["Cliente"].astype(str).eq(str(focus))].copy()

# ============================================================
# KPIs (sempre do mercado inteiro) + (opcional) do foco
# ============================================================
market_total = int(df_base["Qtde"].sum())
fac_total = int(df_base.loc[df_base["Implementadora"].eq(FACCHINI), "Qtde"].sum())
share_total = (fac_total / market_total) if market_total > 0 else 0

if focus:
    # pelo modelo de clientes (mais rápido/consistente)
    row = df_clients[df_clients["Cliente"].astype(str).eq(str(focus))]
    if not row.empty:
        r = row.iloc[0]
        c_market = int(r["Market"])
        c_fac = int(r["Facchini"])
        c_share = float(r["Share_Fac"])
        c_ws = int(r["WhiteSpace"])
    else:
        c_market = int(df_ctx["Qtde"].sum())
        c_fac = int(df_ctx.loc[df_ctx["Implementadora"].eq(FACCHINI), "Qtde"].sum())
        c_share = (c_fac / c_market) if c_market > 0 else 0
        c_ws = max(0, c_market - c_fac)
else:
    c_market = c_fac = c_ws = 0
    c_share = 0

# ============================================================
# UI: foco + limpar
# ============================================================
a, b = st.columns([4, 1], gap="small")
with a:
    st.markdown(
        f"**Foco:** `{focus}`" if focus else "**Foco:** — (clique em um gráfico para focar)"
    )
with b:
    if st.button("Limpar foco", key="btn_clear_focus"):
        st.session_state["cli_focus"] = None
        st.session_state["cli_uf_focus"] = None
        st.session_state["cli_mix_focus"] = None
        st.rerun()

st.markdown("<hr style='opacity:.18'>", unsafe_allow_html=True)

# ============================================================
# 1) RANKING (TOP 200) + CARDS (MERCADO TOTAL + FOCO)
# ============================================================
ui.section("🏆 1) Ranking de Clientes (Top 200 — clique para focar)")

df_top200 = df_clients.sort_values("Market", ascending=False).head(200).copy()

c_left, c_right = st.columns([3.8, 1], gap="small")

with c_left:
    sel_top = alt.selection_point(fields=["Cliente"], name="SEL_TOP", clear="dblclick")
    nova_paleta = ["#ff7777", "#fd2727", "#ff0000", "#be0000", "#990000"]

    ch_rank = (
        alt.Chart(df_top200)
        .mark_bar(cornerRadiusEnd=4, height=16)
        .encode(
            x=alt.X("Market:Q", title=None, axis=None),
            y=alt.Y("Cliente:N", sort="-x",
                    axis=alt.Axis(title=None, labelLimit=260, labelFontSize=12, labelFontWeight=600, grid=False)),
            color=alt.condition(
                alt.datum.Share_Fac == 0,
                alt.value("#cbd5e1"),
                alt.Color("Share_Fac:Q",
                          scale=alt.Scale(domain=[0, 0.25, 0.5, 0.75, 1], range=nova_paleta),
                          legend=None)
            ),
            opacity=alt.condition(sel_top, alt.value(1), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("Cliente"),
                alt.Tooltip("Market", title="Mercado", format=",.0f"),
                alt.Tooltip("Facchini", title="FACCHINI", format=",.0f"),
                alt.Tooltip("Share_Fac", title="Share", format=".1%"),
            ],
        )
        .add_params(sel_top)
        .properties(height=1400)  # top200 ok
        .configure_view(stroke=None)
    )

    with st.container(height=520, border=False):
        st.altair_chart(ch_rank, use_container_width=True, on_select="rerun", key=K_TOP)

    # clique -> foco (e toda página responde)
    apply_click_to_focus(K_TOP, "SEL_TOP", "Cliente")

with c_right:
    st.markdown("##### 🌍 Mercado (Total do recorte)")
    st.markdown(card_robust("Potencial Total", fmt_int(market_total), "Mercado no recorte", "#94a3b8"), unsafe_allow_html=True)
    st.markdown(card_robust("FACCHINI Total", fmt_int(fac_total), f"Share: <b>{share_total:.1%}</b>", "#ef4444"), unsafe_allow_html=True)
    st.markdown(card_robust("Dinheiro na Mesa", fmt_int(market_total - fac_total), "WhiteSpace total", "#3b82f6"), unsafe_allow_html=True)

    if focus:
        st.markdown("##### 🎯 Cliente (Foco)")
        st.markdown(card_robust("Potencial", fmt_int(c_market), "Mercado do cliente", "#94a3b8"), unsafe_allow_html=True)
        st.markdown(card_robust("FACCHINI", fmt_int(c_fac), f"Share: <b>{c_share:.1%}</b>", "#ef4444"), unsafe_allow_html=True)
        st.markdown(card_robust("WhiteSpace", fmt_int(c_ws), "Espaço no cliente", "#3b82f6"), unsafe_allow_html=True)

st.markdown("<hr style='opacity:.18'>", unsafe_allow_html=True)

# ============================================================
# 2) TRIPÉ TÁTICO (sempre calcula no MERCADO, mas clique foca e o resto filtra)
# ============================================================
ui.section("📊 2) Tripé Tático (clique para focar)")

df_dom  = df_clients[df_clients["Share_Fac"] >= 0.50].sort_values("Facchini", ascending=False).head(10)
df_ws   = df_clients.sort_values("WhiteSpace", ascending=False).head(10)

df_loss = pd.DataFrame()
if ano_comp != "—" and "ImpactShare" in df_clients.columns:
    df_loss = df_clients[df_clients["ImpactShare"] < -0.5].sort_values("ImpactShare", ascending=True).head(10).copy()
    df_loss["AbsLoss"] = df_loss["ImpactShare"].abs()

def bar_rank(df_in: pd.DataFrame, dim: str, val: str, sel, title: str, color: str, height: int):
    if df_in.empty:
        return alt.Chart(pd.DataFrame({dim: [], val: []})).mark_bar().properties(height=height, title=title)

    dfp = df_in.copy()
    dfp["_lab"] = dfp[val].apply(fmt_int)
    dfp["__y"] = dfp[dim].astype(str) + "\u2009\u2009·\u2009" + dfp["_lab"].astype(str)

    return (
        alt.Chart(dfp)
        .mark_bar(color=color, cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{val}:Q", title=None, axis=None),
            y=alt.Y("__y:N", sort="-x", axis=alt.Axis(orient="right", title=None, ticks=False, domain=False, labelLimit=320)),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
            tooltip=[alt.Tooltip(dim), alt.Tooltip(val, format=",.0f")],
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
    )

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(f"##### 🛡️ Fortaleza ({len(df_dom)})")
    sel = alt.selection_point(fields=["Cliente"], name="SEL_DOM", clear="dblclick")
    st.altair_chart(bar_rank(df_dom, "Cliente", "Facchini", sel, "", C_GREEN, 350),
                    use_container_width=True, on_select="rerun", key=K_DOM)
    apply_click_to_focus(K_DOM, "SEL_DOM", "Cliente")

with c2:
    st.markdown(f"##### ⚔️ Ataque ({len(df_ws)})")
    sel = alt.selection_point(fields=["Cliente"], name="SEL_WS", clear="dblclick")
    st.altair_chart(bar_rank(df_ws, "Cliente", "WhiteSpace", sel, "", C_AMB, 350),
                    use_container_width=True, on_select="rerun", key=K_WS)
    apply_click_to_focus(K_WS, "SEL_WS", "Cliente")

with c3:
    st.markdown(f"##### 🚨 Sangramento ({len(df_loss)})")
    if df_loss.empty:
        st.info("Sem perdas relevantes (ou selecione ano comparativo).")
    else:
        sel = alt.selection_point(fields=["Cliente"], name="SEL_LOSS", clear="dblclick")
        st.altair_chart(bar_rank(df_loss, "Cliente", "AbsLoss", sel, "", C_RISK, 350),
                        use_container_width=True, on_select="rerun", key=K_LOSS)
        apply_click_to_focus(K_LOSS, "SEL_LOSS", "Cliente")

st.markdown("<hr style='opacity:.18'>", unsafe_allow_html=True)

# ============================================================
# 3) MAPA (RESPONDE AO FOCO)
# ============================================================
ui.section("🗺️ 3) Mapa da Carteira (responde ao foco)")

# Quando tem foco, você pode decidir:
# - mostrar só o ponto do cliente (df_clients_foco)
# - ou mostrar todos com destaque do foco
df_map = df_clients.copy()
df_map["__is_focus"] = df_map["Cliente"].astype(str).eq(str(focus)) if focus else False

sel_map = alt.selection_point(fields=["Cliente"], name="SEL_MAP", clear="dblclick")

ch_map = (
    alt.Chart(df_map)
    .mark_circle()
    .encode(
        x=alt.X("Market:Q", title="Tamanho do cliente (Qtde)", scale=alt.Scale(type="log")),
        y=alt.Y("Share_Fac:Q", title="Share FACCHINI", axis=alt.Axis(format="%")),
        size=alt.Size("Market:Q", legend=None),
        color=alt.Color("Cluster:N", legend=alt.Legend(title="Macro decisão")),
        opacity=alt.condition(sel_map, alt.value(1), alt.value(0.25)),
        tooltip=[
            alt.Tooltip("Cliente:N"),
            alt.Tooltip("Cluster:N"),
            alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
            alt.Tooltip("Facchini:Q", title="FACCHINI", format=",.0f"),
            alt.Tooltip("Share_Fac:Q", title="Share", format=".1%"),
        ],
    )
    .add_params(sel_map)
    .properties(height=520)
    .configure_view(stroke=None)
)

st.altair_chart(ch_map, use_container_width=True, on_select="rerun", key=K_MAP)
apply_click_to_focus(K_MAP, "SEL_MAP", "Cliente")

st.markdown("<hr style='opacity:.18'>", unsafe_allow_html=True)

# ============================================================
# 4) EXEMPLO: QUALQUER BLOCO ABAIXO USA df_ctx (já filtrado pelo foco)
# ============================================================
ui.section("📌 4) Qualquer coisa abaixo responde ao foco")

st.caption("Exemplo: tabela do recorte atual (já filtrada pelo foco)")
st.dataframe(
    df_ctx.groupby(["UF", "Mix Produto"], as_index=False).agg(Qtde=("Qtde", "sum")).sort_values("Qtde", ascending=False).head(30),
    use_container_width=True,
    hide_index=True,
)
