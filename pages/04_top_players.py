import streamlit as st
import pandas as pd
import altair as alt
import data as dt
import lib as lb
import ui

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="Top Players", page_icon="🏆")
ui.apply_style()
ui.header(
    "Top Players (Mercado vs Concorrentes)",
    "Clique no concorrente e veja onde ele ganha/perde da FACCHINI: UF → Cidade → Mix → Cliente (com histórico)."
)

FACCHINI = "FACCHINI"
PANEL_H = 520

# Paleta (fixa e consistente)
C_FAC = "#dc2626"   # vermelho (FACCHINI)
C_COMP = "#2563eb"  # azul (Concorrente)
C_NEU = "#f1f5f9"   # neutro
C_ZERO = "#e5e7eb"  # cinza (mercado zero)
C_GRAY = "#9ca3af"  # cinza barras neutras

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
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").fillna(0).astype(int)
    return df

df = load_df()

# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("tp_comp", None)
st.session_state.setdefault("tp_uf", None)
st.session_state.setdefault("tp_city", None)
st.session_state.setdefault("tp_mix", None)
st.session_state.setdefault("tp_cli", None)

# ============================================================
# HELPERS (seleção via session_state)
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

def selected_uf_from_chart_state(chart_key="tp_map", selection_name="UF_SEL_MAP"):
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
        if "properties.sigla" in dn:
            return dn["properties.sigla"]
        props = dn.get("properties")
        if isinstance(props, dict) and "sigla" in props:
            return props["sigla"]
        return None

    if isinstance(sel, list) and len(sel) > 0:
        return extract(sel[0])
    if isinstance(sel, dict):
        return extract(sel)
    return None

def fmt_int(x) -> str:
    try:
        return lb.formatar_br(int(x))
    except Exception:
        return "0"

def fmt_pp(x) -> str:
    try:
        return f"{float(x)*100:+.1f} pp"
    except Exception:
        return "+0.0 pp"

# ============================================================
# UI: cards pequenos
# ============================================================
def small_metric(label: str, value: str, sub: str = "", border_color="red"):
    # border_color: "red" ou "blue" ou "dark" conforme ui.py
    html = f"""
    <div class="metric-card border-left-{border_color}" style="padding:18px; border-radius:14px;">
        <div class="metric-label" style="margin-bottom:8px;">{label}</div>
        <div class="metric-value" style="font-size:2.2rem; letter-spacing:-1px; white-space:nowrap;">{value}</div>
        <div style="margin-top:10px; color:#94a3b8; font-size:0.85rem;">{sub}</div>
    </div>
    """
    return " ".join(html.split())

# ============================================================
# CORE: cálculo sempre vs FACCHINI
# Adv = Share_Concorrente - Share_FACCHINI
# ============================================================
def calc_vs_fac(df_base: pd.DataFrame, dim: str, comp: str) -> pd.DataFrame:
    if df_base.empty:
        return pd.DataFrame(columns=[dim, "Market", "Facchini", "Comp", "Share_Fac", "Share_Comp", "Adv", "GapVol"])

    m = df_base.groupby(dim, as_index=False).agg(Market=("Qtde", "sum"))
    f = df_base.loc[df_base["Implementadora"].eq(FACCHINI)].groupby(dim, as_index=False).agg(Facchini=("Qtde", "sum"))
    c = df_base.loc[df_base["Implementadora"].eq(comp)].groupby(dim, as_index=False).agg(Comp=("Qtde", "sum"))

    out = m.merge(f, on=dim, how="left").merge(c, on=dim, how="left")
    out["Facchini"] = out["Facchini"].fillna(0)
    out["Comp"] = out["Comp"].fillna(0)

    out["Share_Fac"] = (out["Facchini"] / out["Market"]).fillna(0)
    out["Share_Comp"] = (out["Comp"] / out["Market"]).fillna(0)
    out["Adv"] = out["Share_Comp"] - out["Share_Fac"]
    out["GapVol"] = out["Comp"] - out["Facchini"]
    return out

# ============================================================
# CHARTS (single view + seleção)
# ============================================================
def rank_bar_value_in_axis(df_in: pd.DataFrame, dim_col: str, value_col: str, sel, title: str, height: int, value_fmt="int"):
    """
    Single chart. Valor aparece no eixo (Dim · valor).
    value_fmt: "int" | "pp"
    """
    if df_in.empty:
        return None

    dfp = df_in.copy()
    if value_fmt == "pp":
        dfp["_val"] = dfp[value_col].apply(lambda x: f"{float(x)*100:+.1f} pp")
    else:
        dfp["_val"] = dfp[value_col].apply(lambda x: fmt_int(x))

    dfp["__y"] = dfp[dim_col].astype(str) + "  ·  " + dfp["_val"]

    axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)

    ch = (
        alt.Chart(dfp)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title=None),
            y=alt.Y("__y:N", sort="-x", axis=axis_right),
            tooltip=[
                alt.Tooltip(f"{dim_col}:N", title=dim_col),
                alt.Tooltip("Market:Q", title="Mercado", format=",.0f") if "Market" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
                alt.Tooltip("Comp:Q", title="Concorrente", format=",.0f") if "Comp" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
                alt.Tooltip("Facchini:Q", title="FACCHINI", format=",.0f") if "Facchini" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
                alt.Tooltip("Share_Comp:Q", title="Share Concorrente", format=".1%") if "Share_Comp" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
                alt.Tooltip("Share_Fac:Q", title="Share FACCHINI", format=".1%") if "Share_Fac" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
                alt.Tooltip("Adv:Q", title="Vantagem (pp)", format=".1%") if "Adv" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
                alt.Tooltip("GapVol:Q", title="GAP volume", format=",.0f") if "GapVol" in dfp.columns else alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f"),
            ],
            # cor pela vantagem (AZUL se Adv>=0, VERMELHO se Adv<0) quando existir
            color=alt.condition(
                "isValid(datum.Adv) && datum.Adv >= 0",
                alt.value(C_COMP),
                alt.value(C_FAC),
            ) if "Adv" in dfp.columns else alt.value(C_GRAY),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
        )
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )
    return ch

# ============================================================
# FILTROS DO TOPO (poucos, diretos)
# ============================================================
with st.container():
    f1, f2, f3, f4, f5 = st.columns([1.0, 1.6, 2.2, 1.7, 1.2], gap="medium")

    with f1:
        anos = sorted([a for a in df["Ano"].unique() if a > 0], reverse=True)
        ano_sel = st.selectbox("Ano base", anos if anos else [0])

    df_year = df[df["Ano"].eq(ano_sel)].copy() if ano_sel else df.copy()

    with f2:
        tipos = sorted([t for t in df_year["Tipo"].unique() if t not in ("", "N/I")])
        tipo_sel = st.selectbox("Tipo", ["Consolidado"] + tipos)

    if tipo_sel != "Consolidado":
        df_year = df_year[df_year["Tipo"].eq(tipo_sel)]

    with f3:
        mix_opts = sorted([m for m in df_year["Mix Produto"].unique() if m not in ("", "N/I")])
        mix_sel = st.multiselect("Mix Produto (opcional)", mix_opts, default=[])

    if mix_sel:
        df_year = df_year[df_year["Mix Produto"].isin(mix_sel)]

    with f4:
        uf_opts = sorted([u for u in df_year["UF"].unique() if u not in ("", "N/I")])
        uf_filter = st.multiselect("UF (opcional)", uf_opts, default=[])

    if uf_filter:
        df_year = df_year[df_year["UF"].isin(uf_filter)]

    with f5:
        ignore_ni = st.radio("Ignorar N/I", ["Não", "Sim"], horizontal=True, index=1)

if ignore_ni == "Sim":
    for col in ["Cliente", "Mix Produto", "Municipio", "Implementadora"]:
        df_year = df_year[~df_year[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

if df_year.empty:
    st.warning("Sem dados no recorte atual.")
    st.stop()

# ============================================================
# BLOCO 1 — ESCOLHER CONCORRENTE (Top 10)
# ============================================================
ui.section("🏆 Concorrente (Top 10)")

rank_impl = (
    df_year.groupby("Implementadora", as_index=False)["Qtde"].sum()
    .rename(columns={"Qtde": "Volume"})
    .sort_values("Volume", ascending=False)
)

# default: top 1 que não seja FACCHINI (se existir)
default_comp = next((x for x in rank_impl["Implementadora"].tolist() if x != FACCHINI), FACCHINI)

sel_comp = alt.selection_point(fields=["Implementadora"], name="SEL_COMP", clear="dblclick")

colL, colR = st.columns([1.35, 1.0], gap="large")

with colL:
    top10 = rank_impl.head(10).copy()
    # ranking neutro (cinza) + seleção destacada
    dfp = top10.copy()
    dfp["val_fmt"] = dfp["Volume"].apply(fmt_int)
    dfp["__y"] = dfp["Implementadora"].astype(str) + "  ·  " + dfp["val_fmt"]

    axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)

    ch = (
        alt.Chart(dfp)
        .mark_bar()
        .encode(
            x=alt.X("Volume:Q", title=None),
            y=alt.Y("__y:N", sort="-x", axis=axis_right),
            tooltip=[
                alt.Tooltip("Implementadora:N", title="Implementadora"),
                alt.Tooltip("Volume:Q", title="Volume", format=",.0f"),
            ],
            color=alt.condition(
                "datum.Implementadora == 'FACCHINI'",
                alt.value(C_FAC),          # FACCHINI vermelha
                alt.value(C_COMP),         # todo o resto cinza
            ),
            opacity=alt.condition(sel_comp, alt.value(1), alt.value(0.35)),  # só realça o selecionado
        )
        .add_params(sel_comp)
        .properties(height=420, title="Top 10 por volume (clique para selecionar)")
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )
    st.altair_chart(ch, use_container_width=True, on_select="rerun", key="tp_comp_rank")

picked = selected_value_from_chart_state("tp_comp_rank", "SEL_COMP", "Implementadora")
if not picked:
    picked = st.session_state.get("tp_comp") or default_comp

# se trocar concorrente, limpa recortes abaixo
if picked != st.session_state.get("tp_comp"):
    st.session_state["tp_uf"] = None
    st.session_state["tp_city"] = None
    st.session_state["tp_mix"] = None
    st.session_state["tp_cli"] = None
st.session_state["tp_comp"] = picked
comp = st.session_state["tp_comp"]

with colR:
    market_total = int(df_year["Qtde"].sum())
    fac_total = int(df_year.loc[df_year["Implementadora"].eq(FACCHINI), "Qtde"].sum())
    comp_total = int(df_year.loc[df_year["Implementadora"].eq(comp), "Qtde"].sum())

    fac_share = (fac_total / market_total) if market_total > 0 else 0
    comp_share = (comp_total / market_total) if market_total > 0 else 0

    gap_pp = (comp_share - fac_share)  # proporção
    gap_vol = comp_total - fac_total

    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown(small_metric("Mercado", fmt_int(market_total), f"Ano {ano_sel} • Tipo {tipo_sel}", border_color="dark"), unsafe_allow_html=True)
    with b:
        st.markdown(small_metric("Concorrente", str(comp), "Selecionado", border_color="blue"), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(small_metric("FACCHINI", fmt_int(fac_total), f"Share: {fac_share*100:.1f}%", border_color="red"), unsafe_allow_html=True)
    with c2:
        st.markdown(small_metric(comp, fmt_int(comp_total), f"Share: {comp_share*100:.1f}%", border_color="blue"), unsafe_allow_html=True)

    # azul se concorrente acima, vermelho se facchini acima
    diff_border = "blue" if gap_pp > 0 else "red"
    st.markdown(
        small_metric("Diferença (Concorrente − FACCHINI)", fmt_pp(gap_pp), f"GAP volume: {fmt_int(gap_vol)}", border_color=diff_border),
        unsafe_allow_html=True
    )

# ============================================================
# BLOCO 2 — MAPA (UF): VANTAGEM EM SHARE (AZUL/VERMELHO)
# ============================================================
ui.section("🗺️ Onde o concorrente ganha / onde a FACCHINI ganha (por UF)")

df_uf = calc_vs_fac(df_year, "UF", comp).copy()

max_abs = float(df_uf["Adv"].abs().max()) if not df_uf.empty else 0.0
if max_abs <= 0:
    max_abs = 0.10  # evita escala “quebrada” quando tudo é 0

map_col, side_col = st.columns([1.35, 1.0], gap="large")

with map_col:
    br_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    br_states = alt.Data(url=br_url, format=alt.DataFormat(property="features", type="json"))

    uf_sel_map = alt.selection_point(fields=["properties.sigla"], name="UF_SEL_MAP", clear="dblclick", empty="all")

    color_map = alt.condition(
        "datum.Market === null || datum.Market === 0",
        alt.value(C_ZERO),
        alt.Color(
            "Adv:Q",
            scale=alt.Scale(domain=[-max_abs, 0, max_abs], range=[C_FAC, C_NEU, C_COMP]),
            title=None,
        ),
    )

    mapa = (
        alt.Chart(br_states)
        .mark_geoshape(stroke="#0f172a", strokeWidth=1.3)
        .project(type="mercator")
        .transform_lookup(
            lookup="properties.sigla",
            from_=alt.LookupData(df_uf, "UF", ["Market", "Facchini", "Comp", "Share_Fac", "Share_Comp", "Adv", "GapVol"]),
        )
        .encode(
            tooltip=[
                alt.Tooltip("properties.sigla:N", title="UF"),
                alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
                alt.Tooltip("Comp:Q", title=comp, format=",.0f"),
                alt.Tooltip("Facchini:Q", title="FACCHINI", format=",.0f"),
                alt.Tooltip("Share_Comp:Q", title=f"Share {comp}", format=".1%"),
                alt.Tooltip("Share_Fac:Q", title="Share FACCHINI", format=".1%"),
                alt.Tooltip("Adv:Q", title="Vantagem (pp)", format=".1%"),
                alt.Tooltip("GapVol:Q", title="GAP volume", format=",.0f"),
            ],
            color=color_map,
            opacity=alt.condition(uf_sel_map, alt.value(1), alt.value(0.35)),
        )
        .add_params(uf_sel_map)
        .properties(height=PANEL_H)
        .configure_legend(disable=True)
        .configure_view(stroke=None)
        .configure(background="transparent")
    )

    st.altair_chart(mapa, use_container_width=True, on_select="rerun", key="tp_map")

new_uf = selected_uf_from_chart_state("tp_map", "UF_SEL_MAP")
if new_uf != st.session_state.get("tp_uf"):
    st.session_state["tp_city"] = None
    st.session_state["tp_mix"] = None
    st.session_state["tp_cli"] = None
st.session_state["tp_uf"] = new_uf

with side_col:
    ui.section("📍 Top 15 UFs (vantagem em share)")

    df_rank_uf = df_uf[df_uf["Market"] > 0].sort_values("Adv", ascending=False).head(15).copy()
    sel_uf_bar = alt.selection_point(fields=["UF"], name="SEL_UF_BAR", clear="dblclick")

    ch = rank_bar_value_in_axis(
        df_rank_uf,
        dim_col="UF",
        value_col="Adv",
        sel=sel_uf_bar,
        title="Clique para focar a UF abaixo",
        height=420,
        value_fmt="pp",
    ).add_params(sel_uf_bar)

    st.altair_chart(ch, use_container_width=True, on_select="rerun", key="tp_uf_bar")

clicked_uf = selected_value_from_chart_state("tp_uf_bar", "SEL_UF_BAR", "UF")
if clicked_uf:
    st.session_state["tp_uf"] = clicked_uf

# ============================================================
# BLOCO 3 — CIDADES NA UF EM FOCO
# ============================================================
ui.section("🏙️ Cidades na UF em foco")

uf_focus = st.session_state.get("tp_uf")
df_focus = df_year[df_year["UF"].eq(uf_focus)].copy() if uf_focus else df_year.copy()

st.markdown(
    f"<div style='color:#94a3b8; font-size:0.9rem; margin-top:-6px;'>UF em foco: <b>{uf_focus or 'BRASIL'}</b> • Azul: concorrente acima • Vermelho: FACCHINI acima</div>",
    unsafe_allow_html=True
)

if uf_focus is None:
    st.info("Clique em uma UF no mapa (ou no ranking de UFs) para detalhar as cidades.")
else:
    df_city = calc_vs_fac(df_focus, "Municipio", comp)
    df_city = df_city[df_city["Market"] > 0].sort_values("Adv", ascending=False).head(15).copy()

    sel_city = alt.selection_point(fields=["Municipio"], name="SEL_CITY", clear="dblclick")
    ch = rank_bar_value_in_axis(
        df_city,
        dim_col="Municipio",
        value_col="Adv",
        sel=sel_city,
        title="Top 15 Cidades (vantagem em share) — clique para aplicar no restante",
        height=520,
        value_fmt="pp",
    ).add_params(sel_city)

    st.altair_chart(ch, use_container_width=True, on_select="rerun", key="tp_city_bar")

picked_city = selected_value_from_chart_state("tp_city_bar", "SEL_CITY", "Municipio")
if picked_city != st.session_state.get("tp_city"):
    st.session_state["tp_mix"] = None
    st.session_state["tp_cli"] = None
st.session_state["tp_city"] = picked_city

# contexto (UF + cidade)
df_ctx = df_focus.copy()
if st.session_state.get("tp_city"):
    df_ctx = df_ctx[df_ctx["Municipio"].eq(st.session_state["tp_city"])]

# ============================================================
# BLOCO 4 — MIX + CLIENTES (sempre vs FACCHINI)
# ============================================================
ui.section("🧩 Mix e Clientes (vantagem em share)")

c1, c2 = st.columns([1.0, 1.0], gap="large")

with c1:
    df_mix = calc_vs_fac(df_ctx, "Mix Produto", comp)
    df_mix = df_mix[df_mix["Market"] > 0].sort_values("Adv", ascending=False).head(15).copy()

    sel_mix = alt.selection_point(fields=["Mix Produto"], name="SEL_MIX", clear="dblclick")

    ch = rank_bar_value_in_axis(
        df_mix,
        dim_col="Mix Produto",
        value_col="Adv",
        sel=sel_mix,
        title="Top 15 Mix — clique para filtrar Clientes",
        height=520,
        value_fmt="pp",
    ).add_params(sel_mix)

    st.altair_chart(ch, use_container_width=True, on_select="rerun", key="tp_mix_bar")

picked_mix = selected_value_from_chart_state("tp_mix_bar", "SEL_MIX", "Mix Produto")
st.session_state["tp_mix"] = picked_mix

df_ctx2 = df_ctx.copy()
if st.session_state.get("tp_mix"):
    df_ctx2 = df_ctx2[df_ctx2["Mix Produto"].eq(st.session_state["tp_mix"])]

with c2:
    df_cli = calc_vs_fac(df_ctx2, "Cliente", comp)
    df_cli = df_cli[df_cli["Market"] > 0].sort_values("Adv", ascending=False).head(15).copy()

    sel_cli = alt.selection_point(fields=["Cliente"], name="SEL_CLI", clear="dblclick")

    ch = rank_bar_value_in_axis(
        df_cli,
        dim_col="Cliente",
        value_col="Adv",
        sel=sel_cli,
        title="Top 15 Clientes — clique para Raio-X",
        height=520,
        value_fmt="pp",
    ).add_params(sel_cli)

    st.altair_chart(ch, use_container_width=True, on_select="rerun", key="tp_cli_bar")

picked_cli = selected_value_from_chart_state("tp_cli_bar", "SEL_CLI", "Cliente")
st.session_state["tp_cli"] = picked_cli

# ============================================================
# BLOCO 5 — HISTÓRICO (Mercado vs Concorrente vs FACCHINI)
# ============================================================
ui.section("📈 Histórico (anual) — Mercado vs Concorrente vs FACCHINI")

# filtros iguais do topo, exceto ano
df_hist = df.copy()
if tipo_sel != "Consolidado":
    df_hist = df_hist[df_hist["Tipo"].eq(tipo_sel)]
if mix_sel:
    df_hist = df_hist[df_hist["Mix Produto"].isin(mix_sel)]
if uf_filter:
    df_hist = df_hist[df_hist["UF"].isin(uf_filter)]
if ignore_ni == "Sim":
    for col in ["Cliente", "Mix Produto", "Municipio", "Implementadora"]:
        df_hist = df_hist[~df_hist[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

# aplica recortes atuais (UF/cidade/mix/cliente)
if uf_focus:
    df_hist = df_hist[df_hist["UF"].eq(uf_focus)]
if st.session_state.get("tp_city"):
    df_hist = df_hist[df_hist["Municipio"].eq(st.session_state["tp_city"])]
if st.session_state.get("tp_mix"):
    df_hist = df_hist[df_hist["Mix Produto"].eq(st.session_state["tp_mix"])]
if st.session_state.get("tp_cli"):
    df_hist = df_hist[df_hist["Cliente"].eq(st.session_state["tp_cli"])]

hist_market = df_hist.groupby("Ano", as_index=False).agg(Mercado=("Qtde", "sum"))
hist_comp = df_hist[df_hist["Implementadora"].eq(comp)].groupby("Ano", as_index=False).agg(Concorrente=("Qtde", "sum"))
hist_fac = df_hist[df_hist["Implementadora"].eq(FACCHINI)].groupby("Ano", as_index=False).agg(FACCHINI=("Qtde", "sum"))

hist = hist_market.merge(hist_comp, on="Ano", how="left").merge(hist_fac, on="Ano", how="left")
hist["Concorrente"] = hist["Concorrente"].fillna(0)
hist["FACCHINI"] = hist["FACCHINI"].fillna(0)

melt = hist.melt(id_vars=["Ano"], value_vars=["Mercado", "Concorrente", "FACCHINI"], var_name="Série", value_name="Qtde")
melt = melt.sort_values("Ano")

color_scale = alt.Scale(
    domain=["Mercado", "Concorrente", "FACCHINI"],
    range=[C_GRAY, C_COMP, C_FAC],
)

line = (
    alt.Chart(melt)
    .mark_line(point=True)
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
    .properties(height=360)
    .configure_view(stroke=None)
    .configure(background="transparent")
    .configure_axis(grid=False)
)

st.altair_chart(line, use_container_width=True)

# ============================================================
# RAIO-X (opcional)
# ============================================================
with st.expander("📂 Raio-X (linhas do recorte)", expanded=False):
    df_rx = df_ctx2.copy()
    if st.session_state.get("tp_cli"):
        df_rx = df_rx[df_rx["Cliente"].eq(st.session_state["tp_cli"])]

    cols = ["Ano", "Tipo", "UF", "Municipio", "Cliente", "Implementadora", "Mix Produto", "Modelo", "Representante", "Qtde"]
    cols = [c for c in cols if c in df_rx.columns]
    st.dataframe(
        df_rx[cols].sort_values("Qtde", ascending=False).head(2000),
        use_container_width=True,
        hide_index=True
    )
