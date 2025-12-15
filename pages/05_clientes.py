import streamlit as st
import pandas as pd
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
    "Panorama → Priorizar → Diagnóstico. Clique em um cliente para ver UF → Mix → Implementadoras e histórico."
)

FACCHINI = "FACCHINI"

# Paleta fixa (FACCHINI vermelho, concorrente/líder azul)
C_FAC = "#dc2626"
C_COMP = "#2563eb"
C_GRAY = "#9ca3af"
C_NEU = "#f1f5f9"
C_ZERO = "#e5e7eb"

PANEL_H = 520

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
st.session_state.setdefault("cli_focus", None)
st.session_state.setdefault("cli_uf_focus", None)
st.session_state.setdefault("cli_mix_focus", None)

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

def fmt_int(x) -> str:
    try:
        return lb.formatar_br(int(x))
    except Exception:
        return "0"

def fmt_pct(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "0.0%"

def fmt_pp(x) -> str:
    try:
        return f"{float(x)*100:+.1f} pp"
    except Exception:
        return "+0.0 pp"

# ============================================================
# UI: cards pequenos
# ============================================================
def small_metric(label: str, value: str, sub: str = "", border_color="dark"):
    html = f"""
    <div class="metric-card border-left-{border_color}" style="padding:18px; border-radius:14px;">
        <div class="metric-label" style="margin-bottom:8px;">{label}</div>
        <div class="metric-value" style="font-size:2.2rem; letter-spacing:-1px; white-space:nowrap;">{value}</div>
        <div style="margin-top:10px; color:#94a3b8; font-size:0.85rem;">{sub}</div>
    </div>
    """
    return " ".join(html.split())

# ============================================================
# FILTROS TOPO (poucos e diretos)
# ============================================================
with st.container():
    f1, f2, f3, f4, f5 = st.columns([1.0, 1.6, 2.2, 1.7, 1.2], gap="medium")

    with f1:
        anos = sorted([a for a in df["Ano"].unique() if a > 0], reverse=True)
        ano_sel = st.selectbox("Ano base", anos if anos else [0])

    df_scope = df[df["Ano"].eq(ano_sel)].copy() if ano_sel else df.copy()

    with f2:
        tipos = sorted([t for t in df_scope["Tipo"].unique() if t not in ("", "N/I")])
        tipo_sel = st.selectbox("Tipo", ["Consolidado"] + tipos)

    if tipo_sel != "Consolidado":
        df_scope = df_scope[df_scope["Tipo"].eq(tipo_sel)]

    with f3:
        mix_opts = sorted([m for m in df_scope["Mix Produto"].unique() if m not in ("", "N/I")])
        mix_sel = st.multiselect("Mix Produto (opcional)", mix_opts, default=[])

    if mix_sel:
        df_scope = df_scope[df_scope["Mix Produto"].isin(mix_sel)]

    with f4:
        uf_opts = sorted([u for u in df_scope["UF"].unique() if u not in ("", "N/I")])
        uf_filter = st.multiselect("UF (opcional)", uf_opts, default=[])

    if uf_filter:
        df_scope = df_scope[df_scope["UF"].isin(uf_filter)]

    with f5:
        ignore_ni = st.radio("Ignorar N/I", ["Não", "Sim"], horizontal=True, index=1)

if ignore_ni == "Sim":
    for col in ["Cliente", "Mix Produto", "Municipio", "Implementadora"]:
        if col in df_scope.columns:
            df_scope = df_scope[~df_scope[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

if df_scope.empty:
    st.warning("Sem dados no recorte atual.")
    st.stop()

# ============================================================
# MODELO DE CLIENTE: Mercado vs FACCHINI vs Concorrente #1
# ============================================================
def build_clients_view(df_base: pd.DataFrame) -> pd.DataFrame:
    # Mercado e FACCHINI por cliente
    m = df_base.groupby("Cliente", as_index=False).agg(Market=("Qtde", "sum"))
    f = (
        df_base[df_base["Implementadora"].eq(FACCHINI)]
        .groupby("Cliente", as_index=False)
        .agg(Facchini=("Qtde", "sum"))
    )

    # Top concorrente por cliente (exclui FACCHINI)
    comp_base = df_base[~df_base["Implementadora"].eq(FACCHINI)]
    comp = (
        comp_base.groupby(["Cliente", "Implementadora"], as_index=False)
        .agg(CompVol=("Qtde", "sum"))
        .sort_values(["Cliente", "CompVol"], ascending=[True, False])
    )
    topc = comp.drop_duplicates("Cliente", keep="first").rename(
        columns={"Implementadora": "Comp1", "CompVol": "Comp1Vol"}
    )

    out = m.merge(f, on="Cliente", how="left").merge(topc, on="Cliente", how="left")
    out["Facchini"] = out["Facchini"].fillna(0)
    out["Comp1"] = out["Comp1"].fillna("—")
    out["Comp1Vol"] = out["Comp1Vol"].fillna(0)

    out["Share_Fac"] = (out["Facchini"] / out["Market"]).fillna(0)
    out["Share_Comp1"] = (out["Comp1Vol"] / out["Market"]).fillna(0)

    # Leader do cliente (FACCHINI ou concorrente #1)
    out["Leader"] = out.apply(
        lambda r: FACCHINI if r["Facchini"] >= r["Comp1Vol"] else str(r["Comp1"]),
        axis=1
    )
    out["LeaderColor"] = out["Leader"].apply(lambda x: "FAC" if x == FACCHINI else "COMP")

    # Potenciais
    out["Gap"] = (out["Market"] - out["Facchini"]).clip(lower=0)
    out["Score_Ataque"] = ((1 - out["Share_Fac"]) * out["Market"]).fillna(0)

    return out

df_clients = build_clients_view(df_scope)

# default cliente foco
if st.session_state.get("cli_focus") not in set(df_clients["Cliente"].astype(str)):
    # pega o maior Score (mais intuitivo que só Gap)
    st.session_state["cli_focus"] = str(df_clients.sort_values("Score_Ataque", ascending=False).iloc[0]["Cliente"])
    st.session_state["cli_uf_focus"] = None
    st.session_state["cli_mix_focus"] = None

# ============================================================
# CHART HELPERS (single chart, valor no eixo)
# ============================================================
def rank_bar_axis_value(df_in: pd.DataFrame, dim: str, val: str, sel, title: str, height: int, fmt="int", color_field=None):
    if df_in.empty:
        return None

    dfp = df_in.copy()
    if fmt == "pp":
        dfp["_lab"] = dfp[val].apply(fmt_pp)
    elif fmt == "pct":
        dfp["_lab"] = dfp[val].apply(fmt_pct)
    else:
        dfp["_lab"] = dfp[val].apply(fmt_int)

    dfp["__y"] = dfp[dim].astype(str) + "  ·  " + dfp["_lab"].astype(str)

    axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)

    # cor: se tiver campo categórico (ex: FAC/COMP), usa escala; senão cinza + destaque seleção
    if color_field and color_field in dfp.columns:
        color_enc = alt.Color(
            f"{color_field}:N",
            scale=alt.Scale(domain=["FAC", "COMP"], range=[C_FAC, C_COMP]),
            legend=None
        )
        opacity_enc = alt.condition(sel, alt.value(1), alt.value(0.55))
    else:
        color_enc = alt.value(C_GRAY)
        opacity_enc = alt.condition(sel, alt.value(1), alt.value(0.55))

    ch = (
        alt.Chart(dfp)
        .mark_bar()
        .encode(
            x=alt.X(f"{val}:Q", title=None),
            y=alt.Y("__y:N", sort="-x", axis=axis_right),
            tooltip=[
                alt.Tooltip(f"{dim}:N", title=dim),
                alt.Tooltip("Market:Q", title="Mercado", format=",.0f") if "Market" in dfp.columns else alt.Tooltip(f"{val}:Q", title=val),
                alt.Tooltip("Facchini:Q", title="FACCHINI", format=",.0f") if "Facchini" in dfp.columns else alt.Tooltip(f"{val}:Q", title=val),
                alt.Tooltip("Share_Fac:Q", title="Share FACCHINI", format=".1%") if "Share_Fac" in dfp.columns else alt.Tooltip(f"{val}:Q", title=val),
                alt.Tooltip("Comp1:N", title="Concorrente #1") if "Comp1" in dfp.columns else alt.Tooltip(f"{val}:Q", title=val),
                alt.Tooltip("Share_Comp1:Q", title="Share Conc. #1", format=".1%") if "Share_Comp1" in dfp.columns else alt.Tooltip(f"{val}:Q", title=val),
                alt.Tooltip("Leader:N", title="Líder") if "Leader" in dfp.columns else alt.Tooltip(f"{val}:Q", title=val),
            ],
            color=color_enc,
            opacity=opacity_enc,
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )
    return ch

def scatter_clients(df_in: pd.DataFrame, sel, title: str, height: int = 520):
    if df_in.empty:
        return None

    dfp = df_in.copy()
    dfp["Color"] = dfp["LeaderColor"]  # FAC/COMP

    color_scale = alt.Scale(domain=["FAC", "COMP"], range=[C_FAC, C_COMP])

    ch = (
        alt.Chart(dfp)
        .mark_circle()
        .encode(
            x=alt.X("Market:Q", title="Mercado (Qtde)"),
            y=alt.Y("Share_Fac:Q", title="Share FACCHINI", axis=alt.Axis(format="%")),
            size=alt.Size("Market:Q", legend=None),
            color=alt.Color("Color:N", scale=color_scale, legend=None),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("Cliente:N", title="Cliente"),
                alt.Tooltip("Market:Q", title="Mercado", format=",.0f"),
                alt.Tooltip("Facchini:Q", title="FACCHINI", format=",.0f"),
                alt.Tooltip("Share_Fac:Q", title="Share FACCHINI", format=".1%"),
                alt.Tooltip("Comp1:N", title="Concorrente #1"),
                alt.Tooltip("Comp1Vol:Q", title="Conc. #1 (Qtde)", format=",.0f"),
                alt.Tooltip("Share_Comp1:Q", title="Share Conc. #1", format=".1%"),
                alt.Tooltip("Leader:N", title="Líder"),
                alt.Tooltip("Gap:Q", title="GAP captável", format=",.0f"),
                alt.Tooltip("Score_Ataque:Q", title="Score Ataque", format=",.0f"),
            ],
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )
    return ch

# ============================================================
# 1) PANORAMA (visão global)
# ============================================================
ui.section("🌍 Panorama (visão global)")

market_total = int(df_scope["Qtde"].sum())
fac_total = int(df_scope.loc[df_scope["Implementadora"].eq(FACCHINI), "Qtde"].sum())
fac_share = (fac_total / market_total) if market_total > 0 else 0
pot_total = int(max(0, market_total - fac_total))
n_clients = int((df_clients["Market"] > 0).sum())

k1, k2, k3, k4 = st.columns([1, 1, 1, 1], gap="medium")
with k1:
    st.markdown(small_metric("Mercado", fmt_int(market_total), f"Ano {ano_sel} • Tipo {tipo_sel}", border_color="dark"), unsafe_allow_html=True)
with k2:
    st.markdown(small_metric("FACCHINI", fmt_int(fac_total), f"Share: {fac_share*100:.1f}%", border_color="red"), unsafe_allow_html=True)
with k3:
    st.markdown(small_metric("Potencial captável", fmt_int(pot_total), "Mercado − FACCHINI", border_color="blue"), unsafe_allow_html=True)
with k4:
    st.markdown(small_metric("Clientes ativos", fmt_int(n_clients), "Clientes com volume > 0", border_color="dark"), unsafe_allow_html=True)

# Scatter (cap top N p/ performance)
df_scatter = df_clients.sort_values("Market", ascending=False).head(400).copy()
sel_sc = alt.selection_point(fields=["Cliente"], name="CLI_SCAT_SEL", clear="dblclick")

ch_sc = scatter_clients(df_scatter, sel_sc, "Clientes: Mercado × Share FACCHINI (vermelho=FACCHINI líder | azul=concorrente líder)", height=PANEL_H)
st.altair_chart(ch_sc, use_container_width=True, on_select="rerun", key="cli_scatter")

clicked = selected_value_from_chart_state("cli_scatter", "CLI_SCAT_SEL", "Cliente")
if clicked:
    if str(clicked) != st.session_state.get("cli_focus"):
        st.session_state["cli_uf_focus"] = None
        st.session_state["cli_mix_focus"] = None
    st.session_state["cli_focus"] = str(clicked)

# ============================================================
# 2) PRIORIZAR (onde atirar)
# ============================================================
st.write("---")
ui.section("🎯 Priorizar (onde atirar)")

# Top por GAP e por Score (ambos mostram “valor no eixo”)
TOPN = 40
VISIBLE = 15
ROW_H = 26
scroll_h = VISIBLE * ROW_H + 95
chart_h = max(420, TOPN * ROW_H)

df_gap = df_clients.sort_values("Gap", ascending=False).head(TOPN).copy()
df_score = df_clients.sort_values("Score_Ataque", ascending=False).head(TOPN).copy()

# mantém cor por líder (FAC/COMP)
df_gap["ColorRank"] = df_gap["LeaderColor"]
df_score["ColorRank"] = df_score["LeaderColor"]

cL, cR = st.columns([1, 1], gap="large")

with cL:
    st.markdown("##### 🔥 Top GAP (captura rápida)")
    sel_gap = alt.selection_point(fields=["Cliente"], name="CLI_GAP_SEL", clear="dblclick")
    ch = rank_bar_axis_value(
        df_gap,
        dim="Cliente",
        val="Gap",
        sel=sel_gap,
        title="GAP = Mercado − FACCHINI (clique para focar cliente)",
        height=chart_h,
        fmt="int",
        color_field="ColorRank",
    )
    with st.container(height=scroll_h, border=False):
        st.altair_chart(ch, use_container_width=True, on_select="rerun", key="cli_rank_gap")

with cR:
    st.markdown("##### 🧠 Top Score (grande + share baixo)")
    sel_score = alt.selection_point(fields=["Cliente"], name="CLI_SCORE_SEL", clear="dblclick")
    ch = rank_bar_axis_value(
        df_score,
        dim="Cliente",
        val="Score_Ataque",
        sel=sel_score,
        title="Score = (1 − Share FACCHINI) × Mercado (clique para focar cliente)",
        height=chart_h,
        fmt="int",
        color_field="ColorRank",
    )
    with st.container(height=scroll_h, border=False):
        st.altair_chart(ch, use_container_width=True, on_select="rerun", key="cli_rank_score")

clicked_gap = selected_value_from_chart_state("cli_rank_gap", "CLI_GAP_SEL", "Cliente")
clicked_score = selected_value_from_chart_state("cli_rank_score", "CLI_SCORE_SEL", "Cliente")
picked_from_rank = clicked_gap or clicked_score
if picked_from_rank:
    if str(picked_from_rank) != st.session_state.get("cli_focus"):
        st.session_state["cli_uf_focus"] = None
        st.session_state["cli_mix_focus"] = None
    st.session_state["cli_focus"] = str(picked_from_rank)

# ============================================================
# 3) DIAGNÓSTICO (cliente em foco)
# ============================================================
st.write("---")
ui.section("🔎 Diagnóstico (cliente em foco)")

focus_client = st.session_state.get("cli_focus")
if not focus_client:
    focus_client = str(df_clients.sort_values("Score_Ataque", ascending=False).iloc[0]["Cliente"])
    st.session_state["cli_focus"] = focus_client

df_c = df_scope[df_scope["Cliente"].astype(str) == str(focus_client)].copy()

# concorrente líder do cliente (exclui FACCHINI)
tmp = (
    df_c[~df_c["Implementadora"].eq(FACCHINI)]
    .groupby("Implementadora", as_index=False)["Qtde"].sum()
    .sort_values("Qtde", ascending=False)
)
comp = str(tmp.iloc[0]["Implementadora"]) if not tmp.empty else "—"

market_c = int(df_c["Qtde"].sum())
fac_c = int(df_c.loc[df_c["Implementadora"].eq(FACCHINI), "Qtde"].sum())
comp_c = int(df_c.loc[df_c["Implementadora"].eq(comp), "Qtde"].sum()) if comp != "—" else 0

share_fac_c = (fac_c / market_c) if market_c > 0 else 0
share_comp_c = (comp_c / market_c) if market_c > 0 else 0
gap_c = max(0, market_c - fac_c)
score_c = (1 - share_fac_c) * market_c

k1, k2, k3, k4, k5 = st.columns([1.35, 1, 1, 1, 1], gap="medium")
with k1:
    st.markdown(small_metric("Cliente", str(focus_client), "Em foco", border_color="dark"), unsafe_allow_html=True)
with k2:
    st.markdown(small_metric("Mercado", fmt_int(market_c), "Qtde no recorte", border_color="dark"), unsafe_allow_html=True)
with k3:
    st.markdown(small_metric("FACCHINI", fmt_int(fac_c), f"Share: {share_fac_c*100:.1f}%", border_color="red"), unsafe_allow_html=True)
with k4:
    st.markdown(small_metric("Concorrente #1", comp, f"Share: {share_comp_c*100:.1f}%", border_color="blue"), unsafe_allow_html=True)
with k5:
    st.markdown(small_metric("GAP / Score", fmt_int(gap_c), f"Score: {fmt_int(score_c)}", border_color="blue"), unsafe_allow_html=True)

# ------------------------------------------------------------
# Função: vantagem por dimensão (sempre COMP − FACCHINI)
# ------------------------------------------------------------
def calc_vs_fac_dim(df_base: pd.DataFrame, dim: str, comp_name: str) -> pd.DataFrame:
    if df_base.empty:
        return pd.DataFrame(columns=[dim, "Market", "Facchini", "Comp", "Share_Fac", "Share_Comp", "Adv"])

    m = df_base.groupby(dim, as_index=False).agg(Market=("Qtde", "sum"))
    f = (
        df_base[df_base["Implementadora"].eq(FACCHINI)]
        .groupby(dim, as_index=False)
        .agg(Facchini=("Qtde", "sum"))
    )
    c = (
        df_base[df_base["Implementadora"].eq(comp_name)]
        .groupby(dim, as_index=False)
        .agg(Comp=("Qtde", "sum"))
    ) if comp_name != "—" else pd.DataFrame({dim: [], "Comp": []})

    out = m.merge(f, on=dim, how="left").merge(c, on=dim, how="left")
    out["Facchini"] = out["Facchini"].fillna(0)
    out["Comp"] = out["Comp"].fillna(0)

    out["Share_Fac"] = (out["Facchini"] / out["Market"]).fillna(0)
    out["Share_Comp"] = (out["Comp"] / out["Market"]).fillna(0)
    out["Adv"] = out["Share_Comp"] - out["Share_Fac"]   # + => COMP ganha (azul), - => FACCHINI ganha (vermelho)
    out["Sign"] = out["Adv"].apply(lambda x: "COMP" if x >= 0 else "FAC")
    return out

# ------------------------------------------------------------
# 3 painéis: UF / Mix / Implementadoras (single charts)
# ------------------------------------------------------------
cA, cB, cC = st.columns([1, 1, 1], gap="large")

# UF (clicável)
with cA:
    st.markdown("##### 📍 Onde (UF) — vantagem em share")
    df_uf = calc_vs_fac_dim(df_c, "UF", comp)
    df_uf = df_uf[df_uf["Market"] > 0].copy()
    df_uf["Abs"] = df_uf["Adv"].abs()
    df_uf = df_uf.sort_values("Abs", ascending=False).head(40)

    sel_uf = alt.selection_point(fields=["UF"], name="CLI_UF_SEL", clear="dblclick")
    df_uf["ColorRank"] = df_uf["Sign"]

    ch = rank_bar_axis_value(
        df_uf,
        dim="UF",
        val="Adv",
        sel=sel_uf,
        title="Vantagem (pp) = Share Conc. − Share FACCHINI",
        height=max(420, len(df_uf) * 22),
        fmt="pp",
        color_field="ColorRank",
    )

    with st.container(height=520, border=False):
        st.altair_chart(ch, use_container_width=True, on_select="rerun", key="cli_uf_adv")

    uf_clicked = selected_value_from_chart_state("cli_uf_adv", "CLI_UF_SEL", "UF")
    if uf_clicked != st.session_state.get("cli_uf_focus"):
        st.session_state["cli_mix_focus"] = None
    st.session_state["cli_uf_focus"] = uf_clicked

# Mix (clicável e respeita UF foco)
with cB:
    st.markdown("##### 🧩 Em que (Mix) — vantagem em share")
    df_mix_base = df_c.copy()
    if st.session_state.get("cli_uf_focus"):
        df_mix_base = df_mix_base[df_mix_base["UF"].eq(st.session_state["cli_uf_focus"])]

    df_mix = calc_vs_fac_dim(df_mix_base, "Mix Produto", comp)
    df_mix = df_mix[df_mix["Market"] > 0].copy()
    df_mix["Abs"] = df_mix["Adv"].abs()
    df_mix = df_mix.sort_values("Abs", ascending=False).head(40)

    sel_mix = alt.selection_point(fields=["Mix Produto"], name="CLI_MIX_SEL", clear="dblclick")
    df_mix["ColorRank"] = df_mix["Sign"]

    ch = rank_bar_axis_value(
        df_mix,
        dim="Mix Produto",
        val="Adv",
        sel=sel_mix,
        title="Vantagem (pp) por Mix (respeita UF em foco)",
        height=max(420, len(df_mix) * 22),
        fmt="pp",
        color_field="ColorRank",
    )

    with st.container(height=520, border=False):
        st.altair_chart(ch, use_container_width=True, on_select="rerun", key="cli_mix_adv")

    mix_clicked = selected_value_from_chart_state("cli_mix_adv", "CLI_MIX_SEL", "Mix Produto")
    st.session_state["cli_mix_focus"] = mix_clicked

# Implementadoras (não precisa seleção; só mostra “contra quem”)
with cC:
    st.markdown("##### 🏭 Contra quem (Implementadoras)")
    df_impl_base = df_c.copy()
    if st.session_state.get("cli_uf_focus"):
        df_impl_base = df_impl_base[df_impl_base["UF"].eq(st.session_state["cli_uf_focus"])]
    if st.session_state.get("cli_mix_focus"):
        df_impl_base = df_impl_base[df_impl_base["Mix Produto"].eq(st.session_state["cli_mix_focus"])]

    impl = (
        df_impl_base.groupby("Implementadora", as_index=False)["Qtde"].sum()
        .rename(columns={"Qtde": "Volume"})
        .sort_values("Volume", ascending=False)
        .head(25)
        .copy()
    )

    if impl.empty:
        st.info("Sem dados no recorte.")
    else:
        # cor por papel (sem condition aninhado)
        def role(x: str) -> str:
            x = str(x)
            if x == FACCHINI:
                return "FAC"
            if x == comp:
                return "COMP"
            return "OTH"

        impl["Role"] = impl["Implementadora"].apply(role)
        impl["_lab"] = impl["Volume"].apply(fmt_int)
        impl["__y"] = impl["Implementadora"].astype(str) + "  ·  " + impl["_lab"].astype(str)

        color_scale = alt.Scale(
            domain=["FAC", "COMP", "OTH"],
            range=[C_FAC, C_COMP, C_GRAY]
        )
        axis_right = alt.Axis(orient="right", title=None, labelFontSize=12, ticks=False, domain=False, labelPadding=10)

        ch = (
            alt.Chart(impl)
            .mark_bar()
            .encode(
                x=alt.X("Volume:Q", title=None),
                y=alt.Y("__y:N", sort="-x", axis=axis_right),
                color=alt.Color("Role:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Implementadora:N", title="Implementadora"),
                    alt.Tooltip("Volume:Q", title="Volume", format=",.0f"),
                ],
            )
            .properties(height=max(420, len(impl) * 22), title="FACCHINI (vermelho) | Concorrente #1 (azul)")
            .configure_view(stroke=None)
            .configure(background="transparent")
            .configure_axis(grid=False)
        )

        with st.container(height=520, border=False):
            st.altair_chart(ch, use_container_width=True)

# ============================================================
# HISTÓRICO (sempre Mercado vs FACCHINI vs Concorrente #1)
# ============================================================
st.write("---")
ui.section("📈 Histórico (anual) — Mercado vs FACCHINI vs Concorrente #1")

# filtros iguais ao topo, exceto ano
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

# aplica recortes do foco
df_hist = df_hist[df_hist["Cliente"].astype(str) == str(focus_client)]
if st.session_state.get("cli_uf_focus"):
    df_hist = df_hist[df_hist["UF"].eq(st.session_state["cli_uf_focus"])]
if st.session_state.get("cli_mix_focus"):
    df_hist = df_hist[df_hist["Mix Produto"].eq(st.session_state["cli_mix_focus"])]

hist_market = df_hist.groupby("Ano", as_index=False).agg(Mercado=("Qtde", "sum"))
hist_fac = df_hist[df_hist["Implementadora"].eq(FACCHINI)].groupby("Ano", as_index=False).agg(FACCHINI=("Qtde", "sum"))
hist_comp = df_hist[df_hist["Implementadora"].eq(comp)].groupby("Ano", as_index=False).agg(Concorrente=("Qtde", "sum")) if comp != "—" else pd.DataFrame({"Ano": [], "Concorrente": []})

hist = hist_market.merge(hist_fac, on="Ano", how="left").merge(hist_comp, on="Ano", how="left")
hist["FACCHINI"] = hist["FACCHINI"].fillna(0)
hist["Concorrente"] = hist["Concorrente"].fillna(0)

melt = hist.melt(id_vars=["Ano"], value_vars=["Mercado", "FACCHINI", "Concorrente"], var_name="Série", value_name="Qtde").sort_values("Ano")

color_scale = alt.Scale(
    domain=["Mercado", "FACCHINI", "Concorrente"],
    range=[C_GRAY, C_FAC, C_COMP],
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
    df_rx = df_c.copy()
    if st.session_state.get("cli_uf_focus"):
        df_rx = df_rx[df_rx["UF"].eq(st.session_state["cli_uf_focus"])]
    if st.session_state.get("cli_mix_focus"):
        df_rx = df_rx[df_rx["Mix Produto"].eq(st.session_state["cli_mix_focus"])]

    cols = ["Ano", "Tipo", "UF", "Municipio", "Cliente", "Implementadora", "Mix Produto", "Modelo", "Representante", "Qtde"]
    cols = [c for c in cols if c in df_rx.columns]
    st.dataframe(
        df_rx[cols].sort_values("Qtde", ascending=False).head(2000),
        use_container_width=True,
        hide_index=True
    )
