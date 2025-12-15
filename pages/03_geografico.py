import streamlit as st
import pandas as pd
import altair as alt
import data as dt
import lib as lb
import ui

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="Visão Geográfica", page_icon="🗺️")
ui.apply_style()
ui.header("Visão Geográfica", "Clique no mapa para filtrar UF. Duplo clique para limpar seleções (mapa e gráficos).")

FACCHINI = "FACCHINI"
PANEL_H = 520  # altura única: painel (esq) e mapa (dir)

# ============================================================
# LOAD + PADRONIZAÇÃO
# ============================================================
@st.cache_data(show_spinner=False)
def load_df():
    df = dt.carregar_emplacamento("arquivos/Emplacamento/*.xlsx")
    df = df.rename(columns={"Cidade": "Municipio", "Município": "Municipio"})

    required = ["Ano", "Tipo", "UF", "Municipio", "Qtde", "Implementadora", "Mix Produto", "Cliente", "Representante"]
    for c in required:
        if c not in df.columns:
            df[c] = "N/I"

    df["UF"] = df["UF"].astype(str).str.strip().str.upper()
    df["Municipio"] = df["Municipio"].astype(str).str.strip().str.upper()
    df["Implementadora"] = df["Implementadora"].astype(str).str.strip().str.upper()

    for c in ["Mix Produto", "Cliente", "Representante"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
        df.loc[df[c].eq(""), c] = "N/I"

    df["Qtde"] = pd.to_numeric(df["Qtde"], errors="coerce").fillna(0).astype(int)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").fillna(0).astype(int)
    return df

df = load_df()

# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("geo_uf", None)
st.session_state.setdefault("drill_city", None)
st.session_state.setdefault("drill_impl", None)
st.session_state.setdefault("drill_cli", None)
st.session_state.setdefault("drill_mix", None)

def _normalize_keys(d: dict) -> dict:
    return {str(k).replace("\\", ""): v for k, v in d.items()}

def selected_uf_from_chart_state(chart_key="geo_map", selection_name="UF_Selection"):
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
        if "sigla" in dn:
            return dn["sigla"]
        props = dn.get("properties")
        if isinstance(props, dict) and "sigla" in props:
            return props["sigla"]
        return None

    if isinstance(sel, list) and len(sel) > 0:
        return extract(sel[0])
    if isinstance(sel, dict):
        return extract(sel)
    return None

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
# UI HELPERS: CARD COM ANOS ANTERIORES (mini)
# ============================================================
def _clean_html(s: str) -> str:
    return " ".join(str(s).split())

def _mini_years_html(mini_items, compact=False):
    if not mini_items:
        return ""

    # compact => cards pequenos (não deixa quebrar o número principal)
    max_w = 240 if not compact else 160
    gap = 26 if not compact else 16
    f_year = "0.78rem" if not compact else "0.72rem"
    f_val  = "0.95rem" if not compact else "0.82rem"

    blocks = []
    for y, v in mini_items:
        blocks.append(
            f"""
            <div style="text-align:right; flex:0 0 auto;">
                <div style="font-size:{f_year}; font-weight:800; color:#0f172a; opacity:0.90">{y}</div>
                <div style="font-size:{f_val}; font-weight:900; color:#0f172a; opacity:0.55; margin-top:2px">{v}</div>
            </div>
            """
        )

    # ✅ sem wrap, com largura fixa + hidden => nunca estraga o número grande
    return (
        "<div style='"
        f"display:flex; flex-wrap:nowrap; gap:{gap}px; "
        "justify-content:flex-end; align-items:flex-start; "
        f"max-width:{max_w}px; overflow:hidden; white-space:nowrap;"
        "'>"
        + "".join(blocks)
        + "</div>"
    )

def kpi_card_years(label, value, context_html, mini_items=None, color="red", compact=False):
    border = f"border-left-{color}" if color else ""
    mini_html = _mini_years_html(mini_items or [], compact=compact)

    # ✅ o lado esquerdo tem prioridade (não quebra)
    html = f"""
    <div class="metric-card {border}">
      <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start;">
        <div style="flex:1 1 auto; min-width:0;">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="white-space:nowrap;">{value}</div>
        </div>
        <div style="flex:0 0 auto;">
          {mini_html}
        </div>
      </div>

      <div style="margin-top:18px; padding-top:14px; border-top:1px solid #f1f5f9;">
        <div style="font-size:0.85rem; color:#94a3b8; line-height:1.4">{context_html}</div>
      </div>
    </div>
    """
    return " ".join(html.split())

# ============================================================
# FILTROS (TOPO): Ano, Tipo, Mix Produto, Implementadora
# ============================================================
with st.container():
    f1, f2, f3, f4 = st.columns([1, 1.6, 2.2, 2.2])

    with f1:
        anos = sorted([a for a in df["Ano"].unique() if a > 0], reverse=True)
        ano_sel = st.selectbox("Ano", anos if anos else [0])

    df_base = df[df["Ano"].eq(ano_sel)].copy() if ano_sel else df.copy()

    with f2:
        tipos = sorted([t for t in df_base["Tipo"].unique() if t not in ("", "N/I")])
        tipo_sel = st.selectbox("Tipo", ["Consolidado"] + tipos)

    if tipo_sel != "Consolidado":
        df_base = df_base[df_base["Tipo"].eq(tipo_sel)]

    with f3:
        mix_opts = sorted([m for m in df_base["Mix Produto"].unique() if m not in ("", "N/I")])
        mix_sel = st.multiselect("Mix Produto", mix_opts, default=[])

    with f4:
        impl_opts = sorted([i for i in df_base["Implementadora"].unique() if i not in ("", "N/I")])
        impl_sel = st.multiselect("Implementadora", impl_opts, default=[])

df_f = df_base.copy()
if mix_sel:
    df_f = df_f[df_f["Mix Produto"].isin(mix_sel)]
if impl_sel:
    df_f = df_f[df_f["Implementadora"].isin([x.upper() for x in impl_sel])]

# ============================================================
# AGREGAÇÕES
# ============================================================
def agg_dim(df_base: pd.DataFrame, dim: str, topn: int = 20) -> pd.DataFrame:
    if df_base.empty:
        return pd.DataFrame(columns=[dim, "Total", "Facchini", "Share", "Oportunidade"])

    g = df_base.groupby(dim, as_index=False).agg(
        Total=("Qtde", "sum"),
        Facchini=("Qtde", lambda x: x[df_base.loc[x.index, "Implementadora"].eq(FACCHINI)].sum()),
    )
    g["Share"] = (g["Facchini"] / g["Total"]).fillna(0)
    g["Oportunidade"] = ((1 - g["Share"]) * g["Total"]).fillna(0)
    return g.sort_values("Total", ascending=False).head(topn)

def format_filters_html(uf_sel):
    chips = [
        f"Ano: <b>{ano_sel}</b>",
        f"Tipo: <b>{tipo_sel}</b>",
        f"UF: <b>{uf_sel or 'BRASIL'}</b>",
    ]
    if mix_sel:
        chips.append(f"Mix: <b>{len(mix_sel)}</b> selecionado(s)")
    if impl_sel:
        chips.append(f"Implementadora: <b>{len(impl_sel)}</b> selecionada(s)")
    if st.session_state.get("drill_city"):
        chips.append(f"Cidade: <b>{st.session_state['drill_city']}</b>")
    if st.session_state.get("drill_impl"):
        chips.append(f"Drill Implementadora: <b>{st.session_state['drill_impl']}</b>")
    if st.session_state.get("drill_cli"):
        chips.append(f"Drill Cliente: <b>{st.session_state['drill_cli']}</b>")
    if st.session_state.get("drill_mix"):
        chips.append(f"Drill Mix: <b>{st.session_state['drill_mix']}</b>")
    return "<br>".join(chips) + "<br><span style='color:#94a3b8'>Duplo clique em qualquer gráfico para limpar a seleção</span>"

def bar_chart(df_in: pd.DataFrame, y_col: str, sel, title: str, height: int = 520):
    if df_in.empty:
        return None
    return (
        alt.Chart(df_in)
        .mark_bar()
        .encode(
            x=alt.X("Total:Q", title=None),
            y=alt.Y(f"{y_col}:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip(f"{y_col}:N", title=y_col),
                alt.Tooltip("Total:Q", format=",.0f", title="Mercado"),
                alt.Tooltip("Facchini:Q", format=",.0f", title="Facchini"),
                alt.Tooltip("Share:Q", format=".1%", title="Share Facchini"),
            ],
            color=alt.condition(sel, alt.value("#dc2626"), alt.value("#9ca3af")),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )

# ============================================================
# LAYOUT: PAINEL (ESQ) | MAPA (DIR) — MESMA ALTURA
# ============================================================
cards_col, map_col = st.columns([1.25, 1.0], gap="large")

# ----------------------------
# MAPA (DIREITA)
# ----------------------------
with map_col:
    ui.section("🗺️ Mapa (UF)")

    df_mapa = df_f.groupby("UF", as_index=False).agg(Total=("Qtde", "sum"))

    br_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    br_states = alt.Data(url=br_url, format=alt.DataFormat(property="features", type="json"))

    sel_uf = alt.selection_point(fields=["properties.sigla"], name="UF_Selection", clear="dblclick", empty="all")

    mapa = (
        alt.Chart(br_states)
        .mark_geoshape(stroke="#0f172a", strokeWidth=1.3)
        .project(type="mercator")
        .transform_lookup(
            lookup="properties.sigla",
            from_=alt.LookupData(df_mapa, "UF", ["Total"]),
        )
        .encode(
            tooltip=[
                alt.Tooltip("properties.sigla:N", title="UF"),
                alt.Tooltip("Total:Q", format=",.0f", title="Volume"),
            ],
            # zero / null => cinza; senão vermelho
            color=alt.condition(
                "datum.Total === null || datum.Total === 0",
                alt.value("#e5e7eb"),
                alt.Color("Total:Q", scale=alt.Scale(scheme="reds"), title=None),
            ),
            opacity=alt.condition(sel_uf, alt.value(1), alt.value(0.35)),
        )
        .add_params(sel_uf)
        .properties(height=PANEL_H)
        .configure_legend(disable=True)
        .configure_view(stroke=None)
        .configure(background="transparent")
    )

    st.altair_chart(mapa, use_container_width=True, on_select="rerun", key="geo_map")

# sincroniza UF do mapa
new_uf = selected_uf_from_chart_state("geo_map", "UF_Selection")
prev_uf = st.session_state.get("geo_uf")
st.session_state["geo_uf"] = new_uf

# reseta drilldowns se trocar UF
if prev_uf != new_uf:
    st.session_state["drill_city"] = None
    st.session_state["drill_impl"] = None
    st.session_state["drill_cli"] = None
    st.session_state["drill_mix"] = None

uf_sel = st.session_state["geo_uf"]
df_geo = df_f[df_f["UF"].eq(uf_sel)].copy() if uf_sel else df_f.copy()

# ============================================================
# MINI ANOS (para cards): aplica filtros iguais, exceto "Ano"
# (inclui UF e drilldowns atuais do session_state)
# ============================================================
df_scope = df.copy()

if tipo_sel != "Consolidado":
    df_scope = df_scope[df_scope["Tipo"].eq(tipo_sel)]
if mix_sel:
    df_scope = df_scope[df_scope["Mix Produto"].isin(mix_sel)]
if impl_sel:
    df_scope = df_scope[df_scope["Implementadora"].isin([x.upper() for x in impl_sel])]
if uf_sel:
    df_scope = df_scope[df_scope["UF"].eq(uf_sel)]

# drilldowns (vêm do estado anterior; após clique, atualiza no próximo rerun)
if st.session_state.get("drill_city"):
    df_scope = df_scope[df_scope["Municipio"].eq(st.session_state["drill_city"])]
if st.session_state.get("drill_impl"):
    df_scope = df_scope[df_scope["Implementadora"].eq(st.session_state["drill_impl"])]
if st.session_state.get("drill_cli"):
    df_scope = df_scope[df_scope["Cliente"].eq(st.session_state["drill_cli"])]
if st.session_state.get("drill_mix"):
    df_scope = df_scope[df_scope["Mix Produto"].eq(st.session_state["drill_mix"])]

total_by_year = df_scope.groupby("Ano")["Qtde"].sum()
fac_by_year = df_scope.loc[df_scope["Implementadora"].eq(FACCHINI)].groupby("Ano")["Qtde"].sum()
share_by_year = (fac_by_year / total_by_year).fillna(0)

prev_years = [ano_sel - 1, ano_sel - 2, ano_sel - 3]
mini_total = [(str(y), lb.formatar_br(int(total_by_year.get(y, 0)))) for y in prev_years]
mini_fac = [(str(y), lb.formatar_br(int(fac_by_year.get(y, 0)))) for y in prev_years]
mini_share = [(str(y), f"{(share_by_year.get(y, 0) * 100):.1f}%") for y in prev_years]

# ============================================================
# CARDS (ESQUERDA) — EXATAMENTE (TOTAL / FACCHINI / SHARE) + mini anos
# ============================================================
with cards_col:
    ui.section("📌 Resumo")

    vol_total = int(df_geo["Qtde"].sum())
    vol_fac = int(df_geo.loc[df_geo["Implementadora"].eq(FACCHINI), "Qtde"].sum()) if not df_geo.empty else 0
    share = (vol_fac / vol_total) if vol_total > 0 else 0

    panel_html = (
        f"<div style='height:{PANEL_H}px; display:flex; flex-direction:column; gap:12px;'>"
        f"{kpi_card_years('VOLUME TOTAL', lb.formatar_br(vol_total), format_filters_html(uf_sel), mini_items=mini_total, color='red', compact=False)}"
        "<div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>"
        f"{kpi_card_years('FACCHINI', lb.formatar_br(vol_fac), f'UF: <b>{uf_sel or 'BRASIL'}</b>', mini_items=mini_fac, color='red', compact=True)}"
        f"{kpi_card_years('SHARE FACCHINI', f'{share*100:.1f}%', f'{lb.formatar_br(vol_fac)} / {lb.formatar_br(vol_total)}', mini_items=mini_share, color='red', compact=True)}"
        "</div>"
        "</div>"
    )
    st.markdown(panel_html, unsafe_allow_html=True)

# ============================================================
# EXPLORAÇÃO (layout premium: Score -> Detalhe -> Decomposição)
# ============================================================
st.write("---")

# -----------------------------
# Helpers visuais (cards pequenos)
# -----------------------------
def small_metric_card(label: str, value: str, sub: str = "", color="red"):
    border = f"border-left-{color}" if color else ""
    html = f"""
    <div class="metric-card {border}" style="padding:18px; border-radius:14px;">
        <div class="metric-label" style="margin-bottom:8px;">{label}</div>
        <div class="metric-value" style="font-size:2.2rem; letter-spacing:-1px;">{value}</div>
        <div style="margin-top:10px; color:#94a3b8; font-size:0.85rem;">{sub}</div>
    </div>
    """
    return " ".join(html.split())

def _fmt_int(x): 
    try: return lb.formatar_br(int(x))
    except: return "0"

def _fmt_pct(x):
    try: return f"{float(x)*100:.1f}%"
    except: return "0.0%"

# ============================================================
# BLOCO 1 — SCORE (Holofote) [SEM slider TopN + COM rolagem]
# ============================================================
with st.expander("🔥 Score (holofote)", expanded=True):
    ui.section("🔥 Score (holofote)")

    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-top:-6px;'>"
        "<b>Score</b> = (1 − Share Facchini) × Mercado &nbsp;•&nbsp; Clique nas barras ou nos pontos para ver detalhes"
        "</div>",
        unsafe_allow_html=True
    )

    # dimensão padrão do score: UF no Brasil, Município quando UF selecionada
    default_dim = "Municipio" if uf_sel else "UF"
    dim_opts = [default_dim, "Cliente", "Mix Produto"]
    dim_opts = list(dict.fromkeys(dim_opts))

    c_dim, c_ni = st.columns([2.2, 1.0], gap="medium")

    with c_dim:
        dim_sel = st.selectbox("Analisar Score por", dim_opts, index=0)

    with c_ni:
        ignore_ni = st.radio(
            "Cliente N/I",
            ["Incluir", "Ignorar"],
            horizontal=True,
            index=1,  # default: Ignorar
            label_visibility="visible",
        )

    TOPN_SCORE = 60          # pega mais itens pra rolar
    VISIBLE_ROWS = 15        # “top 15 visíveis”
    ROW_H = 28               # altura por linha/barra
    SCROLL_H = VISIBLE_ROWS * ROW_H + 80  # + folga de título

    score_base = df_geo.copy()

    if ignore_ni == "Ignorar" and "Cliente" in score_base.columns:
        score_base = score_base[~score_base["Cliente"].astype(str).str.strip().isin(["N/I", "NI", "N\\I", "", "None", "nan"])]
        
    score_df = (
        agg_dim(score_base, dim_sel, topn=999999)
        .sort_values("Oportunidade", ascending=False)
        .head(TOPN_SCORE)
        .copy()
    )

    if score_df.empty:
        st.info("Sem dados no recorte atual para calcular Score.")
        picked_val = None
    else:
        score_pick = alt.selection_point(fields=[dim_sel], name="SCORE_PICK", clear="dblclick")

        # ----------- RANKING (ESQ) com rolagem -----------
        bar_h = max(420, len(score_df) * ROW_H)  # altura real (vai rolar no container)

        bar = (
            alt.Chart(score_df)
            .mark_bar()
            .encode(
                x=alt.X("Oportunidade:Q", title=None),
                y=alt.Y(f"{dim_sel}:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip(f"{dim_sel}:N", title=dim_sel),
                    alt.Tooltip("Total:Q", title="Mercado", format=",.0f"),
                    alt.Tooltip("Facchini:Q", title="Facchini", format=",.0f"),
                    alt.Tooltip("Share:Q", title="Share", format=".1%"),
                    alt.Tooltip("Oportunidade:Q", title="Score", format=",.0f"),
                ],
                color=alt.condition(score_pick, alt.value("#dc2626"), alt.value("#9ca3af")),
                opacity=alt.condition(score_pick, alt.value(1), alt.value(0.55)),
            )
            .add_params(score_pick)
            .properties(height=bar_h, title="Ranking por Score (role para ver mais)")
            .configure_view(stroke=None)
            .configure(background="transparent")
            .configure_axis(grid=False)
        )

        # ----------- SCATTER (DIR) -----------
        scatter = (
            alt.Chart(score_df)
            .mark_circle(size=140)
            .encode(
                x=alt.X("Total:Q", title="Mercado"),
                y=alt.Y("Share:Q", title="Share Facchini", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip(f"{dim_sel}:N", title=dim_sel),
                    alt.Tooltip("Total:Q", title="Mercado", format=",.0f"),
                    alt.Tooltip("Facchini:Q", title="Facchini", format=",.0f"),
                    alt.Tooltip("Share:Q", title="Share", format=".1%"),
                    alt.Tooltip("Oportunidade:Q", title="Score", format=",.0f"),
                ],
                color=alt.condition(score_pick, alt.value("#dc2626"), alt.value("#9ca3af")),
                opacity=alt.condition(score_pick, alt.value(1), alt.value(0.55)),
            )
            .add_params(score_pick)
            .properties(height=520, title="Volume × Share")
            .configure_view(stroke=None)
            .configure(background="transparent")
            .configure_axis(grid=False)
        )

        left, right = st.columns([1.35, 1.0], gap="large")

        with left:
            # container com altura fixa => aparece “top 15” e rola para ver o resto
            with st.container(height=SCROLL_H, border=False):
                st.altair_chart(bar, use_container_width=True, on_select="rerun", key="score_bar")

        with right:
            st.altair_chart(scatter, use_container_width=True, on_select="rerun", key="score_scatter")

        # lê seleção (bar ou scatter)
        picked_val = selected_value_from_chart_state("score_bar", "SCORE_PICK", dim_sel)
        if not picked_val:
            picked_val = selected_value_from_chart_state("score_scatter", "SCORE_PICK", dim_sel)

    # ============================================================
    # BLOCO 2 — DETALHE DO SELECIONADO + COMPOSIÇÃO
    # ============================================================

    ui.section("🔎 Detalhes do selecionado")

    if not picked_val:
        st.info("Clique em uma barra/ponto no Score para ver detalhes e composição.")
    else:
        # recorte do selecionado (dentro do df_geo)
        df_sel = score_base[score_base[dim_sel].astype(str) == str(picked_val)].copy()

        sel_total = int(df_sel["Qtde"].sum())
        sel_fac = int(df_sel.loc[df_sel["Implementadora"].eq(FACCHINI), "Qtde"].sum())
        sel_share = (sel_fac / sel_total) if sel_total > 0 else 0
        sel_score = (1 - sel_share) * sel_total
        sel_gap = sel_total - sel_fac

        k1, k2, k3, k4, k5 = st.columns([1.1, 1, 1, 1, 1], gap="medium")
        with k1:
            st.markdown(small_metric_card(dim_sel, str(picked_val), "Item selecionado", color="red"), unsafe_allow_html=True)
        with k2:
            st.markdown(small_metric_card("Mercado", _fmt_int(sel_total), "Qtde total no recorte", color="red"), unsafe_allow_html=True)
        with k3:
            st.markdown(small_metric_card("Facchini", _fmt_int(sel_fac), "Qtde Facchini no recorte", color="red"), unsafe_allow_html=True)
        with k4:
            st.markdown(small_metric_card("Share", f"{sel_share*100:.1f}%", "Facchini / Mercado", color="red"), unsafe_allow_html=True)
        with k5:
            st.markdown(small_metric_card("GAP", _fmt_int(sel_gap), "Mercado − Facchini", color="red"), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        comp1, comp2 = st.columns([1, 1], gap="large")

        with comp1:
            # composição por implementadora
            ui.section("🏭 Composição por Implementadora")
            comp_impl = (
                df_sel.groupby("Implementadora", as_index=False)["Qtde"].sum()
                .sort_values("Qtde", ascending=False)
                .head(12)
                .rename(columns={"Qtde": "Total"})
            )
            if comp_impl.empty:
                st.info("Sem dados de Implementadora nesse selecionado.")
            else:
                comp_impl["Pct"] = (comp_impl["Total"] / comp_impl["Total"].sum()).fillna(0)
                ch = (
                    alt.Chart(comp_impl)
                    .mark_bar()
                    .encode(
                        x=alt.X("Total:Q", title=None),
                        y=alt.Y("Implementadora:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("Implementadora:N", title="Implementadora"),
                            alt.Tooltip("Total:Q", title="Qtde", format=",.0f"),
                            alt.Tooltip("Pct:Q", title="%", format=".1%"),
                        ],
                        color=alt.value("#9ca3af"),
                    )
                    .properties(height=320)
                    .configure_view(stroke=None)
                    .configure(background="transparent")
                    .configure_axis(grid=False)
                )
                st.altair_chart(ch, use_container_width=True)

        with comp2:
            # composição por mix
            ui.section("🧩 Composição por Mix Produto")
            comp_mix = (
                df_sel.groupby("Mix Produto", as_index=False)["Qtde"].sum()
                .sort_values("Qtde", ascending=False)
                .head(12)
                .rename(columns={"Qtde": "Total"})
            )
            if comp_mix.empty:
                st.info("Sem dados de Mix nesse selecionado.")
            else:
                comp_mix["Pct"] = (comp_mix["Total"] / comp_mix["Total"].sum()).fillna(0)
                ch = (
                    alt.Chart(comp_mix)
                    .mark_bar()
                    .encode(
                        x=alt.X("Total:Q", title=None),
                        y=alt.Y("Mix Produto:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("Mix Produto:N", title="Mix"),
                            alt.Tooltip("Total:Q", title="Qtde", format=",.0f"),
                            alt.Tooltip("Pct:Q", title="%", format=".1%"),
                        ],
                        color=alt.value("#9ca3af"),
                    )
                    .properties(height=320)
                    .configure_view(stroke=None)
                    .configure(background="transparent")
                    .configure_axis(grid=False)
                )
                st.altair_chart(ch, use_container_width=True)


def bar_rank(df_in: pd.DataFrame, dim_col: str, sel, title: str, height: int):
    if df_in.empty:
        return None

    dfp = df_in.copy()
    dfp["Total_fmt"] = dfp["Total"].apply(lambda x: lb.formatar_br(int(x)) if pd.notna(x) else "0")

    # ✅ “rótulo no fim” via eixo Y na DIREITA: NOME · 12.345
    dfp["__y"] = dfp[dim_col].astype(str) + "  ·  " + dfp["Total_fmt"]

    axis_right = alt.Axis(
        orient="right",
        title=None,
        labelFontSize=12,
        ticks=False,
        domain=False,
        labelPadding=10,
    )

    return (
        alt.Chart(dfp)
        .mark_bar()
        .encode(
            x=alt.X("Total:Q", title=None),
            y=alt.Y("__y:N", sort="-x", axis=axis_right),
            tooltip=[
                alt.Tooltip(f"{dim_col}:N", title=dim_col),
                alt.Tooltip("Total:Q", title="Mercado", format=",.0f"),
                alt.Tooltip("Facchini:Q", title="Facchini", format=",.0f"),
                alt.Tooltip("Share:Q", title="Share", format=".1%"),
            ],
            color=alt.condition(sel, alt.value("#dc2626"), alt.value("#9ca3af")),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.55)),
        )
        .add_params(sel)
        .properties(height=height, title=title)
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(grid=False)
    )

with st.expander("📌 Decomposição rápida (clique para filtrar)", expanded=True):

    # só os controles essenciais (sem opções extras)
    cA, cB = st.columns([2.2, 1.2], gap="medium")
    with cA:
        dim3 = st.selectbox("3º painel", ["Implementadora", "Mix Produto"], index=0)

    with cB:
        ignore_ni_decomp = st.radio("Ignorar N/I", ["Não", "Sim"], horizontal=True, index=1)

    df_dec = df_geo.copy()

    # ignora N/I
    if ignore_ni_decomp == "Sim":
        for col in ["Municipio", "Cliente", "Implementadora", "Mix Produto"]:
            if col in df_dec.columns:
                df_dec = df_dec[~df_dec[col].astype(str).str.strip().isin(["N/I", "NI", "", "None", "nan"])]

    # cross-filter (seleções atuais)
    if st.session_state.get("drill_city"):
        df_dec = df_dec[df_dec["Municipio"].eq(st.session_state["drill_city"])]
    if st.session_state.get("drill_cli"):
        df_dec = df_dec[df_dec["Cliente"].eq(st.session_state["drill_cli"])]
    if st.session_state.get("drill_impl"):
        df_dec = df_dec[df_dec["Implementadora"].eq(st.session_state["drill_impl"])]
    if st.session_state.get("drill_mix"):
        df_dec = df_dec[df_dec["Mix Produto"].eq(st.session_state["drill_mix"])]

    TOPN = 40
    VISIBLE = 15
    ROW_H = 28
    scroll_h = VISIBLE * ROW_H + 85
    chart_h = max(420, TOPN * ROW_H)

    c1, c2, c3 = st.columns(3, gap="large")

    # Painel 1: Cidades
    with c1:
        st.markdown("##### 🏙️ Cidades")
        if uf_sel is None:
            st.info("Selecione uma UF no mapa.")
        else:
            df_city = agg_dim(df_dec, "Municipio", topn=TOPN)
            sel = alt.selection_point(fields=["Municipio"], name="CITY_SEL", clear="dblclick")
            ch = bar_rank(df_city, "Municipio", sel, "Top Cidades", chart_h)
            with st.container(height=scroll_h, border=False):
                st.altair_chart(ch, use_container_width=True, on_select="rerun", key="chart_city")

    # Painel 2: Clientes
    with c2:
        st.markdown("##### 🏢 Clientes")
        df_cli = agg_dim(df_dec, "Cliente", topn=TOPN)
        sel = alt.selection_point(fields=["Cliente"], name="CLI_SEL", clear="dblclick")
        ch = bar_rank(df_cli, "Cliente", sel, "Top Clientes", chart_h)
        with st.container(height=scroll_h, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key="chart_cli")

    # Painel 3: Implementadoras ou Mix
    with c3:
        title = "🏭 Implementadoras" if dim3 == "Implementadora" else "🧩 Mix Produto"
        st.markdown(f"##### {title}")

        df_dim3 = agg_dim(df_dec, dim3, topn=TOPN)
        sel_name = "IMPL_SEL" if dim3 == "Implementadora" else "MIX_SEL"
        key_name = "chart_impl" if dim3 == "Implementadora" else "chart_mix"

        sel = alt.selection_point(fields=[dim3], name=sel_name, clear="dblclick")
        ch = bar_rank(df_dim3, dim3, sel, f"Top {dim3}", chart_h)
        with st.container(height=scroll_h, border=False):
            st.altair_chart(ch, use_container_width=True, on_select="rerun", key=key_name)

# sincroniza drilldowns
st.session_state["drill_city"] = selected_value_from_chart_state("chart_city", "CITY_SEL", "Municipio")
st.session_state["drill_cli"]  = selected_value_from_chart_state("chart_cli",  "CLI_SEL",  "Cliente")
st.session_state["drill_impl"] = selected_value_from_chart_state("chart_impl", "IMPL_SEL", "Implementadora")
st.session_state["drill_mix"]  = selected_value_from_chart_state("chart_mix",  "MIX_SEL",  "Mix Produto")