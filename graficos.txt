# graficos.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional
import lib as lb
import matplotlib.patheffects as pe

def _load_img(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return plt.imread(str(p))
    return None

def grafico_share_facchini_generico(
    tabela: pd.DataFrame,
    ano_atual: int,
    *,
    # colunas
    col_total: str = "Total_Geral",
    col_fac: str = "Total_Facchini",
    # legenda
    label_fac: str = "FACCHINI",
    label_outros: str = "Mercado Geral",
    # cores
    cor_azul: str = "#0B66C3",
    cor_vermelho: str = "#E10600",
    cor_branco: str = "#FFFFFF",
    # imagens (passe arrays do plt.imread ou deixe None e informe os paths)
    logo_img=None,
    seta_up_img=None,
    seta_down_img=None,
    logo_path: Optional[str] = None,
    seta_up_path: Optional[str] = None,
    seta_down_path: Optional[str] = None,
    # escala/posicionamento
    logo_zoom: float = 0.22,
    fract_dentro_azul: float = 0.8,
    seta_zoom: float = 0.06,
    seta_shift: tuple[float, float] = (0, 0),
    pct_ref: str = "azul",          # "azul" ou "seta"
    pct_pos: str = "above",         # "above" ou "below"
    pct_dist: float = 18,
    figsize: tuple[int, int] = (14, 7),
) -> plt.Figure:
    """
    Gera o gráfico de barras empilhadas (Mercado x FACCHINI) com:
    - valores no bloco azul (total)
    - valores FAC no bloco vermelho
    - share da FAC no topo vermelho
    - setas + variação YoY
    - sem título (padrão). Se quiser título, faça ax.set_title(...) fora da função.

    A 'tabela' deve ter índice = anos e as colunas col_total e col_fac.
    """

    # carrega imagens por caminho, se não vieram em memória
    if logo_img is None:
        logo_img = _load_img(logo_path)
    if seta_up_img is None:
        seta_up_img = _load_img(seta_up_path)
    if seta_down_img is None:
        seta_down_img = _load_img(seta_down_path)

    # ======== DADOS ========
    totais = tabela[col_total].astype(float).copy()
    fac    = tabela[col_fac].astype(float).reindex(totais.index, fill_value=0)
    outros = (totais - fac).clip(lower=0)
    anos   = totais.index.to_list()

    # métricas
    share_fac = np.where(totais.values > 0, (fac.values / totais.values) * 100, np.nan)
    var_tot   = np.r_[np.nan, np.diff(totais.values) / np.where(totais.values[:-1] == 0, np.nan, totais.values[:-1]) * 100]

    # ======== ESTILO ========
    plt.rcParams.update({
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # ======== BARRAS ========
    ax.bar(anos, outros, color=cor_azul, edgecolor="none")
    ax.bar(anos, fac, bottom=outros, color=cor_vermelho, edgecolor="none")

    # ======== RÓTULOS ========
    for i, ano in enumerate(anos):
        y_outros = outros.iloc[i]
        y_fac    = fac.iloc[i]
        y_total  = totais.iloc[i]
        topo     = y_outros + y_fac

        # dentro do azul: valor total
        if y_outros > 0:
            ax.text(
                ano, y_outros/2,
                f"{int(round(y_total,0)):,}".replace(",", "."),
                ha="center", va="center", fontsize=13, fontweight="bold", color=cor_branco
            )

        # faixa vermelha: valor FAC + share + logo
        if y_fac > 0:
            ax.text(
                ano, y_outros + y_fac*0.4,
                f"{int(round(y_fac,0)):,}".replace(",", "."),
                ha="center", va="center", fontsize=13, fontweight="bold", color=cor_branco
            )
            ax.annotate(
                f"{share_fac[i]:.1f}%",
                xy=(ano, topo),
                xytext=(5, -1),
                textcoords="offset points",
                ha="left", va="top",
                fontsize=7, fontweight="bold", color=cor_branco,
                clip_on=True
            )
            if logo_img is not None:
                ab = AnnotationBbox(
                    OffsetImage(logo_img, zoom=logo_zoom),
                    (ano, topo),
                    frameon=False,
                    boxcoords="offset points",
                    xybox=(0, 1),
                    box_alignment=(0.6, 0.0)
                )
                ax.add_artist(ab)

    # ======== VARIAÇÃO YoY + SETAS ========
    for i, ano in enumerate(anos):
        if i == 0 or np.isnan(var_tot[i]):
            continue

        y_outros = outros.iloc[i]
        y_total  = totais.iloc[i]
        y_anchor = fract_dentro_azul * (y_outros if y_outros > 0 else y_total)

        up        = var_tot[i] >= 0
        seta_img  = seta_up_img if up else seta_down_img

        # seta (se existir imagem)
        if seta_img is not None:
            ab = AnnotationBbox(
                OffsetImage(seta_img, zoom=seta_zoom),
                (ano, y_anchor),
                frameon=False,
                xybox=seta_shift,
                boxcoords="offset points",
                box_alignment=(0.5, 0.5),
                clip_on=True
            )
            ax.add_artist(ab)

        # referência para o texto %
        if pct_ref.lower() == "azul":
            y_ref_pct = (y_outros/2) if y_outros > 0 else (0.5 * y_total)
        else:
            y_ref_pct = y_anchor

        if pct_pos.lower() == "below":
            va_pct = "top"
            dy     = +pct_dist
        else:
            va_pct = "bottom"
            dy     = -pct_dist

        ax.annotate(
            f"{var_tot[i]:+,.1f}%".replace(",", "."),
            xy=(ano, y_ref_pct),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center", va=va_pct,
            fontsize=10, fontweight="bold",
            color=cor_branco,
            clip_on=True
        )

    # ======== LEGENDA / EIXOS ========
    leg_outros = Patch(facecolor=cor_azul, label=label_outros)
    leg_fac    = Patch(facecolor=cor_vermelho, label=label_fac)
    ax.legend(handles=[leg_fac, leg_outros], loc="lower center", frameon=False,
              bbox_to_anchor=(0.5, -0.15), ncol=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(anos)
    ax.set_xticklabels(
        [f"PREV - {ano_atual}" if a == ano_atual else str(a) for a in anos],
        fontproperties=fm.FontProperties(weight='bold')
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.margins(x=0.05, y=0.12)

    fig.tight_layout()
    return fig

# def grafico_producao_caminhao(
#     valores_prod_ano,           # pd.Series: índice = anos, valores = produção
#     variacao_prod_ano,          # pd.Series: índice = anos, valores = var. % ano vs. ano
#     titulo="Produção Caminhões 2014 a 2025",
#     figsize=(14, 6),
#     cor_barras="#ff0000",
#     y_max=200000,
#     y_step=20000,
#     renomear_previsao_ano=2025, # se o ano existir, vira "PREV - 2025"
#     mostrar_gradiente=True,     # fundo radial cinza claro
# ):
#     """
#     Retorna (fig, ax) com o gráfico de barras da produção de caminhões por ano,
#     exibindo rótulos de valores e variação percentual acima das barras.
#     """

#     # --- prepara dados ---
#     # garante ordem por ano
#     valores_prod_ano = valores_prod_ano.sort_index()
#     variacao_prod_ano = variacao_prod_ano.reindex(valores_prod_ano.index)

#     # formatação "187.002"
#     fmt_milhar_pt = lambda n: f"{int(round(n)):,}".replace(",", ".")

#     # anos e valores
#     anos = [str(a) for a in valores_prod_ano.index.tolist()]
#     valores = valores_prod_ano.values.astype(float)

#     # variação: primeiro None, demais arredondados
#     variacao = [None] + list(np.round(variacao_prod_ano.iloc[1:].values).astype("float"))

#     # renomear ano de previsão (se existir)
#     if str(renomear_previsao_ano) in anos:
#         i = anos.index(str(renomear_previsao_ano))
#         anos[i] = f"PREV - {renomear_previsao_ano}"

#     # --- figura/axis ---
#     fig, ax = plt.subplots(figsize=figsize)

#     # fundo com gradiente radial (na figura inteira)
#     if mostrar_gradiente:
#         fig.canvas.draw()  # garante dimensões atualizadas
#         fig_w, fig_h = fig.canvas.get_width_height()
#         x = np.linspace(-1, 1, fig_w)
#         y = np.linspace(-1, 1, fig_h)
#         xx, yy = np.meshgrid(x, y)
#         radial = np.sqrt(xx**2 + yy**2)
#         radial = (radial / radial.max()) ** 2.2

#         base = 0.99  # claro no centro
#         amp  = 0.5   # quão escura é a borda
#         bg = np.clip(base - amp * radial, 0, 1)

#         fig.patch.set_alpha(0)
#         ax.set_facecolor((1, 1, 1, 0))
#         fig.figimage(bg, xo=0, yo=0, cmap="gray", vmin=0, vmax=1, zorder=-10)

#     # barras
#     barras = ax.bar(range(len(anos)), valores, color=cor_barras, edgecolor=cor_barras, width=0.72)

#     # título
#     ax.set_title(titulo, fontsize=18, fontweight='bold', pad=16)

#     # eixo X
#     ax.set_xticks(range(len(anos)))
#     ax.set_xticklabels(anos, rotation=0, fontsize=11)

#     # eixo Y
#     ax.set_ylim(0, y_max)
#     yticks = np.arange(0, y_max + y_step, y_step)
#     ax.set_yticks(yticks)
#     ax.set_yticklabels([fmt_milhar_pt(v) for v in yticks], fontsize=10)

#     # limpar bordas / grid
#     ax.grid(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # rótulos (valor e %)
#     for rect, valor, var in zip(barras, valores, variacao):
#         x = rect.get_x() + rect.get_width() / 2
#         y = rect.get_height()

#         # valor acima da barra
#         ax.text(
#             x, y + 3500, fmt_milhar_pt(valor),
#             ha='center', va='bottom', fontsize=12, fontweight='bold', color="#333333"
#         )

#         # variação % (se houver)
#         if var is not None and not np.isnan(var):
#             ax.text(
#                 x, y + 18000, f"{int(var)}%",
#                 ha='center', va='bottom', fontsize=12, fontweight='bold', color="#333333"
#             )

#     # espaçamentos
#     plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.13)

#     return fig, ax

def grafico_producao_caminhao(
    valores_prod_ano,           # pd.Series: índice = anos, valores = produção (anual)
    variacao_prod_ano=None,     # pd.Series ou None -> recalcula
    titulo="Produção Caminhões 2014 a 2025",
    figsize=(14, 6),
    cor_barras="#ff0000",
    y_max=200000,
    y_step=20000,
    renomear_previsao_ano=None, # ex: 2025 -> vira PREV - 2025
    mostrar_gradiente=True,
    df=None,                    # DataFrame mensal bruto (com colunas Ano, Data, Produção)
):
    """
    Gera gráfico de produção anual de caminhões.
    Se df for passado, calcula a previsão para o último ano (YTD + média últimos 3 meses * meses faltantes).
    """

    # --- Série anual base ---
    valores_prod_ano = valores_prod_ano.sort_index()

    # --- previsão do último ano ---
    if df is not None and renomear_previsao_ano is not None:
        df_ano = df[df["Ano"] == renomear_previsao_ano].copy()
        if not df_ano.empty:
            ytd = df_ano["Produção"].sum()
            mes_max = df_ano["Data"].dt.month.max()
            ultimos3 = df_ano.sort_values("Data").tail(3)["Produção"]
            media_ult3 = ultimos3.mean()
            faltam = 12 - mes_max
            prev_total = ytd + media_ult3 * faltam

            # injeta previsão na série anual
            valores_prod_ano = valores_prod_ano.copy()
            valores_prod_ano.loc[renomear_previsao_ano] = prev_total

    # --- recalcula variação se não veio pronto ---
    variacao = (
        valores_prod_ano.pct_change() * 100 if variacao_prod_ano is None
        else variacao_prod_ano.reindex(valores_prod_ano.index)
    )

    # --- formatações ---
    fmt_milhar_pt = lambda n: f"{int(round(n)):,}".replace(",", ".")
    fmt_pct = lambda p: f"{int(round(p))}%"

    anos = valores_prod_ano.index.tolist()
    valores = valores_prod_ano.values.astype(float)

    # renomeia último ano se for previsão
    xticklabels = [str(a) for a in anos]
    if renomear_previsao_ano in anos:
        idx = anos.index(renomear_previsao_ano)
        xticklabels[idx] = f"PREV - {renomear_previsao_ano}"

    # --- figura ---
    fig, ax = plt.subplots(figsize=figsize)

    # fundo com gradiente radial
    if mostrar_gradiente:
        fig.canvas.draw()
        fw, fh = fig.canvas.get_width_height()
        x = np.linspace(-1, 1, fw)
        y = np.linspace(-1, 1, fh)
        xx, yy = np.meshgrid(x, y)
        radial = np.sqrt(xx**2 + yy**2)
        radial = (radial / radial.max()) ** 2.2
        base = 0.99
        amp  = 0.5
        bg = np.clip(base - amp * radial, 0, 1)
        fig.patch.set_alpha(0)
        ax.set_facecolor((1,1,1,0))
        fig.figimage(bg, xo=0, yo=0, cmap="gray", vmin=0, vmax=1, zorder=-10)

    # barras
    barras = ax.bar(range(len(anos)), valores, color=cor_barras, width=0.72)

    # título
    ax.set_title(titulo, fontsize=18, fontweight="bold", pad=16)

    # eixo X
    ax.set_xticks(range(len(anos)))
    ax.set_xticklabels(xticklabels, fontsize=11)

    # eixo Y
    ax.set_ylim(0, y_max)
    yticks = np.arange(0, y_max+y_step, y_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt_milhar_pt(v) for v in yticks], fontsize=10)

    # limpar bordas
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # rótulos
    for rect, valor, var in zip(barras, valores, variacao):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height()
        ax.text(x, y+3500, fmt_milhar_pt(valor),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        if pd.notna(var):
            ax.text(x, y+18000, fmt_pct(var),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.13)
    return fig, ax


# constantes (use as mesmas que você já tem no módulo)
MARCAS = ["FACCHINI","GUERRA","LIBRELATO","NOMA","RANDON","ROSSETTI"]
CORES  = {
    "FACCHINI":"#E10600", "GUERRA":"#82C3FF", "LIBRELATO":"#2E7BE6",
    "NOMA":"#334E8A", "RANDON":"#5BA3E3", "ROSSETTI":"#1FA650", "OUTROS":"#8C8C8C"
}

def plot_share_area(
    df_base: pd.DataFrame,
    ano_atual: int,
    *,
    tipo: str | list[str] | None = None,
    col_tipo: str = "Tipo",
    previsao_media3m=None,
    figsize: tuple[int,int] = (18, 10),
):
    if previsao_media3m is None:
        raise ValueError("Passe a função de previsão via previsao_media3m=...")

    # ---- filtro por tipo ----
    if tipo is None:
        df_filt = df_base
    else:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        df_filt = df_base[df_base[col_tipo].astype(str).isin(tipos)].copy()

    # ---- base saneada ----
    df1 = df_filt.copy()
    df1["Qtde"] = pd.to_numeric(df1["Qtde"], errors="coerce").fillna(0)
    df1["Ano"]  = pd.to_numeric(df1["Ano"],  errors="coerce").astype(int)
    df1["Implementadora"] = df1["Implementadora"].astype(str).str.upper().str.strip()

    total_ano = df1.groupby("Ano")["Qtde"].sum().sort_index()

    por_marca = (
        df1[df1["Implementadora"].isin(MARCAS)]
        .groupby(["Ano","Implementadora"])["Qtde"].sum()
        .unstack("Implementadora")
        .reindex(total_ano.index, fill_value=0)
        .astype(float)
    )
    for m in MARCAS:
        if m not in por_marca.columns:
            por_marca[m] = 0.0
    por_marca = por_marca[MARCAS]

    # ---- previsão pro ano_atual ----
    prev_total = previsao_media3m(df1, ano_atual, implementadora=None)
    prev_marca = {m: previsao_media3m(df1, ano_atual, implementadora=m) for m in MARCAS}

    if ano_atual not in total_ano.index:
        total_ano.loc[ano_atual] = 0
    total_ano.loc[ano_atual] = prev_total

    if ano_atual not in por_marca.index:
        por_marca.loc[ano_atual] = 0.0
    for m in MARCAS:
        por_marca.loc[ano_atual, m] = prev_marca[m]

    total_ano = total_ano.sort_index()
    por_marca = por_marca.sort_index()

    # ---- shares + OUTROS ----
    shares = por_marca.div(total_ano.replace(0, np.nan), axis=0) * 100.0
    shares["OUTROS"] = (100.0 - shares.sum(axis=1)).clip(lower=0)

    # >>> ORDEM: maiores embaixo com base no ano mais recente
    anos = list(shares.index.astype(int))
    ultimo_ano = anos[-1]
    order = (
        shares.drop(columns=["OUTROS"])
            .loc[ultimo_ano].fillna(0)
            .sort_values(ascending=False)
            .index.tolist()
    )

    # OUTROS na BASE (embaixo)
    cols = ["OUTROS"] + order
    shares = shares[cols]

    # ---- plot ----
    plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold", "axes.titlesize": 24})

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # e no plot use SEMPRE shares[cols]
    y = shares[cols].to_numpy().T
    ax.stackplot(anos, y, colors=[CORES[c] for c in cols], linewidth=0)

    ax.set_xlim(min(anos), max(anos))
    ax.set_ylim(-3, 100)             # folga para labels
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks(anos)
    ax.set_xticklabels([f"PREV - {ano_atual}" if a == ano_atual else str(a) for a in anos],
                       fontproperties=fm.FontProperties(weight='bold'))
    ax.set_yticks([])
    for s in ["top","right","left"]:
        ax.spines[s].set_visible(False)

    # legenda mais abaixo + margem inferior maior
    handles = [Patch(facecolor=CORES[c], label=c) for c in cols]
    ax.legend(handles=handles, loc="lower center", frameon=False,
              bbox_to_anchor=(0.5, -0.12), ncol=min(8, len(cols)))
    plt.subplots_adjust(bottom=0.22)

    # rótulos com proteção de borda
    TOP_ROOM = 1.2
    min_label = 1.2
    vals = shares[cols].to_numpy()
    cum = np.zeros(len(anos))
    te = [pe.withStroke(linewidth=2, foreground="black", alpha=0.35)]

    for j, name in enumerate(cols):
        seg = vals[:, j]
        centers = cum + seg/2
        for i, ano in enumerate(anos):
            v = seg[i]
            if np.isnan(v) or v < min_label:
                continue

            # --- posição vertical (proteção topo/rodapé)
            pad = max(0.4, seg[i] * 0.06)
            upper_cap = 100 - TOP_ROOM if name == "OUTROS" else (cum[i] + seg[i] - pad)
            y_pos = np.clip(centers[i], cum[i] + pad, upper_cap)

            # --- deslocamento horizontal em pontos (mantém "dentro")
            if i == 0:
                xoff, ha = +10, "left"     # puxa para dentro no 1º ano
            elif i == len(anos) - 1:
                xoff, ha = -10, "right"    # puxa para dentro no último ano
            else:
                xoff, ha = 0, "center"

            fs = 11 if seg[i] > 3 else 9

            ax.annotate(
                f"{v:.1f}%",
                xy=(ano, y_pos),
                xytext=(xoff, 0), textcoords="offset points",
                ha=ha, va="center",
                fontsize=fs, fontweight="bold",
                color="#FFFFFF", clip_on=True, path_effects=te
            )
        cum += seg

    fig.tight_layout()
    return fig

# def plot_share_area(
#     df_base: pd.DataFrame,
#     ano_atual: int,
#     *,
#     tipo: str | list[str] | None = None,   # None=total; "SC"; "SR"; ["SC","SR"]
#     col_tipo: str = "Tipo",
#     previsao_media3m=None,                 # passe sua função aqui (ex.: lb.previsao_media3m)
#     titulo_base: str = "Distribuição Market Share",
#     figsize: tuple[int,int] = (18, 10),
# ):
#     """
#     Gera o stackplot 100% de market share por marca (+ OUTROS) e retorna fig (sem plt.show()).
#     - Usa previsão para o ano_atual via 'previsao_media3m(df, ano_atual, implementadora=...)'.
#     - Filtra por 'tipo' se informado.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib.patches import Patch
#     from matplotlib import font_manager as fm

#     if previsao_media3m is None:
#         raise ValueError("Passe a função de previsão via previsao_media3m=... (ex.: lb.previsao_media3m)")

#     # ---- filtro por tipo ----
#     if tipo is None:
#         df_filt = df_base
#         rotulo_tipo = "SC + SR"
#     else:
#         tipos = [tipo] if isinstance(tipo, str) else list(tipo)
#         df_filt = df_base[df_base[col_tipo].astype(str).isin(tipos)].copy()
#         rotulo_tipo = " + ".join(tipos)

#     # ---- base saneada ----
#     df1 = df_filt.copy()
#     df1["Qtde"] = pd.to_numeric(df1["Qtde"], errors="coerce").fillna(0)
#     df1["Ano"]  = pd.to_numeric(df1["Ano"],  errors="coerce").astype(int)
#     df1["Implementadora"] = df1["Implementadora"].astype(str).str.upper().str.strip()

#     # totais por ano
#     total_ano = df1.groupby("Ano")["Qtde"].sum().sort_index()

#     # por marca
#     por_marca = (
#         df1[df1["Implementadora"].isin(MARCAS)]
#         .groupby(["Ano","Implementadora"])["Qtde"].sum()
#         .unstack("Implementadora")
#         .reindex(total_ano.index, fill_value=0)
#         .astype(float)
#     )
#     for m in MARCAS:
#         if m not in por_marca.columns:
#             por_marca[m] = 0.0
#     por_marca = por_marca[MARCAS]

#     # ---- previsão pro ano_atual ----
#     prev_total = previsao_media3m(df1, ano_atual, implementadora=None)
#     prev_marca = {m: previsao_media3m(df1, ano_atual, implementadora=m) for m in MARCAS}

#     if ano_atual not in total_ano.index:
#         total_ano.loc[ano_atual] = 0
#     total_ano.loc[ano_atual] = prev_total

#     if ano_atual not in por_marca.index:
#         por_marca.loc[ano_atual] = 0.0
#     for m in MARCAS:
#         por_marca.loc[ano_atual, m] = prev_marca[m]

#     total_ano = total_ano.sort_index()
#     por_marca = por_marca.sort_index()

#     # ---- shares + OUTROS ----
#     shares = por_marca.div(total_ano.replace(0, np.nan), axis=0) * 100.0
#     shares["OUTROS"] = (100.0 - shares.sum(axis=1)).clip(lower=0)
#     shares = shares[MARCAS + ["OUTROS"]]

#     # ---- plot ----
#     plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold", "axes.titlesize": 24})
#     anos = list(shares.index.astype(int))

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_facecolor("white")

#     cols = shares.columns.tolist()
#     y = shares.to_numpy().T
#     ax.stackplot(anos, y, colors=[CORES[c] for c in cols], linewidth=0)

#     ax.set_xlim(min(anos), max(anos))
#     ax.set_ylim(0, 100)
#     ax.set_xlabel(""); ax.set_ylabel("")
#     ax.set_xticks(anos)
#     ax.set_xticklabels([f"PREV - {ano_atual}" if a == ano_atual else str(a) for a in anos],
#                        fontproperties=fm.FontProperties(weight='bold'))
#     ax.set_yticks([])
#     for s in ["top","right","left"]:
#         ax.spines[s].set_visible(False)

#     handles = [Patch(facecolor=CORES[c], label=c) for c in cols]
#     ax.legend(handles=handles, loc="lower center", frameon=False,
#               bbox_to_anchor=(0.5, -0.06), ncol=min(8, len(cols)))

#     # rótulos nas faixas
#     min_label = 1.2
#     vals = shares.to_numpy()
#     cum = np.zeros(len(anos))
#     for j, _name in enumerate(cols):
#         seg = vals[:, j]
#         centers = cum + seg/2
#         for i, ano in enumerate(anos):
#             v = seg[i]
#             if np.isnan(v) or v < min_label:
#                 continue
#             ax.text(ano, centers[i], f"{v:.1f}%",
#                     ha="center", va="center", fontsize=11, fontweight="bold",
#                     color="#FFFFFF", clip_on=True)
#         cum += seg

#     fig.tight_layout()
#     return fig

def tabela_emplacamento_pivot(
    df: pd.DataFrame,
    tipo: str | list[str] | None = None,
    *,
    col_tipo: str = "Tipo",
    col_impl: str = "Implementadora",
    col_ano: str = "Ano",
    col_mes: str = "Mes",
    col_qtd: str = "Qtde",
    top_n: int = 10,
    outros_label: str = "OUTROS",
    ym_ini: int | None = None,   # AAAAMM (inclusive)
    ym_fim: int | None = None,   # AAAAMM (inclusive)
) -> pd.DataFrame:
    """
    Tabela dinâmica estilo Excel (Implementadora x meses MM/AAAA + 'Total Geral'),
    com filtro de ano/tipo e agregação dos demais em 'OUTROS' mantendo apenas top_n linhas.
    """
    df1 = df.copy()

    # saneamento
    df1[col_qtd] = pd.to_numeric(df1[col_qtd], errors="coerce").fillna(0)
    df1[col_ano] = pd.to_numeric(df1[col_ano], errors="coerce").astype("Int64")
    df1[col_mes] = pd.to_numeric(df1[col_mes], errors="coerce").astype("Int64")
    df1[col_impl] = df1[col_impl].astype(str).str.strip()
    df1.loc[df1[col_impl].isin(["", "nan", "None"]), col_impl] = "(vazio)"

    df1['YM'] = (df1[col_ano].astype(int) *100 + df1[col_mes].astype(int))
    if ym_ini is not None or ym_fim is not None:
        if ym_ini is None: ym_ini = int(df1['YM'].min())
        if ym_fim is None: ym_fim = int(df1['YM'].max())
        df1 = df1[(df1["YM"] >= ym_ini) & (df1["YM"] <= ym_fim)]
    
    if tipo is not None:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        df1 = df1[df1[col_tipo].astype(str).isin(tipos)]

    # pivot
    tabela = pd.pivot_table(
        df1,
        values=col_qtd,
        index=col_impl,
        columns="YM",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Total Geral",
    )

    # ordenar colunas (meses) e renomear MM/AAAA
    meses = [c for c in tabela.columns if isinstance(c, (int, np.integer))]
    meses = sorted(meses)
    tabela = tabela[meses + ["Total Geral"]]
    tabela = tabela.rename(columns={ym: f"{ym % 100:02d}/{ym // 100}" for ym in meses})

    # separar linha Total Geral
    total_row = tabela.loc[["Total Geral"]] if "Total Geral" in tabela.index else None
    base = tabela.drop(index="Total Geral", errors="ignore")

    # ordenar por Total Geral desc e aplicar top_n
    base = base.sort_values(by="Total Geral", ascending=False)

    if len(base) > top_n:
        top = base.iloc[:top_n]
        resto = base.iloc[top_n:]
        # soma do resto vira OUTROS
        outros = resto.sum(axis=0).to_frame().T
        outros.index = [outros_label]
        base = pd.concat([top, outros], axis=0)

    # recolocar Total Geral no final
    if total_row is not None:
        base = pd.concat([base, total_row], axis=0)

    return base.astype(int)


def paineis_top4_outros(
    df: pd.DataFrame,
    ano: int | None,
    *,
    tipo: str | list[str] | None = None,
    col_tipo: str = "Tipo",
    col_impl: str = "Implementadora",
    col_ano: str = "Ano",
    col_mes: str = "Mes",
    col_qtd: str = "Qtde",
    top_n: int = 4,
    outros_label: str = "OUTROS",
    ym_ini: int | None = None,   # AAAAMM
    ym_fim: int | None = None,   # AAAAMM
):
    df1 = df.copy()
    df1[col_qtd] = pd.to_numeric(df1[col_qtd], errors="coerce").fillna(0.0)
    df1[col_ano] = pd.to_numeric(df1[col_ano], errors="coerce").astype(int)
    df1[col_mes] = pd.to_numeric(df1[col_mes], errors="coerce").astype(int)
    df1[col_impl] = df1[col_impl].astype(str).str.strip()
    df1.loc[df1[col_impl].eq("") | df1[col_impl].isin(["nan", "None"]), col_impl] = "(vazio)"

    # >>> USE YM PARA FILTRAR E PIVOTAR <<<
    df1["YM"] = df1[col_ano] * 100 + df1[col_mes]
    if ym_ini is not None or ym_fim is not None:
        if ym_ini is None: ym_ini = int(df1["YM"].min())
        if ym_fim is None: ym_fim = int(df1["YM"].max())
        df1 = df1[(df1["YM"] >= ym_ini) & (df1["YM"] <= ym_fim)]
    elif ano is not None:
        df1 = df1[df1[col_ano] == int(ano)]

    if tipo is not None:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        df1 = df1[df1[col_tipo].astype(str).isin(tipos)]

    yms = sorted(df1["YM"].unique().tolist())
    if not yms:
        empty = pd.DataFrame(index=[], columns=pd.MultiIndex.from_product([[], ["Valor","%","share"]]))
        return empty, empty.style

    # PIVOT POR YM (não por Mes)
    pvt = (
        df1.pivot_table(index=col_impl, columns="YM", values=col_qtd, aggfunc="sum", fill_value=0.0)
           .reindex(columns=yms, fill_value=0.0)
           .astype(float)
    )

    # rankeia no período selecionado
    tot_impl = pvt.sum(axis=1).sort_values(ascending=False)
    tops = tot_impl.index[:top_n].tolist()
    resto = pvt.drop(index=tops, errors="ignore")
    if len(resto) > 0:
        linha_outros = resto.sum(axis=0).to_frame().T
        linha_outros.index = [outros_label]
        pvt_top = pd.concat([pvt.loc[tops], linha_outros], axis=0)
    else:
        pvt_top = pvt.loc[tops]

    total_mes = pvt.sum(axis=0)  # total do mês (todos)

    # índice amigável MM/AAAA
    idx = [f"{ym % 100:02d}/{ym // 100}" for ym in yms]
    cols = []
    blocks = []

    def _pct_change(arr):
        arr = arr.astype(float)
        out = np.full_like(arr, np.nan, dtype=float)
        prev = arr[:-1]
        denom = np.where(prev == 0, np.nan, prev)
        out[1:] = (arr[1:] - prev) / denom * 100.0
        return out

    for impl in list(pvt_top.index):
        vals = pvt_top.loc[impl, yms].values.astype(float)
        mom  = _pct_change(vals)
        shr  = np.where(total_mes.values == 0, np.nan, vals / total_mes.values * 100.0)
        blocks.append(pd.DataFrame({(impl,"Valor"):vals, (impl,"%"):mom, (impl,"share"):shr}, index=idx))
        cols.extend([(impl,"Valor"), (impl,"%"), (impl,"share")])

    painel = pd.concat(blocks, axis=1)
    painel.columns = pd.MultiIndex.from_tuples(cols)

    # --- formatação ---
    def fmt_number(v): return "" if pd.isna(v) else (f"{int(v):,}".replace(",", ".") if float(v).is_integer() else f"{v:,.0f}".replace(",", "."))
    def fmt_pct(v):    return "" if pd.isna(v) else f"{v:+.1f}%"
    def fmt_share(v):  return "" if pd.isna(v) else f"{v:.1f}%"

    styler = painel.style
    valor_cols = [c for c in painel.columns if c[1] == "Valor"]
    pct_cols   = [c for c in painel.columns if c[1] == "%"]
    share_cols = [c for c in painel.columns if c[1] == "share"]

    styler = styler.format(fmt_number, subset=valor_cols).format(fmt_pct, subset=pct_cols).format(fmt_share, subset=share_cols)

    def _mom_style(s):
        return ["color:#1FA650;font-weight:bold;" if (pd.notna(x) and x >= 0)
                else "color:#E10600;font-weight:bold;" if pd.notna(x) else "" for x in s]
    for col in pct_cols:
        styler = styler.apply(_mom_style, subset=[col])

    styler = styler.set_properties(subset=valor_cols + pct_cols + share_cols, **{"text-align":"right"})

    return painel, styler

def tabela_share_mix(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    outros_label: str = "OUTROS",
    # nomes de colunas no seu DF
    col_impl: str = "Implementadora",
    col_mix: str  = "Mix Produto",
    col_qtd: str  = "Qtde",
    col_ano: str  = "Ano",
    col_mes: str  = "Mes",
    col_tipo: str = "Tipo",
    # filtros (opcionais)
    ym_ini: int | None = None,    # AAAAMM (inclusive)
    ym_fim: int | None = None,    # AAAAMM (inclusive)
    tipo: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Tabela dinâmica de share por 'Mix Produto':
    - linhas: Mix Produto
    - colunas: top_n implementadoras + OUTROS + 'Total Geral'
    - valores: % do total geral no período/ filtro
    """
    df1 = df.copy()

    # saneamento mínimo
    for c in [col_qtd, col_ano, col_mes]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce")
    df1[col_qtd] = df1[col_qtd].fillna(0.0)
    df1[col_impl] = df1[col_impl].astype(str).str.strip()
    df1[col_mix]  = df1[col_mix].astype(str).str.strip()
    df1.loc[df1[col_impl].isin(["", "nan", "None"]), col_impl] = "(vazio)"

    # filtros
    if ym_ini is not None or ym_fim is not None:
        df1["YM"] = (df1[col_ano].astype(int)*100 + df1[col_mes].astype(int))
        if ym_ini is not None:
            df1 = df1[df1["YM"] >= ym_ini]
        if ym_fim is not None:
            df1 = df1[df1["YM"] <= ym_fim]

    if tipo is not None:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        df1 = df1[df1[col_tipo].astype(str).isin(tipos)]

    if df1.empty:
        return pd.DataFrame()

    # total geral do período/filtro
    grand_total = df1[col_qtd].sum()
    if grand_total == 0:
        return pd.DataFrame()

    # top implementadoras (por volume no período)
    impl_rank = df1.groupby(col_impl)[col_qtd].sum().sort_values(ascending=False)
    tops = impl_rank.head(top_n).index.tolist()

    # agrupa restantes em OUTROS
    df1["ImplGrp"] = np.where(df1[col_impl].isin(tops), df1[col_impl], outros_label)

    # pivot absoluto por Mix x ImplGrp
    pvt = df1.pivot_table(
        index=col_mix,
        columns="ImplGrp",
        values=col_qtd,
        aggfunc="sum",
        fill_value=0.0,
    )

    # garante ordem das colunas: tops + OUTROS
    cols = tops + ([outros_label] if outros_label not in tops else [])
    for c in cols:
        if c not in pvt.columns: pvt[c] = 0.0
    pvt = pvt[cols]

    # coluna "Total Geral" por linha e conversão para %
    pvt["Total Geral"] = pvt.sum(axis=1)
    tab_pct = (pvt / grand_total) * 100.0

    # última linha "Total Geral" (colunas) → share por implementador; 100% no canto
    total_cols = tab_pct.drop(columns=["Total Geral"]).sum(axis=0).to_frame().T
    total_cols.index = ["Total Geral"]
    total_cols["Total Geral"] = 100.0

    tab_pct = pd.concat([tab_pct.sort_values(by="Total Geral", ascending=False), total_cols], axis=0)

    return tab_pct


def tabela_share_mix_por_produto(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    outros_label: str = "OUTROS",
    # nomes de colunas no seu DF
    col_impl: str = "Implementadora",
    col_mix: str  = "Mix Produto",
    col_qtd: str  = "Qtde",
    col_ano: str  = "Ano",
    col_mes: str  = "Mes",
    col_tipo: str = "Tipo",
    # filtros (opcionais)
    ym_ini: int | None = None,    # AAAAMM
    ym_fim: int | None = None,    # AAAAMM
    tipo: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Tabela dinâmica:
      - linhas: Mix Produto
      - colunas: top_n implementadoras + OUTROS + 'Total Geral'
      - valores: % do implementador DENTRO do produto (cada linha soma 100%)
      - última linha 'Total Geral': share por implementador no período (soma 100% na linha)
    """
    df1 = df.copy()

    # saneamento mínimo
    for c in [col_qtd, col_ano, col_mes]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce")
    df1[col_qtd] = df1[col_qtd].fillna(0.0)
    df1[col_impl] = df1[col_impl].astype(str).str.strip()
    df1[col_mix]  = df1[col_mix].astype(str).str.strip()
    df1.loc[df1[col_impl].isin(["", "nan", "None"]), col_impl] = "(vazio)"

    # filtros
    if ym_ini is not None or ym_fim is not None:
        df1["YM"] = (df1[col_ano].astype(int)*100 + df1[col_mes].astype(int))
        if ym_ini is not None: df1 = df1[df1["YM"] >= ym_ini]
        if ym_fim is not None: df1 = df1[df1["YM"] <= ym_fim]
    if tipo is not None:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        df1 = df1[df1[col_tipo].astype(str).isin(tipos)]

    if df1.empty or df1[col_qtd].sum() == 0:
        return pd.DataFrame()

    # ranking de implementadoras no período (para escolher top_n colunas)
    impl_rank = df1.groupby(col_impl)[col_qtd].sum().sort_values(ascending=False)
    tops = impl_rank.head(top_n).index.tolist()

    # agrupa demais como OUTROS
    df1["ImplGrp"] = np.where(df1[col_impl].isin(tops), df1[col_impl], outros_label)

    # pivot absoluto por produto x implementadora-agrupada
    pvt_abs = df1.pivot_table(
        index=col_mix,
        columns="ImplGrp",
        values=col_qtd,
        aggfunc="sum",
        fill_value=0.0,
    )

    # garante ordem: TOPS + OUTROS
    cols = tops + ([outros_label] if outros_label not in tops else [])
    for c in cols:
        if c not in pvt_abs.columns: pvt_abs[c] = 0.0
    pvt_abs = pvt_abs[cols]

    # % por linha (produto)
    row_tot = pvt_abs.sum(axis=1)
    tab_pct = pvt_abs.div(row_tot.replace(0, np.nan), axis=0) * 100.0
    tab_pct["Total Geral"] = np.where(row_tot > 0, 100.0, 0.0)

    # linha 'Total Geral' (share por implementador no período)
    grand_total = pvt_abs.values.sum()
    total_row = (pvt_abs.sum(axis=0) / grand_total * 100.0).to_frame().T
    total_row.index = ["Total Geral"]
    total_row["Total Geral"] = 100.0

    # ordena linhas por volume absoluto do produto (desc) e anexa a linha final
    order_idx = row_tot.sort_values(ascending=False).index
    tab_pct = pd.concat([tab_pct.loc[order_idx], total_row], axis=0)

    return tab_pct


def tabela_mix_dentro_fabricante(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    outros_label: str = "OUTROS",
    # nomes de colunas no seu DF
    col_impl: str = "Implementadora",
    col_mix: str  = "Mix Produto",
    col_qtd: str  = "Qtde",
    col_ano: str  = "Ano",
    col_mes: str  = "Mes",
    col_tipo: str = "Tipo",
    # filtros (opcionais)
    ym_ini: int | None = None,    # AAAAMM
    ym_fim: int | None = None,    # AAAAMM
    tipo: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Linhas: Mix Produto
    Colunas: top_n implementadoras + OUTROS + 'Total Geral'
    Valores: % do produto em relação ao total do fabricante (colunas somam 100%).
             'Total Geral' = % do produto no total do mercado.
    """
    df1 = df.copy()

    # saneamento mínimo
    for c in [col_qtd, col_ano, col_mes]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce")
    df1[col_qtd] = df1[col_qtd].fillna(0.0)
    df1[col_impl] = df1[col_impl].astype(str).str.strip()
    df1[col_mix]  = df1[col_mix].astype(str).str.strip()
    df1.loc[df1[col_impl].isin(["", "nan", "None"]), col_impl] = "(vazio)"

    # filtros
    if ym_ini is not None or ym_fim is not None:
        df1["YM"] = (df1[col_ano].astype(int)*100 + df1[col_mes].astype(int))
        if ym_ini is not None: df1 = df1[df1["YM"] >= ym_ini]
        if ym_fim is not None: df1 = df1[df1["YM"] <= ym_fim]
    if tipo is not None:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        df1 = df1[df1[col_tipo].astype(str).isin(tipos)]

    if df1.empty or df1[col_qtd].sum() == 0:
        return pd.DataFrame()

    # top implementadoras no período
    impl_rank = df1.groupby(col_impl)[col_qtd].sum().sort_values(ascending=False)
    tops = impl_rank.head(top_n).index.tolist()

    # agrupa restantes em OUTROS
    df1["ImplGrp"] = np.where(df1[col_impl].isin(tops), df1[col_impl], outros_label)

    # pivot absoluto: produto x (impl/OUTROS)
    pvt_abs = df1.pivot_table(
        index=col_mix,
        columns="ImplGrp",
        values=col_qtd,
        aggfunc="sum",
        fill_value=0.0,
    )

    # ordenação de colunas: TOPS + OUTROS
    cols = tops + ([outros_label] if outros_label not in tops else [])
    for c in cols:
        if c not in pvt_abs.columns: pvt_abs[c] = 0.0
    pvt_abs = pvt_abs[cols]

    # % por fabricante (normaliza por coluna) -> colunas somam 100%
    col_tot = pvt_abs.sum(axis=0)
    tab_pct = pvt_abs.div(col_tot.replace(0, np.nan), axis=1) * 100.0

    # coluna 'Total Geral' = participação do produto no total do mercado
    grand_total = col_tot.sum()
    tab_pct["Total Geral"] = pvt_abs.sum(axis=1) / (grand_total if grand_total else np.nan) * 100.0

    # ordenar linhas por volume absoluto do produto
    order_idx = pvt_abs.sum(axis=1).sort_values(ascending=False).index
    tab_pct = tab_pct.loc[order_idx]

    # última linha 'Total Geral' = 100% em todas as colunas
    total_row = pd.Series({c: 100.0 for c in tab_pct.columns}, name="Total Geral")
    tab_pct = pd.concat([tab_pct, total_row.to_frame().T], axis=0)

    return tab_pct