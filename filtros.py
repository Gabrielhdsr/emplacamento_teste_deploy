"""
Módulo de filtros reutilizáveis para Streamlit.

Como usar no seu app:

    import filters as flt

    # 1) Filtros simples (Ano + Tipo em colunas)
    ano_sel, tipo_filtro = flt.filtro_ano_tipo(df_emplacamento, ano_padrao=datetime.now().year)

    # 2) Slider de período com Ano/Mês da base (retorna AAAAMM)
    ym_ini, ym_fim, meta = flt.filtro_periodo(df_emplacamento)

    # 3) Segmented control para Tipo (retorna None quando "Todos")
    tipo_filtro = flt.filtro_tipo_segmented(df_emplacamento)

    # 4) Utilitários de formatação
    styler = tabela.style.format(flt.formata_milhar_pt).apply(lb.highlight_facchini, axis=1)

Este módulo não lê dados; ele só constroi os widgets e devolve valores prontos
para aplicar nos seus dataframes e funções de gráfico/tabela.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ==========================
# Helpers de dados/labels
# ==========================

@st.cache_data(show_spinner=False)
def _anos_disponiveis(df: pd.DataFrame, col_ano: str = "Ano") -> List[int]:
    anos = (
        pd.to_numeric(df[col_ano], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(anos)

@st.cache_data(show_spinner=False)
def _tipos_disponiveis(df: pd.DataFrame, col_tipo: str = "Tipo", incluir_todos: bool = True) -> List[str]:
    tipos = (
        df[col_tipo]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    tipos = sorted(tipos)
    return (["Todos"] + tipos) if incluir_todos else tipos

@dataclass
class PeriodoMeta:
    comp: np.ndarray            # vetor de AAAAMM ordenado
    labels: List[str]           # rótulos MM/AAAA
    idx_ini: int                # índice selecionado no slider (início)
    idx_fim: int                # índice selecionado no slider (fim)

@st.cache_data(show_spinner=False)
def _competencias(df: pd.DataFrame, col_ano: str = "Ano", col_mes: str = "Mes") -> Tuple[np.ndarray, List[str]]:
    """Retorna vetor ordenado de competências no formato AAAAMM e os labels MM/AAAA."""
    ano = pd.to_numeric(df[col_ano], errors="coerce").astype("Int64")
    mes = pd.to_numeric(df[col_mes], errors="coerce").astype("Int64")
    comp = (ano * 100 + mes).dropna().astype(int).unique()
    comp = np.sort(comp)
    labels = [f"{c % 100:02d}/{c // 100}" for c in comp]
    return comp, labels

# ==========================
# Widgets de filtro
# ==========================

def filtro_ano_select(
    df: pd.DataFrame,
    col_ano: str = "Ano",
    label: str = "Ano",
    key: Optional[str] = None,
    ano_padrao: Optional[int] = None,
    label_visibility: str = "collapsed",
) -> Optional[int]:
    """Selectbox de Ano. Retorna o ano selecionado (int) ou None se vazio.
    - ano_padrao: se estiver na lista, é pré-selecionado.
    """
    anos = _anos_disponiveis(df, col_ano)
    if not anos:
        st.warning("Nenhum ano disponível para filtro.")
        return None
    if ano_padrao in anos:
        idx = anos.index(ano_padrao)
    else:
        idx = len(anos) - 1  # último disponível
    return st.selectbox(label, anos, index=idx, key=key, label_visibility=label_visibility)


def filtro_tipo_select(
    df: pd.DataFrame,
    col_tipo: str = "Tipo",
    label: str = "Tipo",
    key: Optional[str] = None,
    incluir_todos: bool = True,
    label_visibility: str = "collapsed",
) -> Optional[str]:
    """Selectbox de Tipo. Retorna None quando "Todos".
    """
    tipos = _tipos_disponiveis(df, col_tipo, incluir_todos)
    sel = st.selectbox(label, tipos, key=key, label_visibility=label_visibility)
    return None if (incluir_todos and sel == "Todos") else sel


def filtro_ano_tipo(
    df: pd.DataFrame,
    col_ano: str = "Ano",
    col_tipo: str = "Tipo",
    ano_padrao: Optional[int] = None,
    pesos_colunas: Iterable[float] = (1, 2, 6),
    keys: Tuple[Optional[str], Optional[str]] = ("f_ano", "f_tipo"),
    label_visibility: str = "collapsed",
) -> Tuple[Optional[int], Optional[str]]:
    """Cria dois filtros lado a lado (Ano | Tipo). Retorna (ano_sel, tipo_filtro).
    tipo_filtro é None quando "Todos" estiver marcado.
    """
    col_ano_st, col_tipo_st, _ = st.columns(pesos_colunas)
    with col_ano_st:
        ano_sel = filtro_ano_select(
            df, col_ano=col_ano, key=keys[0], ano_padrao=ano_padrao, label_visibility=label_visibility
        )
    with col_tipo_st:
        tipo_filtro = filtro_tipo_select(
            df, col_tipo=col_tipo, key=keys[1], incluir_todos=True, label_visibility=label_visibility
        )
    return ano_sel, tipo_filtro


def filtro_tipo_segmented(
    df: pd.DataFrame,
    col_tipo: str = "Tipo",
    label: str = "Tipo",
    key: str = "tipo_seg",
) -> Optional[str]:
    """Segmented control (ou radio, se indisponível) para Tipo. Retorna None quando "Todos"."""
    tipos_opts = _tipos_disponiveis(df, col_tipo, incluir_todos=True)
    # st.segmented_control pode não existir em versões antigas — fallback para radio
    if hasattr(st, "segmented_control"):
        sel = st.segmented_control(label, tipos_opts, key=key)
    else:
        sel = st.radio(label, tipos_opts, horizontal=True, key=key)
    return None if sel == "Todos" else sel


def filtro_periodo(
    df: pd.DataFrame,
    col_ano: str = "Ano",
    col_mes: str = "Mes",
    label: str = "Período",
    key: str = "periodo_slider",
    usar_tudo_por_padrao: bool = True,
    label_visibility: str = "collapsed",
) -> Tuple[int, int, PeriodoMeta]:
    """Cria um select_slider para escolher um intervalo de competências.

    Retorna (ym_ini, ym_fim, meta), onde cada ym é um inteiro AAAAMM.
    """
    comp, labels = _competencias(df, col_ano, col_mes)
    if comp.size == 0:
        st.warning("Não há competências (Ano/Mês) válidas na base.")
        return 0, 0, PeriodoMeta(comp=np.array([]), labels=[], idx_ini=0, idx_fim=0)

    if usar_tudo_por_padrao:
        ini_idx, fim_idx = 0, len(comp) - 1
    else:
        # último ano completo (heurística simples)
        anos = sorted({c // 100 for c in comp})
        ano_ultimo = anos[-1]
        # busca primeiro índice do ano_ultimo e último também
        idxs = [i for i, c in enumerate(comp) if c // 100 == ano_ultimo]
        ini_idx, fim_idx = (idxs[0], idxs[-1]) if idxs else (0, len(comp) - 1)

    i_ini, i_fim = st.select_slider(
        label,
        options=list(range(len(comp))),
        value=(ini_idx, fim_idx),
        format_func=lambda i: labels[i],
        key=key,
        label_visibility=label_visibility,
    )

    ym_ini, ym_fim = int(comp[i_ini]), int(comp[i_fim])
    meta = PeriodoMeta(comp=comp, labels=labels, idx_ini=i_ini, idx_fim=i_fim)
    return ym_ini, ym_fim, meta


# ==========================
# Formatação/Styling úteis
# ==========================

def formata_milhar_pt(v: Any) -> str:
    """Inteiro com ponto de milhar (br), para usar em DataFrame.style.format.
       Aceita NaN/None.
    """
    try:
        if pd.isna(v):
            return ""
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return str(v)


def formata_percentual_pt(v: Any, casas: int = 2) -> str:
    """Percentual com vírgula como separador decimal."""
    try:
        if pd.isna(v):
            return ""
        return f"{float(v):.{casas}f}%".replace(".", ",")
    except Exception:
        return str(v)


def estilo_valores_facchini(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    """Encadeia formatação de milhar PT + permite apply externos (ex: highlight_facchini)."""
    return styler.format(formata_milhar_pt)


# ==========================
# Atalhos de aplicação
# ==========================

def aplicar_filtro_tipo(df: pd.DataFrame, tipo: Optional[str], col_tipo: str = "Tipo") -> pd.DataFrame:
    """Filtra o DataFrame por Tipo quando fornecido; senão retorna cópia do df."""
    if tipo is None:
        return df.copy()
    return df[df[col_tipo].astype(str) == str(tipo)].copy()


def ano_atual_padrao() -> int:
    """Resolve o ano atual do sistema (útil para passar como ano_padrao)."""
    return datetime.now().year


def filtro_anos_botoes(
    df: pd.DataFrame,
    col_ano: str = "Ano",
    col_mes: str = "Mes",
    label: str = "Anos",
    key: str = "anos_botoes_intervalo",
    ano_padrao: Optional[int] = None,   # se None, usa último ano disponível
    wrap: int = 8,                      # nº de botões por linha
    label_visibility: str = "collapsed",
) -> Tuple[int, int, PeriodoMeta]:
    """
    Seletor de intervalo de ANOS com botões, compatível com os visuais que usam filtro_periodo.
    Retorna (ym_ini, ym_fim, meta) onde ym_* são AAAAMM presentes na base e
    meta é PeriodoMeta (comp, labels, idx_ini, idx_fim) coerente com _competencias.

    Regras de clique:
    - Clicou fora do intervalo atual -> intervalo expande para incluir esse ano.
    - Clicou dentro do intervalo atual -> intervalo colapsa para aquele ano (ini=fim=ano).
    """
    anos = _anos_disponiveis(df, col_ano)
    comp, labels = _competencias(df, col_ano, col_mes)

    if not anos or comp.size == 0:
        st.warning("Não há anos/competências válidos na base.")
        return 0, 0, PeriodoMeta(comp=np.array([]), labels=[], idx_ini=0, idx_fim=0)

    # estado
    ini_key = f"{key}_ini"
    fim_key = f"{key}_fim"

    if ini_key not in st.session_state or fim_key not in st.session_state:
        # default: ano_padrao (se existir) ou último ano disponível
        ano_default = ano_padrao if (ano_padrao in anos) else anos[-1]
        st.session_state[ini_key] = int(ano_default)
        st.session_state[fim_key] = int(ano_default)

    ano_ini = int(st.session_state[ini_key])
    ano_fim = int(st.session_state[fim_key])
    if ano_ini > ano_fim:
        ano_ini, ano_fim = ano_fim, ano_ini

    if label_visibility != "collapsed":
        st.caption(label)

    # grade de botões (visual de “só botões”)
    for i in range(0, len(anos), wrap):
        cols = st.columns(min(wrap, len(anos) - i))
        for j, ano in enumerate(anos[i : i + wrap]):
            with cols[j]:
                in_range = (ano_ini <= ano <= ano_fim)
                if st.button(
                    str(ano),
                    key=f"{key}_{ano}",
                    type="primary" if in_range else "secondary",
                    use_container_width=True,
                ):
                    # lógica de atualização do intervalo
                    if ano < ano_ini:
                        ano_ini = ano
                    elif ano > ano_fim:
                        ano_fim = ano
                    else:
                        # clicou dentro do range -> colapsa
                        ano_ini = ano
                        ano_fim = ano
                    st.session_state[ini_key] = int(ano_ini)
                    st.session_state[fim_key] = int(ano_fim)

    # converte intervalo de anos para intervalo de competências disponíveis (AAAAMM) dentro do DF
    alvo_ini = ano_ini * 100 + 1     # jan do ano_ini
    alvo_fim = ano_fim * 100 + 12    # dez do ano_fim

    # busca pelos índices válidos no vetor comp (ordenado)
    i_ini = int(np.searchsorted(comp, alvo_ini, side="left"))
    i_fim = int(np.searchsorted(comp, alvo_fim, side="right") - 1)

    if i_ini >= comp.size or i_fim < 0 or i_ini > i_fim:
        st.warning("Não há competências dentro do intervalo de anos selecionado.")
        return 0, 0, PeriodoMeta(comp=comp, labels=labels, idx_ini=0, idx_fim=0)

    ym_ini = int(comp[i_ini])
    ym_fim = int(comp[i_fim])
    meta = PeriodoMeta(comp=comp, labels=labels, idx_ini=i_ini, idx_fim=i_fim)
    return ym_ini, ym_fim, meta