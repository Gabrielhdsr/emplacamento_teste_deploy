###Página 1 – Share por Mix (top10 + OUTROS, % do total)###

import streamlit as st, pandas as pd
from data import load_emplacamento
import graficos as gf
import filtros as flt
from datetime import datetime
import lib as lb

ano_atual = datetime.now().year

st.title("Emplacamento por Implementadora (Top 10 + outros)")
df = load_emplacamento('arquivos/Emplacamento/*.xlsx')
row = st.columns([5,6,7])  # 2 selects compactos + “respiro”


with row[0]:
    df_flt = df[df['Ano']>= 2024]
    ym_ini, ym_fim, meta =  flt.filtro_anos_botoes(df_flt)
with row[2]:
    # tipos = ["Todos"] + sorted(df["Tipo"].dropna().astype(str).unique())
    # tipo_sel = st.selectbox("Tipo", tipos, index=0, label_visibility="collapsed")
    tipo_sel = flt.filtro_tipo_segmented(df)

st.subheader("Emplacamento por Implementadora (dinâmico)")

tabela_dyn = gf.tabela_emplacamento_pivot(
    df,
    ym_ini=ym_ini,
    ym_fim=ym_fim,
    tipo=None if tipo_sel == "Todos" else tipo_sel,
    top_n=10,                # <- aqui
    outros_label="OUTROS"    # <- opcional
)
# exibição com milhar "."
styler = tabela_dyn.style.format(lambda v: f"{int(v):,}".replace(",", ".")).apply(lb.highlight_facchini, axis=1)
linhas = len(tabela_dyn)

st.dataframe(
    styler,
    use_container_width=True,
    height=(linhas + 1) * 35   # altura aproximada por linha
)

# filtros como você já tem...
tipo_filtro = None if tipo_sel == "Todos" else tipo_sel

painel_df, painel_styler = gf.paineis_top4_outros(
    df,
    ano = None,
    tipo=tipo_filtro,
    top_n=4,                # 4 maiores
    outros_label="OUTROS",
    ym_ini=ym_ini,
    ym_fim=ym_fim,
)

st.subheader("Top 4 Implementadores + OUTROS (Valor / % MoM / Share)")
linhas = len(painel_df)
st.dataframe(
    painel_styler,
    use_container_width=True,
    height=(linhas + 2) * 36,   # ajusta p/ evitar rolagem
)