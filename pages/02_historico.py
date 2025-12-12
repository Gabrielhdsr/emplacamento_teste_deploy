import streamlit as st
import pandas as pd
import altair as alt

import data as dt
import lib as lb
import ui  # Seu arquivo de design

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(layout="wide", page_title="Histórico de Mercado", page_icon="📊")

# ============================================================
# CARGA DE DADOS
# ============================================================
df = dt.carregar_emplacamento('arquivos/Emplacamento/*.xlsx')

SC = "SOBRE CHASSI"
SR = "SEMIRREBOQUE"
FACCHINI = "FACCHINI"

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================
def preparar_dados_empilhados(df_raw, tipo_filtro=None):
    """
    Prepara dados para gráfico de barras empilhadas (Facchini vs Concorrência).
    """
    df_f = df_raw.copy()
    if tipo_filtro:
        df_f = df_f[df_f["Tipo"] == tipo_filtro]
    
    # Agrupa por Ano e Implementadora (Facchini vs Outros)
    # Primeiro, marcamos quem é quem
    df_f['Player'] = df_f['Implementadora'].apply(lambda x: FACCHINI if x == FACCHINI else 'Concorrência')
    
    # Agrupamos
    df_ano = df_f.groupby(['Ano', 'Player'])['Qtde'].sum().reset_index()
    
    # Calcula totais para labels
    df_totais = df_f.groupby('Ano')['Qtde'].sum().reset_index().rename(columns={'Qtde': 'Total_Mercado'})
    
    # Junta tudo
    df_final = pd.merge(df_ano, df_totais, on='Ano')
    
    # Ordena para o gráfico (Facchini em baixo ou em cima, conforme preferencia. Normalmente destaque fica na base ou topo)
    # Vamos deixar Facchini em destaque (cor vermelha)
    return df_final

def preparar_ranking_atual(df_raw, ano_foco, tipo_filtro=None):
    """
    Gera o ranking das Top 5 + Outros para o ano selecionado.
    """
    df_f = df_raw[df_raw["Ano"] == ano_foco].copy()
    if tipo_filtro:
        df_f = df_f[df_f["Tipo"] == tipo_filtro]
        
    ranking = df_f.groupby("Implementadora")["Qtde"].sum().reset_index()
    ranking = ranking.sort_values("Qtde", ascending=False)
    
    total_ano = ranking["Qtde"].sum()
    ranking["Share"] = (ranking["Qtde"] / total_ano) * 100
    
    # Top 5 e agrupa resto
    top5 = ranking.head(5).copy()
    outros_qtd = ranking.iloc[5:]["Qtde"].sum()
    outros_share = ranking.iloc[5:]["Share"].sum()
    
    if outros_qtd > 0:
        row_outros = pd.DataFrame([{"Implementadora": "OUTROS", "Qtde": outros_qtd, "Share": outros_share}])
        top5 = pd.concat([top5, row_outros])
        
    return top5

# ============================================================
# GRÁFICOS (ALTAIR)
# ============================================================
def plot_mercado_total(df_dados, titulo):
    # Definindo cores: Facchini Vermelho, Concorrência Cinza
    scale_color = alt.Scale(domain=[FACCHINI, 'Concorrência'], range=['#dc2626', '#cbd5e1'])
    
    base = alt.Chart(df_dados).encode(
        x=alt.X('Ano:O', title=None),
        y=alt.Y('Qtde:Q', title='Volume de Vendas'),
        order=alt.Order('Player', sort='ascending') # Facchini na base ou topo
    )

    # Barras Empilhadas
    bars = base.mark_bar(size=40).encode(
        color=alt.Color('Player', scale=scale_color, legend=alt.Legend(title="Composição")),
        tooltip=['Ano', 'Player', 'Qtde', alt.Tooltip('Total_Mercado', title='Mercado Total')]
    )

    # Texto com o Total no topo da barra
    # Para isso, usamos o dataset de totais (apenas um registro por ano)
    text_total = alt.Chart(df_dados.drop_duplicates('Ano')).mark_text(
        dy=-10, color='#1e293b', fontWeight='bold'
    ).encode(
        x=alt.X('Ano:O'),
        y=alt.Y('Total_Mercado:Q'),
        text=alt.Text('Total_Mercado:Q', format='.')
    )
    
    # Texto com o valor da Facchini (dentro da barra vermelha)
    text_facchini = base.mark_text(dy=0, color='white', fontWeight='bold').encode(
        text=alt.Text('Qtde:Q', format='.'),
        opacity=alt.condition(alt.datum.Player == FACCHINI, alt.value(1), alt.value(0))
    )

    chart = (bars + text_total + text_facchini).properties(
        title=titulo,
        height=400
    )
    return chart

def plot_pizza_share(df_ranking, titulo):
    base = alt.Chart(df_ranking).encode(
        theta=alt.Theta("Qtde", stack=True)
    )
    
    # Cores: Destaca Facchini, outros em tons de cinza/azul
    pie = base.mark_arc(outerRadius=120).encode(
        color=alt.Color("Implementadora", 
                        scale=alt.Scale(domain=[FACCHINI, 'OUTROS'], range=['#dc2626', '#94a3b8']), 
                        legend=None), # Legenda customizada ou automática
        order=alt.Order("Qtde", sort="descending"),
        tooltip=["Implementadora", "Qtde", alt.Tooltip("Share", format=".1f")]
    )
    
    text = base.mark_text(radius=140).encode(
        text=alt.Text("Share", format=".1f"),
        order=alt.Order("Qtde", sort="descending"),
        color=alt.value("black")  
    )
    
    # Vamos usar um gráfico de barras horizontal simples para o ranking, é mais "profissional" que pizza as vezes
    # Mas como pediu "distribuição", vamos de Barras Horizontais com Facchini destacada
    
    bars = alt.Chart(df_ranking).mark_bar().encode(
        x=alt.X('Share:Q', title='Market Share (%)'),
        y=alt.Y('Implementadora:N', sort='-x', title=None),
        color=alt.condition(
            alt.datum.Implementadora == FACCHINI,
            alt.value('#dc2626'),  # Vermelho se for Facchini
            alt.value('#cbd5e1')   # Cinza se não for
        ),
        tooltip=['Implementadora', 'Qtde', alt.Tooltip('Share', format='.1f')]
    )
    
    text_bar = bars.mark_text(align='left', dx=2).encode(
        text=alt.Text('Share:Q', format='.1f')
    )
    
    return (bars + text_bar).properties(title=titulo, height=300)

# ============================================================
# PÁGINA
# ============================================================
ui.header("Evolução de Mercado", "Volume Total e Market Share • 2013 a 2025")
ui.apply_style()

# 1. GRÁFICO PRINCIPAL (CONSOLIDADO)
ui.section("Mercado Total (Consolidado)")
df_stack_cons = preparar_dados_empilhados(df)
st.altair_chart(plot_mercado_total(df_stack_cons, ""), use_container_width=True)

# 2. GRÁFICOS SEGMENTADOS (LADO A LADO)
ui.section("Detalhamento por Segmento")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### {SC}")
    df_stack_sc = preparar_dados_empilhados(df, SC)
    st.altair_chart(plot_mercado_total(df_stack_sc, ""), use_container_width=True)

with col2:
    st.markdown(f"#### {SR}")
    df_stack_sr = preparar_dados_empilhados(df, SR)
    st.altair_chart(plot_mercado_total(df_stack_sr, ""), use_container_width=True)

# 3. DISTRIBUIÇÃO DE SHARE (RANKING)
ui.section("Posição Competitiva (Ranking)")
st.caption(f"Comparativo de Market Share: Facchini vs Principais Concorrentes ({df['Ano'].max()})")

col_rank_sc, col_rank_sr = st.columns(2)
ano_atual = df["Ano"].max()

with col_rank_sc:
    st.markdown(f"**Ranking: {SC} ({ano_atual})**")
    df_rank_sc = preparar_ranking_atual(df, ano_atual, SC)
    st.altair_chart(plot_pizza_share(df_rank_sc, ""), use_container_width=True)

with col_rank_sr:
    st.markdown(f"**Ranking: {SR} ({ano_atual})**")
    df_rank_sr = preparar_ranking_atual(df, ano_atual, SR)
    st.altair_chart(plot_pizza_share(df_rank_sr, ""), use_container_width=True)