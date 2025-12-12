# app.py

import streamlit as st
from datetime import datetime

import data as dt

st.set_page_config(page_title="Dashboard Emplacamentos FACCHINI", layout="wide")



# ----------------------------------------------------------
# 1) PLACEHOLDER — A PÁGINA APARECE ANTES DOS DADOS CARREGAREM
# ----------------------------------------------------------

carregamento = st.empty()   # espaço reservado para mensagem de carregamento

# Mensagem aparece imediatamente
with carregamento.container():
    st.info("Carregando dados... Aguarde alguns segundos.")


# ----------------------------------------------------------
# 2) CARREGAMENTO EM SEGUNDO PLANO (SESSÃO DO USUÁRIO)
# ----------------------------------------------------------

if "dados_carregados" not in st.session_state:

    # Carrega tudo normalmente — só que agora a interface já apareceu
    try:
        df_anfavea = dt.base_anfavea(
            caminho_arquivo='arquivos/SeriesTemporais_Autoveiculos (1).xlsm'
        )

        df_emplacamento = dt.carregar_emplacamento(
            pasta_arquivos='arquivos/Emplacamento/*.xlsx'
        )

    except Exception as e:
        carregamento.error(f"Erro ao carregar os dados: {e}")
        st.stop()

    # Salva tudo em session_state
    st.session_state.df = df_emplacamento
    st.session_state.df_anfavea = df_anfavea
    st.session_state.ano_atual = datetime.now().year
    st.session_state.dados_carregados = True

    # Remove a mensagem agora que terminou
    carregamento.empty()

# ----------------------------------------------------------
# 3) DADOS DISPONÍVEIS PARA O RESTO DO APP
# ----------------------------------------------------------

df = st.session_state.df
ano_atual = st.session_state.ano_atual


# ----------------------------------------------------------
# 4) SUA INTERFACE ORIGINAL (sem mudanças)
# ----------------------------------------------------------

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.markdown("<h1 style='text-align: center;'>Dashboard de Análise de Emplacamentos FACCHINI</h1>", unsafe_allow_html=True)

    st.header("👋 Introdução")
    st.write("""
        Este painel interativo oferece uma análise detalhada e segmentada sobre os emplacamentos 
        de implementos rodoviários (SC e SR), focando na performance da FACCHINI 
        em relação ao mercado total e seus principais concorrentes.
    """)

    st.header("🔗 Estrutura de Navegação")
    st.markdown("""
        Utilize o menu lateral para navegar entre as análises, que seguem uma progressão lógica:
        
        * **1. Visão Geral - KPIs:** Desempenho atual (Anual, Trimestral, Mensal) vs. Períodos anteriores.
        * **2. Share Histórico:** Evolução do Market Share por tipo de implemento ao longo dos anos.
        * **3. Análise Regional:** Ranking de desempenho por Estado e suas tendências.
        * **4. Top Players:** Análise da concorrência e identificação dos maiores *players* do mercado.
        * **5. Visão de Mercado:** Tabelas detalhadas e dados agregados.
    """)

    st.header("📊 Fontes dos Dados e Cache")
    st.markdown("""
        Os dados são provenientes de **Arquivos de Emplacamento** e séries temporais da **ANFAVEA**. 
        O sistema de cache está ativo para garantir que os dados sejam carregados rapidamente.
    """)
