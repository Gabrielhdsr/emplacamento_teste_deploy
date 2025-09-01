#IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter
import numpy as np
import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io
import math

def base_anfavea(caminho_arquivo= str | Patch, ano_minimo= 2013)->pd.DataFrame:
    """
    Lê os dados de produção de caminhão da planilha da Anfavea e retorna um DataFrame filtrado por ano.

    Parâmetros
    ----------
    caminho_arquivo : str
        Caminho para o arquivo Excel (.xlsm).
    ano_minimo : int, opcional
        Ano mínimo para filtrar os dados (default: 2013).

    Retorno
    -------
    pd.DataFrame
        DataFrame com colunas de interesse filtrado a partir do ano especificado.
    """
    df = pd.read_excel(caminho_arquivo, header=None)
    df = df.drop(index=[0, 1, 2, 3])
    df.columns = df.iloc[0]
    df = df.drop(index=4)
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[:, [0, 16, 17, 18, 19, 20]]
    df = df.rename(columns={df.columns[0]: 'Data'})
    df['Data'] = pd.to_datetime(df['Data'])
    df['Ano'] = df['Data'].dt.year
    df_filtrado = df[df['Ano'] >= ano_minimo]
    
    return df_filtrado

def carregar_emplacamento(pasta_arquivos: str, ano_minimo: int = 2013) -> pd.DataFrame:
    """
    Lê e consolida os arquivos de emplacamento em uma única base.

    Parâmetros
    ----------
    pasta_arquivos : str
        Caminho da pasta contendo os arquivos Excel de emplacamento.
    ano_minimo : int, opcional
        Ano mínimo para filtrar os dados. Se None, não aplica filtro.

    Retorno
    -------
    pd.DataFrame
        DataFrame consolidado e processado.
    """
    
    # Localiza todos os arquivos Excel na pasta
    arquivos_emplacamento = glob.glob(pasta_arquivos)
    
    dfs = []
    for arquivo in arquivos_emplacamento:
        df = pd.read_excel(arquivo, skiprows=3, header=None)
        df = df.drop(columns=[11], axis= 1, errors="ignore")
        dfs.append(df)
    
    # Concatena todos os arquivos
    df_emplacamento = pd.concat(dfs, ignore_index=True)
    
    # Ajusta nomes das colunas
    df_emplacamento.columns = [
        'Tipo', 'UF', 'Cidade', 'Qtde', 'Implementadora', 'Mix Produto',
        'Modelo', 'Cliente', 'Representante', 'Faturado', 'Data'
    ]
    
    # Converte coluna de datas
    df_emplacamento['Data'] = pd.to_datetime(df_emplacamento['Data'], errors='coerce')
    
    # Extrai mês e ano
    df_emplacamento['Mes'] = df_emplacamento['Data'].dt.month
    df_emplacamento['Ano'] = df_emplacamento['Data'].dt.year
    
    # Aplica filtro por ano, se solicitado
    if ano_minimo is not None:
        df_emplacamento = df_emplacamento[df_emplacamento['Ano'] >= ano_minimo]
    
    return df_emplacamento


def previsao_media3m(df_base, ano, implementadora=None):
    """
    Método: média dos últimos 3 meses *do próprio ano* × meses restantes + acumulado do ano.
    df_base: DataFrame com colunas 'Ano','Mes','Qtde' e (opcional) 'Implementadora'
    ano: int do ano alvo (ex.: ano atual)
    implementadora: None para total geral ou uma string (ex.: 'FACCHINI')
    """
    df_ano = df_base[df_base['Ano'] == ano].copy()
    if implementadora is not None and 'Implementadora' in df_ano.columns:
        df_ano = df_ano[df_ano['Implementadora'].str.upper().str.strip() == implementadora.upper().strip()]

    if df_ano.empty:
        return 0.0  # sem dados no ano → previsão 0

    # garante tipos
    df_ano['Mes'] = pd.to_numeric(df_ano['Mes'], errors='coerce').astype('Int64')

    # agrega por mês dentro do ano
    mensal = (
        df_ano.dropna(subset=['Mes'])
              .groupby('Mes', as_index=True)['Qtde']
              .sum()
              .sort_index()
    )

    if mensal.empty:
        return 0.0

    # acumulado do ano (YTD)
    ytd = mensal.sum()

    # último mês com dado e meses restantes no ano
    ultimo_mes = int(mensal.index.max())
    meses_restantes = max(0, 12 - ultimo_mes)

    # média dos últimos 3 meses com dados no ano (se tiver menos, usa o que houver)
    ultimos3_idx = mensal.index[-3:]
    media_3m = mensal.loc[ultimos3_idx].mean()

    # previsão = YTD + média_3m * meses_restantes
    return float(ytd + media_3m * meses_restantes)

def prepara_emplacamento(df, ano_atual= datetime.now().year):
    df['Qtde'] = pd.to_numeric(df['Qtde'], errors= 'coerce').fillna(0)
    df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')

    totais = df.groupby("Ano")['Qtde'].sum()

    totais_facchini =(
        df[df['Implementadora'].str.upper().str.strip() == 'FACCHINI']
        .groupby("Ano")['Qtde'].sum()
        .reindex(totais.index, fill_value= 0)
    )

    tabela = pd.DataFrame({
        "Total_Geral": totais,
        "Total_Facchini": totais_facchini
    }).sort_index()
    #Adiciona previsão no dataframe

    prev_geral = previsao_media3m(df, ano_atual, implementadora=None)

    prev_facchini = previsao_media3m(df, ano_atual, implementadora= 'FACCHINI')

    # adiciona as colunas de previsão na sua 'tabela' apenas para o ano atual
    if ano_atual in tabela.index:
        tabela.loc[ano_atual, 'Previsto_Total_Geral']= prev_geral
        tabela.loc[ano_atual, 'Previsto_Total_Facchini'] = prev_facchini
    else:
        # se o ano ainda não existe na tabela, cria a linha
        tabela.loc[ano_atual, 'Total_Geral'] = 0
        tabela.loc[ano_atual, 'Total_Facchini'] = 0
        tabela.loc[ano_atual, 'Previsto_Total_Geral'] = prev_geral
        tabela.loc[ano_atual, 'Previsto_Total_Facchini'] = prev_facchini

    if ano_atual in tabela.index:
        if "Previsto_Total_Geral" in tabela.columns:
            tabela.loc[ano_atual, "Total_Geral"] = tabela.loc[ano_atual, "Previsto_Total_Geral"]
        if "Previsto_Total_Facchini" in tabela.columns:
            tabela.loc[ano_atual, "Total_Facchini"] = tabela.loc[ano_atual, "Previsto_Total_Facchini"]

    tabela = tabela.sort_index()  
    
    return tabela

def highlight_facchini(row):
    if row.name.strip().upper() == "FACCHINI":
        return ['font-weight: bold;'] * len(row)
    return [''] * len(row)


def previsao_total_ano_prod(
    df_mensal: pd.DataFrame,
    ano: int,
    col_ano: str = "Ano",
    col_mes: str = "Mes",
    col_valor: str = "Valor"
) -> dict:
    """
    Calcula a previsão de TOTAL do ano:
      total_prev = soma(YTD) + média(últimos 3 meses disponíveis) * (12 - último_mês_disponível)
    Retorna dict com total_prev, ytd, add_prev (parcela prevista) e mes_max.
    """
    base = df_mensal.copy()
    base[col_ano] = pd.to_numeric(base[col_ano], errors="coerce").astype("Int64")
    base[col_mes] = pd.to_numeric(base[col_mes], errors="coerce").astype("Int64")
    base[col_valor] = pd.to_numeric(base[col_valor], errors="coerce")

    m = base.query(f"{col_ano} == @ano").dropna(subset=[col_mes, col_valor])
    if m.empty:
        return {"total_prev": None, "ytd": None, "add_prev": None, "mes_max": None}

    meses_disponiveis = sorted(m[col_mes].unique().tolist())
    mes_max = int(m[col_mes].max())
    ytd = float(m[col_valor].sum())

    # pega os últimos 3 meses efetivamente disponíveis
    ultimos3 = meses_disponiveis[-3:] if len(meses_disponiveis) >= 3 else meses_disponiveis
    media_ult3 = float(m.loc[m[col_mes].isin(ultimos3), col_valor].mean())

    faltam = max(0, 12 - mes_max)
    add_prev = media_ult3 * faltam
    total_prev = ytd + add_prev

    return {
        "total_prev": total_prev,
        "ytd": ytd,
        "add_prev": add_prev,
        "mes_max": mes_max
    }