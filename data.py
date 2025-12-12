import streamlit as st
from matplotlib.patches import Patch
import glob
import pandas as pd


@st.cache_data(show_spinner="Carregando e processando dados de produção...")
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
    #CARREGA ARQUIVO
    try:
        df = pd.read_excel(caminho_arquivo, header=None)
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo '{caminho_arquivo}': {e}")
    
    #REMOVE INICIO
    df = df.drop(index=[0, 1, 2, 3])

    #DEFINE CABEÇALHO
    df.columns = df.iloc[0]
    df = df.drop(index=4)
    df.reset_index(drop=True, inplace=True)

    #SELEÇAO DE COLUNAS
    df = df.iloc[:, [0, 16, 17, 18, 19, 20]]

    #TRATANDO DATAS
    df = df.rename(columns={df.columns[0]: 'Data'})
    df['Data'] = pd.to_datetime(df['Data'])
    df['Ano'] = df['Data'].dt.year
    df = df[df['Ano'] >= ano_minimo].reset_index(drop=True)
    
    return df

@st.cache_data(show_spinner="Carregando e processando dados de emplacamento...")
def carregar_emplacamento(pasta_arquivos: str, ano_minimo: int = 2013) -> pd.DataFrame:
    """
    Lê e consolida todos os arquivos de emplacamento em uma única base padronizada.
    """

    # 1. Encontrar arquivos
    arquivos = glob.glob(pasta_arquivos)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: {pasta_arquivos}")

    dfs = []
    for arquivo in arquivos:
        try:
            df = pd.read_excel(arquivo, skiprows=3, header=None)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")
            continue  # Ignora arquivo problemático

        # 2. Validação mínima: checar número de colunas
        if df.shape[1] < 11:
            print(f"Arquivo ignorado por ter menos colunas que o esperado: {arquivo}")
            continue

        # 3. Limpeza padronizada
        df = df.drop(columns=[11], axis=1, errors="ignore")

        # 4. Garantir formato com exatamente 11 colunas
        df = df.iloc[:, :11]

        dfs.append(df)

    if not dfs:
        raise ValueError("Nenhum arquivo válido para processamento.")

    # 5. Concatenação
    df_emplacamento = pd.concat(dfs, ignore_index=True)

    # 6. Nome das colunas
    colunas = [
        'Tipo', 'UF', 'Cidade', 'Qtde', 'Implementadora', 'Mix Produto',
        'Modelo', 'Cliente', 'Representante', 'Faturado', 'Data'
    ]
    df_emplacamento.columns = colunas

    # 7. Converter tipos
    df_emplacamento['Data'] = pd.to_datetime(df_emplacamento['Data'], errors='coerce')
    df_emplacamento['Qtde'] = pd.to_numeric(df_emplacamento['Qtde'], errors='coerce').fillna(0).astype(int)

    # 8. Remover registros sem data
    df_emplacamento = df_emplacamento.dropna(subset=['Data'])

    # 9. Criar mês e ano
    df_emplacamento['Mes'] = df_emplacamento['Data'].dt.month
    df_emplacamento['Ano'] = df_emplacamento['Data'].dt.year

    # 10. Filtrar anos
    if ano_minimo is not None:
        df_emplacamento = df_emplacamento[df_emplacamento['Ano'] >= ano_minimo]

    # 11. Reindexar
    df_emplacamento = df_emplacamento.reset_index(drop=True)

    return df_emplacamento


