import streamlit as st

# Data wrangling
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


## -- Machine learning -- ##
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import k_means, dbscan, mean_shift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.cluster import AgglomerativeClustering

from collections import Counter
from itertools import combinations

# ===========================================================================
# Funções do projeto
# ===========================================================================

def categoria_climatizados(row):
  if row['rua'] in ['C1', 'C2', 'C3']:
    return 'Climatizados'
  else:
    return row['categoria']


def freq_sku(df_produtos):
  freq_df = (df_produtos.groupby('sku_id')['order_id']
             .count().sort_values(ascending=False))
  return freq_df


def matriz_co_ocorrencia_rapida(df_pedidos):
    # Conta todas as coocorrências
    pares_contagem = Counter()

    # Agrupa os pedidos por ID
    df_grouped = df_pedidos.groupby('order_id')['sku_id'].agg(list)

    
    for produtos in df_grouped:
        produtos = list(set(produtos))  # opcional: evita repetição de SKU no mesmo pedido
        pares = combinations(sorted(produtos), 2)
        pares_contagem.update(pares)

    # Cria DataFrame a partir do dicionário de contagens
    dados = []
    for (sku1, sku2), count in pares_contagem.items():
        dados.append((sku1, sku2, count))
        dados.append((sku2, sku1, count))  # adiciona simetria

    df_co = pd.DataFrame(dados, columns=['sku1', 'sku2', 'count'])

    # Pivot para matriz
    matriz = df_co.pivot_table(index='sku1', columns='sku2', values='count', fill_value=0)

    # Preenche diagonal com zeros (ou np.nan se preferir)
    for sku in matriz.index:
        if sku in matriz.columns:
            matriz.loc[sku, sku] = 0

    return matriz


def cluster_catergoria(matriz, n_clusters):
  # Pipeline interno só com os passos encaixáveis
  pipe = Pipeline([
  ('pca', PCA(n_components=5)),
  ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
  ])

  # Aplica a pipeline
  clusters = pipe.fit_predict(matriz)

  # Recupera as componentes principais
  pca_result = pipe.named_steps['pca'].transform(matriz)

  # Retorna um DataFrame organizado
  df_result = pd.DataFrame(pca_result,
                          columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
                          index=matriz.index)
  df_result.index.name = 'sku_id'
  df_result['cluster'] = clusters
  return df_result['cluster']


def df_ordenado(df_freq, df_cluster):
  # Transformar lim_freq em DataFrame
  df_freq = df_freq.to_frame(name='frequencia')

  # Juntar com os clusters
  df_ordenado = df_freq.join(df_cluster)

  # Resetar índice para facilitar
  df_ordenado = df_ordenado.reset_index()

  # Para cada cluster, ordenar por frequência
  grupos = df_ordenado.groupby('cluster', dropna=False).apply(lambda g: g.sort_values('frequencia', ascending=False))

  # Agora ordena os grupos pela frequência do produto mais frequente de cada um
  grupos = grupos.reset_index(drop=True)
  grupos['cluster_max_freq'] = grupos.groupby('cluster')['frequencia'].transform('max')

  # Ordena os clusters por essa frequência máxima
  grupos = grupos.sort_values(by=['cluster_max_freq', 'cluster', 'frequencia'], ascending=[False, True, False])

  # Lista final
  lista_produtos = grupos[['sku_id', 'frequencia', 'cluster']]
  return lista_produtos


def index_of_max(ord_produtos):
  dic_freq_max = {}
  for nome in ord_produtos: # Iterate directly through keys
      dic_freq_max[nome] = ord_produtos[nome]['frequencia'].max() #access dict by key


  df = pd.DataFrame.from_dict(dic_freq_max, orient='index', columns=['Frequencia Maxima'])
  index_of_max = df['Frequencia Maxima'].idxmax()
  return index_of_max


ruas = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C1', 'C2', 'C3',
 'B1', 'B2', 'R9',
 'B3', 'B4', 'B5', 'B6', 'R10',
  'B7', 'B8', 'R11']
  #essa lista de ruas direto do mapeamento está como precisamos

ruas_principais = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C1', 'C2', 'C3',
 'R9',  'R10', 'R11']


def posicao_produtos_ordem(posicao_produtos):
  ruas = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C1', 'C2', 'C3',
  'R9', 'B1', 'B2',
  'R10', 'B3', 'B4', 'B5', 'B6',
  'R11','B7', 'B8', ]
  ruas_ordem = ruas.copy() # cópia para manipulação
  for rua in ruas: #analisando se uma rua está na lista pelos produtos
    if rua not in posicao_produtos:
      ruas_ordem.remove(rua)
  return ruas_ordem


def convert_bolsao(rua):
    if rua in ['B1', 'B2']:
        return 'R9'
    elif rua in ['B3', 'B4', 'B5', 'B6']:
        return 'R10'
    elif rua in ['B7', 'B8']:
        return 'R11'
    return rua


def sentido_rua(rua):
    return dict_sentidos[rua]


def rua_anterior(rua):
    idx = ruas_principais.index(rua)
    return ruas_principais[idx - 1] if idx > 0 else rua


def prox_rua(altura, rua_desejada):
    rua_desejada = convert_bolsao(rua_desejada)
    sentido = sentido_rua(rua_desejada)

    if (sentido == 'Para Baixo' and altura == 1) or (sentido == 'Para Cima' and altura == -1):
        return rua_desejada
    else:
        return rua_anterior(rua_desejada)


def rota(posicao_produtos):
    altura = 1
    rota_final = []

    for rua in posicao_produtos_ordem(posicao_produtos):
        if rua in rota_final:
          continue
        if convert_bolsao(rua) in rota_final:
          rota_final.append(rua)
          continue

        next_rua = prox_rua(altura, rua)
        rota_final.append(next_rua)
        if next_rua != rua:
          if convert_bolsao(rua) != rua and convert_bolsao(rua) != next_rua:
            rota_final.append(convert_bolsao(rua))
          rota_final.append(rua)

        if next_rua == convert_bolsao(rua):
          altura += -2 if sentido_rua(next_rua) == 'Para Baixo' else 2

    if altura == -1:
        rota_final.append('R11')
        next = prox_rua(altura, rua)

    return rota_final


def distancia_pedido(lista_produtos):
    posicao_produtos = [dict_produtos[sku] for sku in lista_produtos if sku in dict_produtos]
    rota_atual = rota(posicao_produtos)
    return sum(dict_distancias.get(r, 0) for r in rota_atual)


def reposicionamento(clim,):
  # Todas as ruas do galpão e seus respectivos espaços disponívies
  ruas = {}
  if clim == False:
    for rua in galpao[~galpao['Sigla'].isin(['C1', 'C2', 'C3'])]['Sigla']: # sem climatizados
      ruas[rua] = int(galpao[galpao['Sigla'] == rua]['Qtd Posições'].values[0])
  else:
    for rua in galpao[galpao['Sigla'].isin(['C1', 'C2', 'C3'])]['Sigla']: # climatizados
      ruas[rua] = int(galpao[galpao['Sigla'] == rua]['Qtd Posições'].values[0])
  
  df_final = pd.DataFrame()

  # Loop dentro das ruas para preenche-las
  for rua in list(ruas.keys()):
    # Verificando a qnt de posições disponíveis na rua no total
    posicoes = ruas[rua]

    # Dicionário com os df
    if clim == False:
      ord_produtos = ord_produtos_geral
    else:
      ord_produtos = ord_produtos_clim

    fechar_rua = False
    while fechar_rua == False:
      # Caso já tenha rodado muito e a base está vazia:
      if all(df.empty for df in ord_produtos.values()): #todos já estão vazios
        break

      # Selecionando a base com os produtos mais frequêntes (prioridade)
      df_freq_max = ord_produtos[index_of_max(ord_produtos)]

      soma_posicoes = int(df_freq_max['qtd_posicoes'].sum())

      # Verificando se a categoria será suficiente para completar a rua
      if soma_posicoes < posicoes:
        df = df_freq_max.copy()
        df['rua'] = rua #registrando a rua
        posicoes = posicoes - int(df['qtd_posicoes'].sum())
        df_final = pd.concat([df_final, df]) # adicionar últimos produtos na lista
        df_freq_max.drop(index=df_freq_max.index, inplace= True) # remove-los do df deles

        #se a categoria usada nessa rua for alimentos, não pode continuar com limpeza, vice-versa
        if index_of_max(ord_produtos) == 'ord_alim':
          ord_produtos = {
            'ord_alim': ord_alim_teste,
            'ord_out': ord_out_teste}
        if index_of_max(ord_produtos) == 'ord_lim' or index_of_max(ord_produtos) == 'ord_hig':
          ord_produtos = {
            'ord_lim': ord_lim_teste,
            'ord_hig': ord_hig_teste,
            'ord_out': ord_out_teste}


        continue # próxima categoria mais frequente

      if soma_posicoes == posicoes:
        df = df_freq_max.copy()
        df['rua'] = rua #registrando a rua
        df_final = pd.concat([df_final, df]) # adicionar produtos na lista
        df_freq_max.drop(index=df_freq_max.index, inplace= True) # remove-los do df deles
        fechar_rua = True

      if soma_posicoes > posicoes:
        for i in range(len(df_freq_max)):
          df = df_freq_max.iloc[0:i].copy() # Salvando os produtos que cabem na rua
          if df['qtd_posicoes'].sum() == posicoes:
            df_freq_max.drop(index=df_freq_max.iloc[0:i].index, inplace= True) # remove-los do df deles
            break # Encerra o for
          elif df['qtd_posicoes'].sum() > posicoes:
            df = df_freq_max.iloc[0:(i-1)].copy()
            df_freq_max.drop(index=df_freq_max.iloc[0:(i-1)].index, inplace= True) # remove-los do df deles
            break # Encerra o for
        df['rua'] = rua #registrando a rua
        if rua == 'R1':
          df_final = df
        else:
          df_final = pd.concat([df_final, df]) # adicionar produtos na lista
        fechar_rua = True


  return df_final

# ===========================================================================
# Streamlit App
# ===========================================================================
st.set_page_config(page_title="Otimização picking", layout="wide")
st.title('📦 Otimização da rota de picking')

produtos_esperadas = ['sku_id', 'rua', 'categoria', 'qtd_posicoes']
pedidos_esperadas = ['order_id', 'delivery_date', 'sku_id']
galpao_esperadas = ['Sigla', 'Qtd Posições', 'Rota (m)', 'Sentido']


st.markdown(
    '<h3><span style="color:#07AB7A; font-weight:bold;">O que este aplicativo faz?</span></h1>',
    unsafe_allow_html=True)
st.markdown("""
Este app foi criado para **otimizar a organização de produtos em galpões logísticos** com base em dados reais de pedidos.  
Ele utiliza técnicas de **análise de dados e aprendizado de máquina** para propor uma nova configuração dos produtos que:

- **Reduz a distância percorrida pelos operadores** durante a separação (picking);
- **Agrupa itens que costumam ser pedidos juntos**;
- **Aumenta a eficiência e reduz custos operacionais** no centro de distribuição.""")

with st.expander("ℹ️ Como usar o app", expanded=False):
    st.markdown("""
### 📝 Passo a passo para usar o otimizador de picking:

1. **Carregue os 3 arquivos CSV obrigatórios:**
   - **Base de produtos**: informações como sku_id, rua, categoria,  qtd_posicoes.
   - **Base de pedidos**: composição de pedidos com order_id, delivery_date, sku_id.
   - **Base do galpão**: mapeamento atual de localização dos produtos com Sigla, Qtd Posições, Rota (m), Sentido.


2. **Clique no botão `Processar dados`** para iniciar a análise e otimização.

3. O app irá:
    - Validar os dados
    - Pre-processar os dados
        - Tipos de dados das colunas
        - Entradas nulas e duplicadas
        - Produtos ativos e categorias necessárias
    - Determinar a frequência que os produtos são comprados
    - Agrupar os produtos por frequentemente comprados juntos
    - Redistribuir os produtos nas ruas baseado na frequência de compra (considerando os frequentemente comprados juntos, e as regras de distribuição por categorias)
    - Calcular a Distância média por picking na organização atual e com a nova proposta pelo programa


4. **Clique em `Baixar resultados`** para exportar a nova configuração do galpão.

---

**⚠️ Requisitos dos arquivos:**
- Arquivos devem estar no formato `.csv`.
- Tamanho máximo de 200MB por arquivo.

---

Se tiver dúvidas, fale com a equipe de dados ou consulte a documentação do projeto.
    """)

# Uploads
produtos_file = st.sidebar.file_uploader("📄 Base de produtos", type="csv")
pedidos_file = st.sidebar.file_uploader("📄 Base de pedidos", type="csv")
galpao_file = st.sidebar.file_uploader("📄 Base de mapeamento do galpão", type="csv")

# Inicializa os estados de validação se ainda não estiverem presentes
for key in ['validado1', 'validado2', 'validado3']:
    if key not in st.session_state:
        st.session_state[key] = False
    
produtos_esperadas = ['sku_id', 'rua', 'categoria', 'qtd_posicoes']
pedidos_esperadas = ['order_id', 'delivery_date', 'sku_id']
galpao_esperadas = ['Sigla', 'Qtd Posições', 'Rota (m)', 'Sentido']

# Função de validação
def validar_arquivo(uploaded_file, colunas_esperadas, session_key):
    if uploaded_file and not st.session_state[session_key]:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in colunas_esperadas):
            st.toast(f"{uploaded_file.name} validado com sucesso ✅")
            st.session_state[session_key] = True
        else:
            st.error(f"{uploaded_file.name} está faltando colunas necessárias: {colunas_esperadas}")

# Valida cada arquivo
validar_arquivo(produtos_file, produtos_esperadas, 'validado1')
validar_arquivo(pedidos_file, pedidos_esperadas, 'validado2')
validar_arquivo(galpao_file, galpao_esperadas, 'validado3')


if produtos_file is None and pedidos_file is None and galpao_file is None:
    st.warning("Por favor, carregue os arquivos antes de processar.")


if st.button('Processar dados'):
    produtos = pd.read_csv(produtos_file)
    pedidos = pd.read_csv(pedidos_file)
    galpao = pd.read_csv(galpao_file)

    with st.status('Processando os dados...'):
        # Preprocessing: -------------------------------------------------------
        st.write("Pre-processando os dados...")
        # Produtos
        produtos['sku_id'] = produtos['sku_id'].astype('object')
        produtos['rua'] = produtos['rua'].astype('object')
        produtos['categoria'] = produtos['categoria'].astype('object')
        produtos['qtd_posicoes'] = produtos['qtd_posicoes'].astype('int')
        produtos.dropna(inplace=True)
        produtos.drop_duplicates(inplace=True)
        # somente produtos em ruas ativas
        ruas = galpao['Sigla'].to_list()
        produtos = produtos[produtos['rua'].isin(ruas)]
        # adicionando uma categoria de climatizados
        produtos['categoria'] = produtos.apply(categoria_climatizados, axis=1)

        #Pedidos
        pedidos['order_id'] = pedidos['order_id'].astype('object')
        pedidos['delivery_date'] = pd.to_datetime(pedidos['delivery_date'])
        pedidos['sku_id'] = pedidos['sku_id'].astype('object')
        pedidos.dropna(inplace=True)
        pedidos.drop_duplicates(inplace=True)
        # somente produtos na base de pedidos que existam na base de produtos
        produtos_ativos = produtos['sku_id'].to_list()
        pedidos = pedidos[pedidos['sku_id'].isin(produtos_ativos)]
        #separação por categoria
        pedidos = pd.merge(pedidos, produtos[['sku_id', 'categoria']], on='sku_id', how='left')
        pedidos_alim = pedidos.query('categoria == "Alimentos"')
        pedidos_hig = pedidos.query('categoria == "Higiene"')
        pedidos_lim = pedidos.query('categoria == "Limpeza"')
        pedidos_clim = pedidos.query('categoria == "Climatizados"')
        pedidos_out = pedidos.query('categoria == "Outros"')

        #Galpão
        galpao['Sentido'] = galpao['Sentido'].str.strip().str.lower()

        # Modeling -------------------------------------------------------
        #Frequencia de cada sku por categoria
        st.write("Calculando as frequências de compras...")
        max_freq = freq_sku(pedidos).max()
        alim_freq = freq_sku(pedidos_alim) / max_freq
        hig_freq = freq_sku(pedidos_hig) / max_freq
        lim_freq = freq_sku(pedidos_lim) / max_freq
        clim_freq = freq_sku(pedidos_clim) / max_freq
        out_freq = freq_sku(pedidos_out) / max_freq

        # Lista de sku por categoria, ordenado por frequência geral e
        #frequentemente comprados juntos
        st.write("Produtos frequentemente comprados juntos, por categoria...")
        # Alimentos
        with st.spinner("Alimentos"):
            matriz_alim = matriz_co_ocorrencia_rapida(pedidos_alim)
            cluster_alim = cluster_catergoria(matriz_alim, n_clusters=7)
            ord_alim = df_ordenado(alim_freq, cluster_alim)
            ord_alim = ord_alim.join(produtos[['sku_id', 'qtd_posicoes']].set_index('sku_id'), on='sku_id')
            ord_alim = ord_alim.reset_index(drop=True)
            ord_alim['categoria'] = 'Alimentos'

        # Limpeza
        with st.spinner("Limpeza"):
            matriz_lim = matriz_co_ocorrencia_rapida(pedidos_lim)
            cluster_lim = cluster_catergoria(matriz_lim, n_clusters=7)
            ord_lim = df_ordenado(lim_freq, cluster_lim)
            ord_lim = ord_lim.join(produtos[['sku_id', 'qtd_posicoes']].set_index('sku_id'), on='sku_id')
            ord_lim = ord_lim.reset_index(drop=True)
            ord_lim['categoria'] = 'Limpeza'

        # Higiene
        with st.spinner("Higiene"):
            matriz_hig = matriz_co_ocorrencia_rapida(pedidos_hig)
            cluster_hig = cluster_catergoria(matriz_hig, n_clusters=7)
            ord_hig = df_ordenado(hig_freq, cluster_hig)
            ord_hig = ord_hig.join(produtos[['sku_id', 'qtd_posicoes']].set_index('sku_id'), on='sku_id')
            ord_hig = ord_hig.reset_index(drop=True)
            ord_hig['categoria'] = 'Higiene'

        # Outros
        with st.spinner("Outros"):
            matriz_out = matriz_co_ocorrencia_rapida(pedidos_out)
            cluster_out = cluster_catergoria(matriz_out, n_clusters=7)
            ord_out = df_ordenado(out_freq, cluster_out)
            ord_out = ord_out.join(produtos[['sku_id', 'qtd_posicoes']].set_index('sku_id'), on='sku_id')
            ord_out = ord_out.reset_index(drop=True)
            ord_out['categoria'] = 'Outros'

        # Climatizados
        with st.spinner("Climatizados"):
            matriz_clim = matriz_co_ocorrencia_rapida(pedidos_clim)
            cluster_clim = cluster_catergoria(matriz_clim, n_clusters=7)
            ord_clim = df_ordenado(clim_freq, cluster_clim)
            ord_clim = ord_clim.join(produtos[['sku_id', 'qtd_posicoes']].set_index('sku_id'), on='sku_id')
            ord_clim = ord_clim.reset_index(drop=True)
            ord_clim['categoria'] = 'Climatizados'

        # Montando nova organização no galpão
        st.write("Redistribuindo os produtos no galpão...")
        # Copiando os df para poder manipula-los com tranquilidade
        ord_alim_teste = ord_alim.copy()
        ord_hig_teste = ord_hig.copy()
        ord_lim_teste = ord_lim.copy()
        ord_clim_teste = ord_clim.copy()
        ord_out_teste = ord_out.copy()

        ord_produtos_geral = {
            'ord_alim': ord_alim_teste,
            'ord_lim': ord_lim_teste,
            'ord_hig': ord_hig_teste,
            'ord_out': ord_out_teste
        }

        ord_produtos_clim = {'ord_clim': ord_clim_teste}

        produtos_novo = reposicionamento(clim=False)
        produtos_novo_clim = reposicionamento(clim=True)
        produtos_novo = pd.concat([produtos_novo, produtos_novo_clim])


        # Evaluating -------------------------------------------------------
        st.write("Avaliando resultados...")
        ruas = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C1', 'C2', 'C3',
        'B1', 'B2', 'R9',
        'B3', 'B4', 'B5', 'B6', 'R10',
        'B7', 'B8', 'R11']

        ruas_principais = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C1', 'C2', 'C3',
        'R9',  'R10', 'R11']

        dict_sentidos = galpao.set_index('Sigla')['Sentido'].to_dict()
        dict_distancias = galpao.set_index('Sigla')['Rota (m)'].to_dict()

        pedidos_calcular = (pedidos.groupby('order_id')['sku_id']
                        .agg(list).reset_index())

        # Distância na organização atual
        dict_produtos = produtos.set_index('sku_id')['rua'].to_dict()
        pedidos_calcular['Distancia_antes'] = pedidos_calcular['sku_id'].apply(distancia_pedido)
        media_antes = float(round(pedidos_calcular['Distancia_antes'].mean(), 2))

        # Distância na organização nova
        dict_produtos = produtos_novo.set_index('sku_id')['rua'].to_dict() #usar a base de organização condizente
        pedidos_calcular['Distancia_nova'] = pedidos_calcular['sku_id'].apply(distancia_pedido)
        media_nova = float(round(pedidos_calcular['Distancia_nova'].mean(), 2))

        # Diferença
        razao = round(((media_nova - media_antes) / media_antes)*100, 1)


    st.markdown(
    '<h3><span style="color:#07AB7A; font-weight:bold;">Resultados:</span></h1>',
    unsafe_allow_html=True)
    csv = produtos_novo[["sku_id","rua","categoria","qtd_posicoes"]].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Baixar nova base de produtos", csv, "produtos_nova_disposicao.csv", "text/csv")
    a, b = st.columns(2)
    a.metric("Distância média por piking na configuração atual:", f"{media_antes} metros", border=True)
    b.metric("Distância média por piking na configuração nova:", f"{media_nova} metros", f'{razao}%', border=True, delta_color="inverse")
    
    pedidos_mes = float(round(pedidos.groupby(pedidos['delivery_date'].dt.to_period('M'))['order_id']
     .nunique().mean(), 1))
    pedidos_mes_novo = round((1-razao/100)*pedidos_mes,1)
    c, d = st.columns(2)
    c.metric('Pedidos processados por mês - atual', f"{pedidos_mes/1000:.1f} mil", border=True)
    d.metric('Predição: Pedidos processados por mês - novo', f"{pedidos_mes_novo/1000:.1f} mil", f'{-razao}%', border=True)


    st.markdown("---")
    st.markdown(
    '<h3><span style="color:#07AB7A; font-weight:bold;">Comparando as organizações atuais e novas:</span></h1>',
    unsafe_allow_html=True)
    # Visualização
    # Calcular o maior total de SKUs em uma rua (em ambos os conjuntos)
    max_y1 = produtos.groupby('rua')['sku_id'].count().max()
    max_y2 = produtos_novo.groupby('rua')['sku_id'].count().max()
    max_y = max(max_y1, max_y2)

    cores_personalizadas = {
    'Alimentos': '#07ab7a',      
    'Limpeza': '#8a4fff',    
    'Higiene': '#e953b2',      
    'Climatizados': '#0155ac',     
    'Outros': '#3db2ff',          
}

    # Gráfico Atual
    totais_por_rua = produtos.groupby('rua')['sku_id'].count().sort_values(ascending=False)
    ruas_ordenadas = totais_por_rua.index.tolist()
    produtos['rua'] = pd.Categorical(produtos['rua'], categories=ruas_ordenadas, ordered=True)
    df_grouped = produtos.groupby(['rua', 'categoria'])['sku_id'].count().reset_index()
    df_grouped.rename(columns={'sku_id': 'quantidade'}, inplace=True)
    # Plotar gráfico
    fig = px.bar(df_grouped, x='rua', y='quantidade', color='categoria',
        title='Distribuição de produtos - Atual',
        labels={'quantidade': 'Quantidade de SKUs'}, text_auto=False, color_discrete_map=cores_personalizadas)
    fig.update_layout(barmode='stack', xaxis_title='Rua', yaxis_title='Quantidade de SKUs', legend_title='', 
    legend=dict(orientation='v', yanchor='top', y=1, xanchor='right', x=1),
        xaxis_tickangle=-45, height=500, yaxis=dict(range=[0, max_y]))

    # Gráfico Novo
    totais_por_rua = produtos_novo.groupby('rua')['sku_id'].count().sort_values(ascending=False)
    ruas_ordenadas = totais_por_rua.index.tolist()
    produtos_novo['rua'] = pd.Categorical(produtos_novo['rua'], categories=ruas_ordenadas, ordered=True)
    df_grouped = produtos_novo.groupby(['rua', 'categoria'])['sku_id'].count().reset_index()
    df_grouped.rename(columns={'sku_id': 'quantidade'}, inplace=True)
    # Plotar gráfico
    fig2 = px.bar(df_grouped, x='rua', y='quantidade', color='categoria',
        title='Distribuição de produtos - Nova', labels={'quantidade': 'Quantidade de SKUs'},
        text_auto=False, color_discrete_map=cores_personalizadas)
    fig2.update_layout(barmode='stack', xaxis_title='Rua', yaxis_title='Quantidade de SKUs', legend_title='',
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='right', x=1),
        xaxis_tickangle=-45, height=500, yaxis=dict(range=[0, max_y]))

    # Mostrar gráficos lado a lado
    e, f = st.columns(2)
    e.plotly_chart(fig, use_container_width=True)
    f.plotly_chart(fig2, use_container_width=True)


    # Histograma das distâncias
    num_bins = 18

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=pedidos_calcular['Distancia_antes'], 
      histnorm='percent', name='Config. Atual', opacity=1 , nbinsx=num_bins, marker_color='#0155ac'))
    fig3.add_trace(go.Histogram(x=pedidos_calcular['Distancia_nova'], 
      histnorm='percent', name='Config. Nova', opacity=1, nbinsx=num_bins,  marker_color='#07AB7A'))
    fig3.update_layout(
      title_text='Histograma: Distâncias por picking', xaxis_title_text='Distâncias', yaxis_title_text='Contagem (%)',
      bargap=0.25, # gap between bars of adjacent location coordinates
      bargroupgap=0.05, # gap between bars of the same location coordinates
      xaxis_tickangle=-45, height=500
      )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("Essa é a cara da nova base de produtos, com as ruas redistribuidas")
    st.table(produtos_novo[["sku_id","rua","categoria","qtd_posicoes"]].head(10))

    


st.markdown("---")
st.caption("""📦 Desenvolvido pelo Grupo 9 | Projeto DNC - Shopper | v1.0
              Fernando De Faria
              Francisco de Assis
              Giovanni Periago""")