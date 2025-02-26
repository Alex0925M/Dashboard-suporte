from flask import Flask, render_template_string, request
import pandas as pd
import plotly.express as px
import plotly.utils
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from groq import Groq

# Carrega as variáveis de ambiente
load_dotenv()

# Inicializa o cliente Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def create_robot_analysis_prompt(df):
    """
    Cria o prompt para análise dos dados dos robôs com formato específico.
    """
    # Agrupa por nome do robô e conta ocorrências
    robot_counts = df['Nome do robô'].value_counts()
    
    prompt = """Você é um analista técnico especializado em suporte de automação. Analise os dados fornecidos e crie um relatório 
    seguindo EXATAMENTE este formato para cada robô:

    [Nome do Robô] ([X] Chamados)

    [Categoria do Problema]
    [Descrição explicativa do problema, incluindo a causa raiz identificada nos comentários]

    Regras importantes:
    1. Não use porcentagens ou estatísticas numéricas
    2. Agrupe problemas similares em uma única categoria
    3. Use os comentários realizados para explicar a causa real de cada problema
    4. Mantenha um tom técnico e direto
    5. Cada categoria deve ter uma explicação clara do problema e sua causa
    6. Use as informações dos campos 'Motivo do contato' e 'Comentários do que foi realizado' para criar explicações precisas
    7. Não crie seções de "Conclusão" ou "Resumo"

    Exemplo do formato esperado:
    DCTFWeb (27 Chamados)

    Erro na importação de planilha
    Problemas identificados durante o carregamento de arquivos para processamento em lote, causados principalmente por incompatibilidade de formato ou estrutura dos dados.

    Instabilidade do sistema
    Falhas na execução devido a problemas de conexão com os portais governamentais, resultando em interrupções no processamento automático.

    Dados para análise:
    """
    
    # Adiciona dados de cada robô
    for robot, count in robot_counts.items():
        robot_tickets = df[df['Nome do robô'] == robot]
        
        prompt += f"\n{robot} ({count} Chamados)\n"
        prompt += "Motivos e Comentários:\n"
        
        # Combina motivos e comentários para análise mais precisa
        for _, ticket in robot_tickets.iterrows():
            motivo = ticket['Motivo do contato']
            comentario = ticket['Comentários do que foi realizado']
            if pd.notna(motivo):
                prompt += f"Motivo: {motivo}\n"
            if pd.notna(comentario):
                prompt += f"Comentário: {comentario}\n"
            prompt += "---\n"
    
    prompt += """
    Crie o relatório seguindo EXATAMENTE o formato solicitado, usando as informações fornecidas para
    criar categorias claras e explicações precisas baseadas nos motivos e comentários reais.
    Não adicione informações estatísticas ou conclusões gerais.
    """
    
    return prompt

def get_groq_analysis(prompt):
    """
    Envia o prompt para a API do Groq e retorna a análise.
    """
    try:
        # Faz a chamada para a API do Groq
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Você é um analista técnico que deve criar relatórios precisos e diretos. Siga EXATAMENTE o formato solicitado, sem adicionar seções extras ou estatísticas."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",  # Você pode ajustar o modelo conforme necessário
        )
        
        # Retorna o texto da análise
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao processar análise: {str(e)}"

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard de Atendimentos - Melius Suporte</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@2.8.2/dist/alpine.min.js" defer></script>
</head>
<body class="bg-gray-100">
    <div class="min-h-screen" x-data="{ currentTab: 'daily' }">
        <nav class="bg-white shadow-lg">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between h-16">
                    <div class="flex items-center">
                        <h1 class="text-xl font-bold text-emerald-500">Dashboard de Atendimentos - Suporte Melius</h1>
                    </div>
                </div>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <!-- Filtros -->
            <div class="bg-white rounded-lg shadow p-6 mb-6">
                <form method="get" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Período</label>
                        <select name="period" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                            <option value="day" {% if period == 'day' %}selected{% endif %}>Hoje</option>
                            <option value="week" {% if period == 'week' %}selected{% endif %}>Última Semana</option>
                            <option value="month" {% if period == 'month' %}selected{% endif %}>Último Mês</option>
                            <option value="custom" {% if period == 'custom' %}selected{% endif %}>Personalizado</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Data Inicial</label>
                        <input type="date" name="start_date" value="{{ start_date }}"
                    min="{{ min_date }}" max="{{ max_date }}"
                    class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Data Final</label>
                        <input type="date" name="end_date" value="{{ end_date }}"
                        min="{{ min_date }}" max="{{ max_date }}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
                    </div>
                    <div class="md:col-span-3">
                        <button type="submit"
                                class="w-full md:w-auto bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded">
                            Filtrar
                        </button>
                    </div>
                </form>
            </div>

            <!-- Cards de Resumo -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900">Total de Atendimentos</h3>
                    <p class="text-3xl font-bold text-indigo-600">{{ total_atendimentos }}</p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900">Média por Responsável</h3>
                    <p class="text-3xl font-bold text-indigo-600">{{ media_atendimentos }}</p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900">Tempo Médio de Resolução</h3>
                    <p class="text-3xl font-bold text-indigo-600">{{ tempo_medio }}</p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900">Atendimentos no Prazo/Fora do Prazo</h3>
                    <p class="text-3xl font-bold">
                    <span class="text-green-600">{{ total_atendimentos_no_prazo }}</span>
                    <span class="text-black">/</span>
                    <span class="text-red-600">{{ total_atendimentos_fora_do_prazo }}</span>
                    </p>
                </div>
                
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900">Percentual de nota no atendimento</h3>
                    <p class="text-3xl font-bold text-red-600">0%</p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900">Escalações</h3>
                    <p class="text-3xl font-bold text-red-600">0</p>
                </div>
            </div>

            <!-- Gráficos -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900 mb-4">Atendimentos por Responsável</h3>
                    <div id="grafico_responsaveis" class="h-96"></div>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-medium text-gray-900 mb-4">Distribuição por Sistema</h3>
                    <div id="grafico_sistemas" class="h-96"></div>
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-6 mt-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Motivo do contato</h3>
                <div id="grafico_motivo_contato" class="h-96"></div>
            </div>
            <div class="bg-white rounded-lg shadow p-6 mt-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Quantidade de Clientes</h3>
                <div id="grafico_quantidade_clientes" class="h-96"></div>
            </div>

            <!-- Análise do Groq -->
            <div class="bg-white rounded-lg shadow p-6 mt-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Análise de Chamados por Robô</h3>
                <div class="prose max-w-none">
                    <pre class="whitespace-pre-wrap font-sans text-base leading-relaxed">{{ analysis_result }}</pre>
                </div>
            </div>
        </main>
    </div>

    <script>
        var grafico_responsaveis = {{ grafico_responsaveis | safe }};
        var grafico_sistemas = {{ grafico_sistemas | safe }};
        var grafico_motivo_contato = {{ grafico_motivo_contato | safe }};
        var grafico_quantidade_clientes = {{ grafico_quantidade_clientes | safe }};
        
        Plotly.newPlot('grafico_responsaveis', grafico_responsaveis.data, {
            ...grafico_responsaveis.layout,
            margin: { t: 10, r: 10, b: 50, l: 50 }
        });
        
        Plotly.newPlot('grafico_sistemas', grafico_sistemas.data, {
            ...grafico_sistemas.layout,
            margin: { t: 10, r: 10, b: 50, l: 50 }
        });

        Plotly.newPlot('grafico_motivo_contato', grafico_motivo_contato.data, {
            ...grafico_motivo_contato.layout,
            margin: { t: 10, r: 10, b: 50, l: 50 }
        });

         Plotly.newPlot('grafico_quantidade_clientes', grafico_quantidade_clientes.data, {
            ...grafico_quantidade_clientes.layout,
            margin: { t: 10, r: 10, b: 50, l: 50 }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    # Obtendo parâmetros do filtro
    period = request.args.get('period', 'week')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Lendo o arquivo CSV
    df = pd.read_csv("./DEAL_20250224_b0d3db60_67bc7aaac162c.csv", sep=';', encoding='utf-8')
    
    # Convertendo datas
    df['Criado'] = pd.to_datetime(df['Criado'], format='%d/%m/%Y %H:%M:%S')
    df['Data Encerramento (Automático)'] = pd.to_datetime(df['Data Encerramento (Automático)'], format='%d/%m/%Y %H:%M:%S')
    df['Prazo de Resolução'] = pd.to_datetime(df['Prazo de Resolução'], format='%d/%m/%Y %H:%M:%S')

    # Verificando se o atendimento foi encerrado no prazo
    df['Encerrado no Prazo'] = df['Data Encerramento (Automático)'] <= df['Prazo de Resolução']

    # Determinando limites de data
    min_date = df['Criado'].min().date()
    max_date_csv = df['Criado'].max().date()
    max_date = min(max_date_csv, datetime.now().date())

    # Aplicando filtros de data
    if period == 'day':
        start_date = datetime.now().date()
        end_date = start_date
    elif period == 'week':
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
    elif period == 'month':
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
    elif period == 'custom' and start_date and end_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    else:
        start_date = min_date
        end_date = max_date

    # Filtrando o DataFrame
    df = df[
        (df['Criado'].dt.date >= start_date) &
        (df['Criado'].dt.date <= end_date) &
        (df['Responsável'] != 'Leonardo Barros')
    ]

    # Calculando tempo médio de resolução
    df['Tempo de Resolução'] = df['Data Encerramento (Automático)'] - df['Criado']
    tempo_medio = str(df['Tempo de Resolução'].mean()).split('.')[0]

    # Calculando estatísticas adicionais
    total_atendimentos_no_prazo = df['Encerrado no Prazo'].sum()
    total_atendimentos_fora_do_prazo = len(df) - total_atendimentos_no_prazo
   
    # Criando gráficos
    fig_responsaveis = px.bar(
        df['Responsável'].value_counts(),
        title='',
        labels={'value': 'Quantidade', 'index': 'Responsável'}
    )

    top_10_robos = df['Nome do robô'].value_counts().nlargest(5)

    fig_sistemas = px.pie(
        top_10_robos,
        values=top_10_robos.values,
        names=top_10_robos.index,
        title='Top 5 robôs'
    )

    # Filtrando os top 10 motivos de contato
    top_10_motivos = df['Motivo do contato'].value_counts().nlargest(10)

    # Criando o gráfico de pizza com os top 10 motivos
    fig_contato = px.pie(
        top_10_motivos,
        values=top_10_motivos.values,
        names=top_10_motivos.index,
        title='Top 10 Motivos de Contato'
    )

    top_5_clientes = df['Contato'].value_counts().nlargest(5)

    # Cria o gráfico de pizza com apenas os top 5 clientes
    fig_clientes = px.pie(
        top_5_clientes,
        values=top_5_clientes.values,
        names=top_5_clientes.index,
        title='Top 5 Clientes com Mais Atendimentos'
    )

    # Calculando estatísticas
    total_atendimentos = len(df)
    if len(df['Responsável'].unique()) > 0:
        media_atendimentos = round(len(df) / len(df['Responsável'].unique()), 2)
    else:
        media_atendimentos = 0

    # Gera o prompt para análise
    analysis_prompt = create_robot_analysis_prompt(df)
    
    # Obtém a análise do Groq
    analysis_result = get_groq_analysis(analysis_prompt)

    # Convertendo gráficos para JSON
    grafico_responsaveis = json.dumps(fig_responsaveis, cls=plotly.utils.PlotlyJSONEncoder)
    grafico_sistemas = json.dumps(fig_sistemas, cls=plotly.utils.PlotlyJSONEncoder)
    grafico_motivo_contato = json.dumps(fig_contato, cls=plotly.utils.PlotlyJSONEncoder)
    grafico_quantidade_clientes = json.dumps(fig_clientes, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template_string(
        HTML_TEMPLATE,
        grafico_responsaveis=grafico_responsaveis,
        analysis_result=analysis_result,
        grafico_sistemas=grafico_sistemas,
        grafico_motivo_contato=grafico_motivo_contato,
        grafico_quantidade_clientes=grafico_quantidade_clientes,
        total_atendimentos=total_atendimentos,
        media_atendimentos=media_atendimentos,
        tempo_medio=tempo_medio,
        total_atendimentos_no_prazo=total_atendimentos_no_prazo,
        total_atendimentos_fora_do_prazo=total_atendimentos_fora_do_prazo,
        period=period,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        min_date=min_date.strftime('%Y-%m-%d'),
        max_date=max_date.strftime('%Y-%m-%d')
    )

if __name__ == '__main__':
    app.run(debug=True)