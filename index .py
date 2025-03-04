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
    <style>
        [x-cloak] { display: none !important; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="min-h-screen flex flex-col" x-data="{ 
        sidebarOpen: true,
        mobileMenuOpen: false,
        currentTab: 'dashboard'
    }" x-cloak>
        <!-- Header com gradiente e logo -->
        <header class="bg-gradient-to-r from-emerald-600 to-indigo-700 shadow-lg z-10">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <!-- Botão de menu mobile -->
                        <button @click="mobileMenuOpen = !mobileMenuOpen" class="md:hidden mr-2 inline-flex items-center justify-center p-2 rounded-md text-white hover:text-white hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                            <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        
                        <!-- Toggle sidebar no desktop -->
                        <button @click="sidebarOpen = !sidebarOpen" class="hidden md:inline-flex mr-2 items-center justify-center p-2 rounded-md text-white hover:text-white hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                            <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        
                        <!-- Logo e título -->
                        <div class="bg-white p-2 rounded-full shadow-md mr-4">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <div>
                            <h1 class="text-2xl font-bold text-white">Dashboard de Atendimentos</h1>
                            <p class="text-indigo-100">Suporte Melius - Análise de chamados</p>
                        </div>
                    </div>
                    <div class="text-white text-right">
                        <p class="font-medium">Período: {{ start_date }} a {{ end_date }}</p>
                        <p class="text-sm text-indigo-100">Atualizado em: {{ datetime.now().strftime('%d/%m/%Y %H:%M') }}</p>
                    </div>
                </div>
            </div>
        </header>
        
        <div class="flex flex-1 overflow-hidden">
            <!-- Sidebar para Mobile (overlay) -->
            <div x-show="mobileMenuOpen" 
                 class="fixed inset-0 flex z-40 md:hidden"
                 x-transition:enter="transition-opacity ease-linear duration-300"
                 x-transition:enter-start="opacity-0"
                 x-transition:enter-end="opacity-100"
                 x-transition:leave="transition-opacity ease-linear duration-300"
                 x-transition:leave-start="opacity-100"
                 x-transition:leave-end="opacity-0">
                
                <!-- Overlay backdrop -->
                <div class="fixed inset-0" 
                     @click="mobileMenuOpen = false">
                    <div class="absolute inset-0 bg-gray-600 opacity-75"></div>
                </div>
                
                <!-- Sidebar content -->
                <div class="relative flex-1 flex flex-col max-w-xs w-full bg-gray-800 transition-all transform ease-in-out duration-300">
                    <!-- Botão fechar -->
                    <div class="absolute top-0 right-0 -mr-12 pt-2">
                        <button @click="mobileMenuOpen = false" class="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                            <svg class="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                    
                    <div class="px-2 pt-6 pb-3 space-y-1">
                        <!-- Links da Sidebar -->
                        <a href="#" @click.prevent="currentTab = 'dashboard'; mobileMenuOpen = false" class="block px-3 py-2 rounded-md text-base font-medium" :class="currentTab == 'dashboard' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                            <div class="flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                                </svg>
                                Dashboard
                            </div>
                        </a>
                        <a href="#" @click.prevent="currentTab = 'responsaveis'; mobileMenuOpen = false" class="block px-3 py-2 rounded-md text-base font-medium" :class="currentTab == 'responsaveis' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                            <div class="flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                                </svg>
                                Responsáveis
                            </div>
                        </a>
                        <a href="#" @click.prevent="currentTab = 'robos'; mobileMenuOpen = false" class="block px-3 py-2 rounded-md text-base font-medium" :class="currentTab == 'robos' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                            <div class="flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                                </svg>
                                Robôs
                            </div>
                        </a>
                        <a href="#" @click.prevent="currentTab = 'analise'; mobileMenuOpen = false" class="block px-3 py-2 rounded-md text-base font-medium" :class="currentTab == 'analise' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                            <div class="flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                Análise
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            
            <!-- Sidebar para Desktop (fixa) -->
            <div class="hidden md:flex md:flex-shrink-0">
                <div class="flex flex-col w-64 transition-width duration-300" :class="sidebarOpen ? 'w-64' : 'w-16'">
                    <div class="flex flex-col flex-grow bg-gray-800 pt-5 overflow-y-auto">
                        <div class="flex-grow flex flex-col">
                            <nav class="flex-1 px-2 space-y-1 pb-4">
                                <!-- Links da Sidebar -->
                                <a href="#" @click.prevent="currentTab = 'dashboard'" class="flex items-center px-2 py-2 text-sm font-medium rounded-md" :class="currentTab == 'dashboard' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                                    </svg>
                                    <span :class="{ 'hidden': !sidebarOpen }">Dashboard</span>
                                </a>
                                <a href="#" @click.prevent="currentTab = 'responsaveis'" class="flex items-center px-2 py-2 text-sm font-medium rounded-md" :class="currentTab == 'responsaveis' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                                    </svg>
                                    <span :class="{ 'hidden': !sidebarOpen }">Responsáveis</span>
                                </a>
                                <a href="#" @click.prevent="currentTab = 'robos'" class="flex items-center px-2 py-2 text-sm font-medium rounded-md" :class="currentTab == 'robos' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                                    </svg>
                                    <span :class="{ 'hidden': !sidebarOpen }">Robôs</span>
                                </a>
                                <a href="#" @click.prevent="currentTab = 'analise'" class="flex items-center px-2 py-2 text-sm font-medium rounded-md" :class="currentTab == 'analise' ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    <span :class="{ 'hidden': !sidebarOpen }">Análise</span>
                                </a>
                            </nav>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Conteúdo principal -->
            <div class="flex-1 overflow-auto">
                <!-- Filtros -->
                <div class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
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
                    
                    <!-- Conteúdo da Dashboard (todos) -->
                    <div x-show="currentTab === 'dashboard'">
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
                                <div id="grafico_sistemas" class="h-80"></div>
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
                    </div>
                    
                    <!-- Conteúdo da aba Responsáveis -->
                    <div x-show="currentTab === 'responsaveis'" class="bg-white rounded-lg shadow p-6 mb-6">
                        <h2 class="text-xl font-semibold mb-4">Detalhes por Responsável</h2>
                        <div id="grafico_responsaveis_detalhado" class="h-96"></div>
                    </div>
                    
                    <!-- Conteúdo da aba Robôs -->
                    <div x-show="currentTab === 'robos'" class="bg-white rounded-lg shadow p-6 mb-6">
                        <h2 class="text-xl font-semibold mb-4">Detalhes por Robô</h2>
                        <div id="grafico_robos_detalhado" class="h-96"></div>
                    </div>
                    
                    <!-- Conteúdo da aba Análise -->
                    <div x-show="currentTab === 'analise'" class="bg-white rounded-lg shadow p-6 mb-6">
                        <h2 class="text-xl font-semibold mb-4">Análise de Chamados por Robô</h2>
                        <div class="prose max-w-none">
                            <pre class="whitespace-pre-wrap font-sans text-base leading-relaxed">{{ analysis_result }}</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <footer class="bg-gray-800 text-white py-4">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <p class="text-center text-sm">© {{ datetime.now().year }} Melius - Dashboard de Suporte</p>
            </div>
        </footer>
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
    df = pd.read_csv("./DEAL_20250304_8327d3d5_67c735c9da7c0.csv", sep=';', encoding='utf-8')
    
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
        max_date=max_date.strftime('%Y-%m-%d'),
        datetime=datetime
    )

def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run(debug=True)

 