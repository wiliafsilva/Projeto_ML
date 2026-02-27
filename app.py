
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.analysis import (
    prepare_evaluation_data, evaluate_model,
    plot_confusion_matrix, plot_roc, plot_precision_recall,
    plot_calibration_curve, plot_feature_importance
)
from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats

st.set_page_config(layout="wide")

st.title("Scientific Replica – Previsão da EPL (Streamlit)")

@st.cache_data
def load_models():
    try:
        results_metadata = joblib.load("models/trained_models.pkl")
        # Extrair apenas os modelos para compatibilidade
        return results_metadata.get('models', results_metadata)
    except Exception:
        return {}

models = load_models()

page = st.sidebar.selectbox("Navegação", [
    "Visão Geral",
    "Comparação de Modelos",
    "Avaliação e Métricas",
    "Análise Científica Consolidada",
    "Ajuste de Hiperparâmetros",
    "Distribuições e Importância de Features"
])

if page == "Visão Geral":
    st.header("Conjunto de Dados e Atributos")
    df = load_all_data()
    
    # Criar coluna com resultado legível
    resultado_map = {'H': 'Vitória Casa', 'D': 'Empate', 'A': 'Vitória Visitante'}
    df['Resultado_Legivel'] = df['FTR'].map(resultado_map)
    
    # Renomear colunas para exibição amigável (apenas as que existem)
    colunas_legiveis = {
        "Date": "Data",
        "HomeTeam": "Time da Casa",
        "FTHG": "Gols Casa",
        "FTAG": "Gols Visitante",
        "AwayTeam": "Time Visitante",
        "Resultado_Legivel": "Resultado Final",
        "Season": "Temporada"
    }
    df_exibicao = df.rename(columns=colunas_legiveis)
    
    st.subheader("📊 Estatísticas Gerais do Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Partidas", f"{len(df):,}")
    with col2:
        st.metric("Temporadas", f"{df['Season'].nunique()}")
    with col3:
        st.metric("Times Únicos", f"{pd.concat([df['HomeTeam'], df['AwayTeam']]).nunique()}")
    with col4:
        avg_goals = (df['FTHG'] + df['FTAG']).mean()
        st.metric("Média Gols/Jogo", f"{avg_goals:.2f}")
    
    # Informação sobre a divisão treino/teste
    st.info(f"""
    📋 **Metodologia do Artigo Científico:**
    - **Período de Treinamento:** 2005-2014 (9 temporadas)
    - **Período de Teste:** 2014-2016 (2 temporadas)
    - **Total:** {len(df):,} partidas ({df['Season'].min()}-{df['Season'].max()})
    """)
    
    # Distribuição de Resultados
    st.subheader("✓ Distribuição de Resultados")
    result_counts = df['FTR'].value_counts()
    total_games = len(df)
    
    # Resultados das partidas
    col1, col2, col3 = st.columns(3)
    with col1:
        home_wins = result_counts.get('H', 0)
        home_pct = (home_wins / total_games) * 100
        st.metric("🏠 Vitória Casa", f"{home_wins:,}", f"{home_pct:.1f}%")
    with col2:
        draws = result_counts.get('D', 0)
        draw_pct = (draws / total_games) * 100
        st.metric("🤝 Empate", f"{draws:,}", f"{draw_pct:.1f}%")
    with col3:
        away_wins = result_counts.get('A', 0)
        away_pct = (away_wins / total_games) * 100
        st.metric("✈️ Vitória Visitante", f"{away_wins:,}", f"{away_pct:.1f}%")
    
    # Distribuição de Gols (3 colunas para melhor alinhamento)
    st.write("")  # Espaçamento
    total_home_goals = df['FTHG'].sum()
    total_away_goals = df['FTAG'].sum()
    total_goals = total_home_goals + total_away_goals
    home_goals_pct = (total_home_goals / total_goals) * 100
    away_goals_pct = (total_away_goals / total_goals) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_goals_display = int(total_goals)
        st.metric("⚽ Total de Gols", f"{total_goals_display:,}", "")
    with col2:
        st.metric("⚽ Gols em Casa", f"{int(total_home_goals):,}", f"{home_goals_pct:.1f}%")
    with col3:
        st.metric("⚽ Gols Fora de Casa", f"{int(total_away_goals):,}", f"{away_goals_pct:.1f}%")
    
    st.subheader("Amostra do conjunto de dados (primeiros 10 jogos)")
    colunas_mostrar = ['Data', 'Time da Casa', 'Gols Casa', 'Gols Visitante', 'Time Visitante', 'Resultado Final', 'Temporada']
    st.dataframe(df_exibicao[colunas_mostrar].head(10))
    
    features = calculate_team_stats(df)
    
    # Criar coluna com resultado legível nas features
    resultado_numerico_map = {0: 'Vitória Casa', 1: 'Empate', 2: 'Vitória Visitante'}
    features['Resultado_Legivel'] = features['Result'].map(resultado_numerico_map)
    
    # Renomear colunas das features para exibição amigável
    colunas_features_legiveis = {
        "gd_diff": "Diferença de Saldo de Gols",
        "streak_diff": "Diferença de Sequência",
        "weighted_diff": "Diferença Ponderada",
        "Resultado_Legivel": "Resultado Final",
        "Season": "Temporada"
    }
    features_exibicao = features.rename(columns=colunas_features_legiveis)
    
    st.subheader("Features Geradas")
    st.info("💡 **Importante:** Os primeiros jogos de cada temporada têm features zeradas porque os times ainda não têm histórico acumulado. As features começam a ter valores após alguns jogos.")
    
    # Mostrar estatísticas das features
    st.write("**Estatísticas das Features:**")
    stats = features[['gd_diff', 'streak_diff', 'weighted_diff']].describe().transpose()
    stats.index = ['Diferença de Saldo de Gols', 'Diferença de Sequência', 'Diferença Ponderada']
    st.dataframe(stats.style.format("{:.3f}"))
    
    # Contagem de zeros
    zeros_gd = (features['gd_diff'] == 0).sum()
    zeros_streak = (features['streak_diff'] == 0).sum()
    zeros_weighted = (features['weighted_diff'] == 0).sum()
    total = len(features)
    
    st.write(f"**Percentual de valores zero:** Diferença Saldo: {zeros_gd/total*100:.1f}% | Diferença Sequência: {zeros_streak/total*100:.1f}% | Diferença Ponderada: {zeros_weighted/total*100:.1f}%")
    
    # Mostrar amostras diferentes
    tab1, tab2, tab3 = st.tabs(["Primeiros 10 Jogos (Semana 1)", "Jogos 11-20 (Com Histórico)", "Jogos Aleatórios"])
    
    colunas_features_mostrar = ['Diferença de Saldo de Gols', 'Diferença de Sequência', 'Diferença Ponderada', 'Resultado Final', 'Temporada']
    
    with tab1:
        st.write("Jogos iniciais da primeira temporada (features zeradas = esperado)")
        st.dataframe(features_exibicao[colunas_features_mostrar].head(10))
    
    with tab2:
        st.write("Jogos após acúmulo de histórico (features não-zeradas)")
        st.dataframe(features_exibicao[colunas_features_mostrar].iloc[11:21])
    
    with tab3:
        st.write("Amostra aleatória de 15 jogos")
        st.dataframe(features_exibicao[colunas_features_mostrar].sample(15, random_state=42).sort_index())
    
    st.write("**Temporadas nos dados:**", sorted(df['Season'].unique()))

if page == "Comparação de Modelos":
    st.header("Resumo dos Modelos Treinados")
    if not models:
        st.warning("Nenhum modelo treinado encontrado. Execute `main.py` para treinar e gerar `models/trained_models.pkl`.")
    else:
        data = []
        for name, info in models.items():
            data.append([name, info.get('accuracy'), info.get('f1'), info.get('rps')])
        dfm = pd.DataFrame(data, columns=["Model","Accuracy","F1","RPS"])
        st.dataframe(dfm.set_index('Model'))
        st.subheader("Gráfico de barras: acurácia média de teste")
        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x='Model', y='Accuracy', data=dfm, ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)

if page == "Avaliação e Métricas":
    st.header("Avaliação do Modelo e Visualizações")
    X_test, y_test = prepare_evaluation_data()
    if not models:
        st.warning("Nenhum modelo treinado encontrado. Execute `main.py` para treinar os modelos primeiro.")
    else:
        model_names = list(models.keys())
        sel = st.selectbox("Selecione o modelo", model_names)
        info = models[sel]
        st.subheader(f"Métricas para {sel}")
        ev = evaluate_model(info['model'], X_test, y_test)
        # show classification report
        cr = ev['classification_report']
        st.write(pd.DataFrame(cr).transpose())
        
        # Mostrar distribuição com nomes legíveis
        classes_legiveis = {0: 'Vitória Casa', 1: 'Empate', 2: 'Vitória Visitante'}
        y_test_dist = pd.Series(ev.get('y_test_counts')).rename(classes_legiveis)
        preds_dist = pd.Series(ev.get('preds_counts')).rename(classes_legiveis)
        
        st.write("**Distribuição real das classes:**", dict(y_test_dist))
        st.write("**Distribuição prevista das classes:**", dict(preds_dist))
        st.write("Pontuação Brier:", ev['brier_score'])
        if ev['roc_auc'] is not None:
            st.write("ROC AUC (macro):", ev['roc_auc'])

        # plots
        st.subheader("Matriz de Confusão")
        fig_cm = plot_confusion_matrix(ev['confusion_matrix'])
        st.pyplot(fig_cm)

        st.subheader("Curvas ROC")
        fig_roc = plot_roc(ev['probs'], y_test)
        st.pyplot(fig_roc)

        st.subheader("Precisão-Recall")
        fig_pr = plot_precision_recall(ev['probs'], y_test)
        st.pyplot(fig_pr)

        st.subheader("Calibração")
        fig_cal = plot_calibration_curve(ev['probs'], y_test)
        st.pyplot(fig_cal)

        st.subheader("Importância das Features (se disponível)")
        feat_plot = plot_feature_importance(info['model'], X_test.columns)
        if feat_plot is not None:
            st.pyplot(feat_plot)
        else:
            st.write("Importância das features não disponível para este modelo.")

if page == "Análise Científica Consolidada":
    st.header("📊 Análise Científica Consolidada")
    st.markdown("Esta seção apresenta tabelas e visualizações consolidadas para inclusão no artigo científico.")
    
    # Verificar se os arquivos existem
    import os
    import subprocess
    tabelas_existem = os.path.exists('models/tabela1_resumo_dataset.csv')
    figuras_existem = os.path.exists('models/figures/fig1_radar_comparison.png')
    
    # Botões para gerar tabelas e figuras
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Gerar Tabelas", use_container_width=True):
            with st.spinner("Gerando tabelas..."):
                try:
                    result = subprocess.run(
                        ["python", "scripts/generate_tables.py"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        st.success("✅ Tabelas geradas com sucesso!")
                        st.rerun()
                    else:
                        st.error(f"❌ Erro ao gerar tabelas: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Timeout: O processo demorou muito.")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
    
    with col2:
        if st.button("🎨 Gerar Figuras", use_container_width=True):
            with st.spinner("Gerando figuras (pode demorar)..."):
                try:
                    result = subprocess.run(
                        ["python", "scripts/generate_figures.py"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        st.success("✅ Figuras geradas com sucesso!")
                        st.rerun()
                    else:
                        st.error(f"❌ Erro ao gerar figuras: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Timeout: O processo demorou muito.")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
    
    with col3:
        st.info(f"📊 Status: {'✅ Tabelas OK' if tabelas_existem else '❌ Tabelas faltando'} | {'✅ Figuras OK' if figuras_existem else '❌ Figuras faltando'}")
    
    st.markdown("---")
    
    if not tabelas_existem or not figuras_existem:
        st.warning("⚠️ Tabelas e/ou figuras não encontradas. Use os botões acima ou execute:")
        col_cmd1, col_cmd2 = st.columns(2)
        with col_cmd1:
            st.code("python scripts\\generate_tables.py", language="bash")
        with col_cmd2:
            st.code("python scripts\\generate_figures.py", language="bash")
    
    # Se os arquivos existirem, mostrar as tabelas e figuras
    if tabelas_existem and figuras_existem:
        # ============================================================
        # TABELAS
        # ============================================================
        st.subheader("📋 Tabelas Consolidadas")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Resumo Dataset", 
            "Estatísticas Features", 
            "Comparação Modelos",
            "Performance por Temporada",
            "Classificação Detalhada"
        ])
        
        with tab1:
            st.markdown("**Tabela 1: Resumo Completo do Dataset**")
            df_tab1 = pd.read_csv('models/tabela1_resumo_dataset.csv')
            st.dataframe(df_tab1, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download CSV",
                df_tab1.to_csv(index=False).encode('utf-8'),
                "tabela1_resumo_dataset.csv",
                "text/csv"
            )
        
        with tab2:
            st.markdown("**Tabela 2: Estatísticas Descritivas das Features**")
            df_tab2 = pd.read_csv('models/tabela2_estatisticas_features.csv')
            st.dataframe(df_tab2, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download CSV",
                df_tab2.to_csv(index=False).encode('utf-8'),
                "tabela2_estatisticas_features.csv",
                "text/csv"
            )
        
        with tab3:
            st.markdown("**Tabela 3: Comparação Completa de Modelos**")
            df_tab3 = pd.read_csv('models/tabela3_comparacao_modelos.csv')
            st.dataframe(df_tab3, use_container_width=True, hide_index=True)
            st.info("💡 Note que o **Baseline** (prever sempre a classe majoritária) serve como referência mínima de performance.")
            st.download_button(
                "📥 Download CSV",
                df_tab3.to_csv(index=False).encode('utf-8'),
                "tabela3_comparacao_modelos.csv",
                "text/csv"
            )
        
        with tab4:
            st.markdown("**Tabela 5: Performance por Temporada de Teste**")
            df_tab5 = pd.read_csv('models/tabela5_performance_temporada.csv')
            st.dataframe(df_tab5, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download CSV",
                df_tab5.to_csv(index=False).encode('utf-8'),
                "tabela5_performance_temporada.csv",
                "text/csv"
            )
        
        with tab5:
            st.markdown("**Tabela 6: Relatório de Classificação Detalhado por Modelo**")
            modelo_sel = st.selectbox("Selecione o modelo:", ['SVM', 'RandomForest', 'XGBoost'])
            df_tab6 = pd.read_csv(f'models/tabela6_classificacao_{modelo_sel.lower()}.csv')
            st.dataframe(df_tab6, use_container_width=True)
            st.download_button(
                "📥 Download CSV",
                df_tab6.to_csv().encode('utf-8'),
                f"tabela6_classificacao_{modelo_sel.lower()}.csv",
                "text/csv"
            )
        
        st.markdown("---")
        
        # ============================================================
        # FIGURAS
        # ============================================================
        st.subheader("📈 Visualizações Científicas")
        
        fig_tab1, fig_tab2, fig_tab3, fig_tab4, fig_tab5, fig_tab6 = st.tabs([
            "Radar Multi-Métrica",
            "Correlação Features",
            "Boxplots por Resultado",
            "Feature Importance",
            "Calibração",
            "Comparação Barras"
        ])
        
        with fig_tab1:
            st.markdown("**Figura 1: Comparação Multi-Métrica (Radar Chart)**")
            st.image('models/figures/fig1_radar_comparison.png', use_container_width=True)
            st.caption("Comparação visual de todas as métricas de performance dos três modelos. Quanto mais próximo da borda externa, melhor a performance.")
        
        with fig_tab2:
            st.markdown("**Figura 2: Matriz de Correlação entre Features**")
            st.image('models/figures/fig2_feature_correlation.png', use_container_width=True)
            st.caption("Heatmap mostrando a correlação linear entre as três features utilizadas. Valores próximos de 1/-1 indicam forte correlação positiva/negativa.")
        
        with fig_tab3:
            st.markdown("**Figura 3: Distribuição das Features por Resultado**")
            st.image('models/figures/fig3_boxplots_by_result.png', use_container_width=True)
            st.caption("Boxplots mostrando como cada feature se distribui por tipo de resultado (Vitória Casa, Empate, Vitória Visitante).")
        
        with fig_tab4:
            st.markdown("**Figura 4: Comparação de Importância de Features**")
            st.image('models/figures/fig4_feature_importance_comparison.png', use_container_width=True)
            st.caption("Comparação lado a lado da importância das features segundo Random Forest e XGBoost.")
        
        with fig_tab5:
            st.markdown("**Figura 5: Curvas de Calibração por Classe**")
            st.image('models/figures/fig5_calibration_comparison.png', use_container_width=True)
            st.caption("Análise de calibração das probabilidades preditas. Curvas próximas da diagonal indicam boa calibração.")
        
        with fig_tab6:
            st.markdown("**Figura 6: Comparação de Métricas (Barras Agrupadas)**")
            st.image('models/figures/fig6_metrics_comparison_bars.png', use_container_width=True)
            st.caption("Visualização comparativa das principais métricas (Accuracy, Precision, Recall, F1-Score) entre os três modelos.")
        
        st.markdown("---")
        st.success("✅ Todas as tabelas e figuras foram geradas e podem ser exportadas para o artigo científico!")
        
        with st.expander("ℹ️ Como usar essas tabelas e figuras"):
            st.markdown("""
            **Para o Artigo Científico:**
            1. As **tabelas CSV** podem ser formatadas em LaTeX usando ferramentas online
            2. As **figuras PNG** (300 DPI) estão prontas para inclusão direta
            3. Cada figura tem legenda descritiva para referência
            
            **Scripts de Geração:**
            - `python scripts/generate_tables.py` - Regenerar tabelas
            - `python scripts/generate_figures.py` - Regenerar figuras
            
            **Localização dos Arquivos:**
            - Tabelas: `models/tabela*.csv`
            - Figuras: `models/figures/fig*.png`
            """)

if page == "Ajuste de Hiperparâmetros":
    st.header("Ajuste rápido de hiperparâmetros (grades pequenas)")
    df = load_all_data()
    features = calculate_team_stats(df)
    train = features[features['Season'] <= 2014]  # Treino: 2005-2014
    X_train = train.drop(['Result','Season'], axis=1)
    y_train = train['Result']

    tune_model = st.selectbox("Escolha o modelo para ajustar", ["SVM","RandomForest","XGBoost"])
    if st.button("Executar ajuste"):
        with st.spinner("Executando busca em grade (pode levar um momento)..."):
            st.info("Ajuste iniciado — pode demorar. Verifique o terminal para logs do GridSearch.")
            if tune_model == 'SVM':
                param_grid = {'C':[0.5,1,2], 'gamma':[0.01,0.05,0.1]}
                gs = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            elif tune_model == 'RandomForest':
                param_grid = {'n_estimators':[50,100], 'max_depth':[5,10,None]}
                gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            else:
                param_grid = {'learning_rate':[0.01,0.05], 'max_depth':[3,6]}
                gs = GridSearchCV(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            try:
                gs.fit(X_train, y_train)
            except Exception as e:
                st.error("O ajuste falhou — verifique os logs para detalhes.")
                st.exception(e)
            else:
                st.write("Melhores parâmetros:", gs.best_params_)
                results = pd.DataFrame(gs.cv_results_)
                st.dataframe(results[['params','mean_test_score','std_test_score']])
            # small heatmap if 2 params
            if len(param_grid) == 2:
                p1, p2 = list(param_grid.keys())
                # build pivot table
                results['p1'] = results['param_' + p1].astype(str)
                results['p2'] = results['param_' + p2].astype(str)
                pivot = results.pivot(index='p1', columns='p2', values='mean_test_score')
                fig, ax = plt.subplots()
                sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax)
                ax.set_xlabel(p2)
                ax.set_ylabel(p1)
                st.pyplot(fig)

if page == "Distribuições e Importância de Features":
    st.header("Distribuições das features e importância")
    df = load_all_data()
    features = calculate_team_stats(df)
    st.subheader("Distribuições univariadas")
    cols = ['gd_diff','streak_diff','weighted_diff']
    for c in cols:
        fig, ax = plt.subplots(figsize=(3,2))
        sns.kdeplot(features[c].dropna(), shade=True, ax=ax)
        ax.set_title(c)
        st.pyplot(fig)

    st.subheader("Importância das features pelo RandomForest")
    if 'RandomForest' in models:
        rf = models['RandomForest']['model']
        fi_fig = plot_feature_importance(rf, features.drop(['Result','Season'],axis=1).columns)
        if fi_fig is not None:
            st.pyplot(fi_fig)
    else:
        st.write("Treine o RandomForest (execute main.py) para mostrar importâncias.")
