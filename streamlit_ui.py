"""
Streamlit UI для агентной системы анализа longevity-интервенций
Запуск: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import json
from typing import Dict, Any
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Импорт агентной системы (предполагается, что в том же каталоге)
try:
    from langgraph_agent_system import LongevityAgentSystem
    from world_model import WorldModel
except ImportError:
    st.error("Не удалось импортировать модули. Убедитесь, что все файлы на месте.")
    st.stop()


# ============ КОНФИГУРАЦИЯ ============

st.set_page_config(
    page_title="Longevity Agent System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============ ИНИЦИАЛИЗАЦИЯ ============

@st.cache_resource
def init_agent_system():
    """Инициализация агентной системы (кэшируется)"""
    try:
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY не установлен в переменных окружения!")
            st.stop()
        
        system = LongevityAgentSystem(
            world_model_path="ontology.yaml",
            llm_model="gpt-4o",
            temperature=0.0
        )
        return system
    except Exception as e:
        st.error(f"Ошибка инициализации системы: {str(e)}")
        st.stop()


@st.cache_resource
def load_world_model():
    """Загрузка World Model для визуализации"""
    return WorldModel("ontology.yaml")


# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============

def create_knowledge_graph_viz(world_model: WorldModel, intervention_id: str = None):
    """Создает интерактивную визуализацию графа знаний"""
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()
    
    # Если указана интервенция - показываем её окружение
    if intervention_id:
        iv = world_model.get_intervention(intervention_id)
        if iv:
            # Узел интервенции
            net.add_node(
                iv.id, 
                label=iv.name, 
                color="#FF6B6B",
                size=30,
                title=f"Type: {iv.type}"
            )
            
            # Таргеты (гены)
            for gene_id in iv.targets:
                gene = world_model.genes.get(gene_id)
                if gene:
                    net.add_node(
                        gene_id,
                        label=gene_id,
                        color="#4ECDC4",
                        size=20,
                        title=f"Species: {', '.join(gene.species)}"
                    )
                    net.add_edge(iv.id, gene_id, label="targets")
                    
                    # Пути
                    for pw_id in gene.pathways:
                        pw = world_model.pathways.get(pw_id)
                        if pw:
                            net.add_node(
                                pw_id,
                                label=pw.name,
                                color="#95E1D3",
                                size=15,
                                title=f"Conserved: {pw.conserved_human_mouse}"
                            )
                            net.add_edge(gene_id, pw_id, label="regulates")
            
            # Виды, на которых тестировалось
            for effect in iv.effects:
                species_node = f"species_{effect.species}"
                net.add_node(
                    species_node,
                    label=effect.species,
                    color="#F38181",
                    size=15,
                    title=f"Effect: {effect.lifespan_change_pct}%"
                )
                net.add_edge(iv.id, species_node, label="tested_on")
    
    # Сохранение в HTML
    net.save_graph("temp_graph.html")
    
    with open("temp_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    return html


def create_radar_chart(analysis: Dict[str, Any]):
    """Создает radar chart для метрик"""
    
    # Пример метрик (можно расширить)
    categories = [
        'Human Relevance Score',
        'Confidence',
        'Mechanistic Understanding',
        'Evidence Quality',
        'Safety Profile'
    ]
    
    # Нормализация HRS к шкале 0-5
    hrs_normalized = analysis['human_relevance_score'] / 20
    
    # Confidence mapping
    confidence_map = {"high": 5, "medium": 3, "low": 1}
    confidence_score = confidence_map.get(analysis['confidence'], 2.5)
    
    # Пример значений (в реальной системе - из данных)
    values = [
        hrs_normalized,
        confidence_score,
        4,  # placeholder
        3.5,  # placeholder
        4,  # placeholder
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=analysis['intervention_id']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False,
        height=400
    )
    
    return fig


def render_experiment_card(exp: Dict[str, Any]):
    """Рендерит карточку эксперимента"""
    priority_colors = {1: "🔴", 2: "🟡", 3: "🟢"}
    priority_emoji = priority_colors.get(exp['priority'], "⚪")
    
    with st.expander(f"{priority_emoji} Priority {exp['priority']}: {exp['experiment_type']}", expanded=False):
        st.markdown(f"**Design:** {exp['design_summary']}")
        st.markdown(f"**Addresses gap:** {exp['addresses_gap']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Uncertainty Reduction", f"{exp['expected_uncertainty_reduction']}%")
        with col2:
            st.metric("Cost", exp['estimated_cost'])
        with col3:
            st.metric("Duration", exp['estimated_duration'])


# ============ ГЛАВНЫЙ ИНТЕРФЕЙС ============

def main():
    # Заголовок
    st.title("🧬 Longevity Agent System")
    st.markdown("*Агентная система для анализа трансляционного потенциала longevity-интервенций*")
    
    # Инициализация
    agent_system = init_agent_system()
    world_model = load_world_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        st.subheader("Доступные интервенции")
        intervention_list = [iv.name for iv in world_model.interventions.values()]
        st.info(f"В базе: {len(intervention_list)} интервенций")
        with st.expander("Показать список"):
            for name in sorted(intervention_list):
                st.text(f"• {name}")
        
        st.markdown("---")
        
        st.subheader("Примеры запросов")
        example_queries = [
            "Проанализируй рапамицин",
            "Сравни метформин и калорийную рестрикцию",
            "Какие эксперименты нужны для валидации сенолитиков?",
            "Оцени потенциал NMN для человека",
        ]
        
        for eq in example_queries:
            if st.button(eq, key=eq):
                st.session_state.query = eq
    
    # Основной интерфейс
    st.header("💬 Введите запрос")
    
    query = st.text_area(
        "Ваш вопрос о longevity-интервенциях:",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="Например: Проанализируй потенциал рапамицина для продления жизни человека"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        analyze_button = st.button("🚀 Анализировать", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Очистить", use_container_width=True)
    
    if clear_button:
        st.session_state.clear()
        st.rerun()
    
    # Обработка запроса
    if analyze_button and query:
        with st.spinner("🤖 Агенты обрабатывают запрос..."):
            try:
                result = agent_system.process_query(query)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Ошибка обработки: {str(e)}")
                st.exception(e)
                return
    
    # Отображение результатов
    if "result" in st.session_state:
        result = st.session_state.result
        
        # Tabs для организации результатов
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Отчет",
            "🔬 Детальный анализ", 
            "🧪 Эксперименты",
            "🕸️ Граф знаний"
        ])
        
        # TAB 1: Отчет
        with tab1:
            if result.get("errors"):
                st.error("Обнаружены ошибки:")
                for error in result["errors"]:
                    st.error(error)
            
            st.markdown("### 📋 Полный отчет")
            st.markdown(result["report"])
            
            # Кнопка скачивания
            st.download_button(
                label="📥 Скачать отчет (Markdown)",
                data=result["report"],
                file_name="longevity_report.md",
                mime="text/markdown"
            )
        
        # TAB 2: Детальный анализ
        with tab2:
            st.markdown("### 🔬 Детальный анализ интервенций")
            
            analyses = result.get("analyses", [])
            
            if not analyses:
                st.warning("Нет данных для анализа")
            else:
                for analysis in analyses:
                    st.markdown(f"## {analysis['intervention_id'].upper()}")
                    
                    # Метрики в колонках
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Human Relevance Score",
                            f"{analysis['human_relevance_score']:.1f}/100"
                        )
                    
                    with col2:
                        confidence_emoji = {
                            "high": "🟢",
                            "medium": "🟡",
                            "low": "🔴"
                        }
                        st.metric(
                            "Confidence",
                            f"{confidence_emoji.get(analysis['confidence'], '⚪')} {analysis['confidence']}"
                        )
                    
                    with col3:
                        st.metric("Bottom Line", "✓" if analysis['human_relevance_score'] > 60 else "⚠")
                    
                    # Radar chart
                    try:
                        fig = create_radar_chart(analysis)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Не удалось построить radar chart: {e}")
                    
                    # Сильные стороны и риски
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Сильные стороны:**")
                        for strength in analysis['key_strengths']:
                            st.success(strength)
                    
                    with col2:
                        st.markdown("**⚠️ Риски и ограничения:**")
                        for concern in analysis['key_concerns']:
                            st.warning(concern)
                    
                    st.markdown("**🧠 Механистическое обоснование:**")
                    st.info(analysis['mechanistic_reasoning'])
                    
                    st.markdown("**📌 Bottom Line:**")
                    st.markdown(f"> {analysis['bottom_line']}")
                    
                    st.markdown("---")
        
        # TAB 3: Эксперименты
        with tab3:
            st.markdown("### 🧪 Предлагаемые эксперименты")
            
            plans = result.get("experiment_plans", [])
            
            if not plans:
                st.warning("Нет планов экспериментов")
            else:
                for plan in plans:
                    st.markdown(f"## {plan['intervention_name']}")
                    
                    experiments = plan.get("experiments", [])
                    
                    if not experiments:
                        st.info("Нет предложенных экспериментов")
                        continue
                    
                    # Сортировка по приоритету
                    experiments_sorted = sorted(experiments, key=lambda x: x['priority'])
                    
                    for exp in experiments_sorted:
                        render_experiment_card(exp)
                    
                    st.markdown("---")
        
        # TAB 4: Граф знаний
        with tab4:
            st.markdown("### 🕸️ Граф знаний")
            
            entities = result.get("entities", {})
            intervention_names = entities.get("intervention_names", [])
            
            if intervention_names:
                selected_iv = st.selectbox(
                    "Выберите интервенцию для визуализации:",
                    intervention_names
                )
                
                # Поиск ID интервенции
                iv = world_model.find_intervention_by_name(selected_iv)
                
                if iv:
                    try:
                        html = create_knowledge_graph_viz(world_model, iv.id)
                        components.html(html, height=600)
                    except Exception as e:
                        st.error(f"Ошибка визуализации графа: {e}")
                        st.info("Установите библиотеку pyvis: pip install pyvis")
            else:
                st.info("Для визуализации графа необходимо проанализировать интервенцию")
        
        # Логи (опционально, в expander)
        with st.expander("🔍 Логи выполнения агентов"):
            for log in result.get("logs", []):
                st.text(log)


if __name__ == "__main__":
    main()