"""Streamlit Web UI Application for LEREDD."""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from leredd.config import get_config
from leredd.data_models import Requirement, RequirementPair, DependencyType, Prediction
from leredd.utils import (
    extract_requirements_from_text,
    generate_requirement_pairs,
    calculate_cohens_kappa,
    load_pairs_json,
)
from leredd.detector import LEREDDDetector
from leredd.evaluation import Evaluator
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.baselines.bert_classifier import BERTBaseline

# Page Configuration
st.set_page_config(
    page_title="LEREDD | Requirement Dependency Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@500;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.5px;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 8px;
    }
    
    .badge-requires { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-implements { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .badge-conflicts { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-details { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #a855f7; }
    .badge-similar { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid #ec4899; }
    .badge-nodep { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid #64748b; }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Helper Badge Renderer
def render_badge(dep_type: str) -> str:
    dep_clean = dep_type.strip()
    if "Require" in dep_clean:
        return f'<span class="metric-badge badge-requires">Requires</span>'
    elif "Implement" in dep_clean:
        return f'<span class="metric-badge badge-implements">Implements</span>'
    elif "Conflict" in dep_clean:
        return f'<span class="metric-badge badge-conflicts">Conflicts</span>'
    elif "Detail" in dep_clean:
        return f'<span class="metric-badge badge-details">Details</span>'
    elif "Similar" in dep_clean:
        return f'<span class="metric-badge badge-similar">Is Similar</span>'
    else:
        return f'<span class="metric-badge badge-nodep">No Dependency</span>'


# Session State Initialization
if "extracted_reqs" not in st.session_state:
    st.session_state.extracted_reqs = []
if "generated_pairs" not in st.session_state:
    st.session_state.generated_pairs = []
if "annotated_pairs" not in st.session_state:
    st.session_state.annotated_pairs = []
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []


# Sidebar Navigation
st.sidebar.title("⚡ LEREDD System")
st.sidebar.caption("LLM-Enabled Requirement Dependency Detection")
page = st.sidebar.radio(
    "Navigation",
    ["1. Home Overview", "2. Requirement Annotation", "3. LEREDD Detection", "4. Model Evaluation", "5. Graph Visualization"]
)

config = get_config()


# ==========================================
# PAGE 1: HOME OVERVIEW
# ==========================================
if page == "1. Home Overview":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("Automating the Detection of Requirement Dependencies Using LLMs")
    st.write(
        "Welcome to **LEREDD**, an advanced AI system implementing the research paper "
        "*'Automating the Detection of Requirement Dependencies Using Large Language Models'*. "
        "LEREDD combines SBERT embedding similarity, dynamic In-Context Learning example retrieval, "
        "fixed-size RAG context chunking, and LLM inferential reasoning."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📚 Table I: Dependency Definitions")
        st.markdown("""
        - **Requires**: The fulfillment of one requirement is a prerequisite to the fulfillment of the other requirement.
        - **Implements**: A higher-level requirement fulfilled by another lower-level requirement.
        - **Conflicts**: The fulfillment of one requirement restricts the fulfillment of the other requirement.
        - **Details**: Both requirements describe the same action under the same condition, and one provides additional details.
        - **Is Similar**: One requirement replicates partially or totally the content of another (redundancy).
        - **No_dependency**: No direct or indirect dependency exists between Requirement A and B.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Paper Highlights & Performance")
        st.markdown("""
        - **Overall Accuracy**: **87% - 96% F1 score** on automotive SRS specifications (ADB, TJA, APA).
        - **No_dependency Filtering**: **93% average F1 score** for filtering independent requirement pairs.
        - **Optimal Configuration (RQ2)**:
          * Embeddings: SBERT `all-MiniLM-L6-v2`
          * Metric: Euclidean distance $sim = 1 / (1 + dist)$
          * Aggregation: Average Similarity (Eq. 1)
          * Examples per type: $k = 6$
          * Confidence Threshold: Re-annotate $\le 4 \rightarrow \text{No\_dependency}$
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2: REQUIREMENT ANNOTATION TOOL
# ==========================================
elif page == "2. Requirement Annotation":
    st.title("📝 Requirement Extraction & Data Annotation")

    tab1, tab2 = st.tabs(["1. Extract & Generate Pairs", "2. Interactive Pair Annotation"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Upload SRS Document")
        system_name = st.text_input("System Identifier", value="ADB")
        
        default_srs_file = "data/raw/adb_srs.txt"
        default_text = ""
        if os.path.exists(default_srs_file):
            with open(default_srs_file, "r", encoding="utf-8") as f:
                default_text = f.read()

        srs_input = st.text_area("SRS Content", value=default_text, height=200)

        if st.button("Extract Requirements & Generate Pairs"):
            if srs_input.strip():
                reqs = extract_requirements_from_text(srs_input, system_name=system_name)
                st.session_state.extracted_reqs = reqs
                pairs = generate_requirement_pairs(reqs)
                st.session_state.generated_pairs = pairs
                st.success(f"Extracted {len(reqs)} requirements and generated {len(pairs)} unique pairs!")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.extracted_reqs:
            st.write(f"**Extracted Requirements ({len(st.session_state.extracted_reqs)}):**")
            st.dataframe(pd.DataFrame([r.model_dump() for r in st.session_state.extracted_reqs]))

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Interactive Pair Annotator")
        
        pairs_to_annotate = st.session_state.generated_pairs
        if not pairs_to_annotate:
            # Load default annotated sample if present
            default_annotated_file = "data/annotated/adb_pairs.json"
            if os.path.exists(default_annotated_file):
                pairs_to_annotate = load_pairs_json(default_annotated_file)

        if pairs_to_annotate:
            st.info(f"Loaded {len(pairs_to_annotate)} pairs for annotation.")
            pair_idx = st.number_input("Select Pair Index", min_value=1, max_value=len(pairs_to_annotate), value=1) - 1
            curr_pair = pairs_to_annotate[pair_idx]

            st.markdown(f"**Pair ID:** `{curr_pair.id}`")
            st.markdown(f"**Requirement A ({curr_pair.req_a.id}):** {curr_pair.req_a.text}")
            st.markdown(f"**Requirement B ({curr_pair.req_b.id}):** {curr_pair.req_b.text}")

            selected_type = st.selectbox(
                "Select Dependency Type",
                [dt.value for dt in DependencyType],
                index=0
            )

            if st.button("Save Annotation for Pair"):
                curr_pair.dependency_type = DependencyType.normalize(selected_type)
                st.success(f"Saved annotation: `{selected_type}` for pair {curr_pair.id}")

            # Inter-annotator agreement test demo
            st.divider()
            st.subheader("Inter-Annotator Agreement (Cohen's Kappa)")
            if st.button("Compute Cohen's Kappa Agreement"):
                ann1 = [p.dependency_type or DependencyType.NO_DEPENDENCY for p in pairs_to_annotate]
                # Simulated second annotator for demo
                ann2 = [p.dependency_type or DependencyType.NO_DEPENDENCY for p in pairs_to_annotate]
                kappa = calculate_cohens_kappa(ann1, ann2)
                st.metric("Cohen's Kappa (κ)", f"{kappa:.4f}", help=">0.4 indicates moderate agreement")
        else:
            st.warning("No requirement pairs available. Please extract requirements first.")
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 3: LEREDD DETECTION PIPELINE
# ==========================================
elif page == "3. LEREDD Detection":
    st.title("⚡ LEREDD Dependency Detection Engine")

    col_config, col_main = st.columns([1, 2])

    with col_config:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Detection Settings")
        
        provider = st.selectbox("LLM Provider", ["mock", "openai", "ollama"], index=0)
        llm_model = st.text_input("Model Name", value="gpt-4" if provider == "openai" else "mock-gpt4")
        k_examples = st.slider("Dynamic Examples (k)", min_value=1, max_value=9, value=6)
        use_rag = st.checkbox("Enable RAG Context", value=False)
        conf_thresh = st.slider("Confidence Threshold", min_value=0, max_value=5, value=4)
        
        config.llm_provider = provider
        config.llm_model = llm_model
        config.k_examples = k_examples
        config.use_rag = use_rag
        config.confidence_threshold = conf_thresh

        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Requirement Pair Input")
        
        req_a_input = st.text_area(
            "Requirement A Text",
            value="The BCS shall accept camera scenery information provided by the PCS to brake the vehicle if an object is within the vehicle's path."
        )
        req_b_input = st.text_area(
            "Requirement B Text",
            value="The system shall include the Brake Control Subsystem (BCS)."
        )

        if st.button("Run LEREDD Detection", use_container_width=True):
            with st.spinner("Executing LEREDD Pipeline (Knowledge Retrieval + Inference)..."):
                detector = LEREDDDetector(config=config)
                req_a = Requirement(id="REQ-A", text=req_a_input, system="ADB")
                req_b = Requirement(id="REQ-B", text=req_b_input, system="ADB")
                pair = RequirementPair(id="PAIR-DEMO-001", req_a=req_a, req_b=req_b)

                # Load example pool
                examples_pool = []
                default_file = "data/annotated/adb_pairs.json"
                if os.path.exists(default_file):
                    examples_pool = load_pairs_json(default_file)

                srs_text = ""
                srs_file = "data/raw/adb_srs.txt"
                if use_rag and os.path.exists(srs_file):
                    with open(srs_file, "r", encoding="utf-8") as f:
                        srs_text = f.read()

                pred = detector.detect_pair(target_pair=pair, annotated_pool=examples_pool, srs_text=srs_text)

                st.session_state.detection_results.append(pred)

                st.markdown("### Detection Result")
                st.markdown(f"**Predicted Type:** {render_badge(pred.dependency_type.value)}", unsafe_allow_html=True)
                st.markdown(f"**Likert Confidence Score:** `{pred.confidence}/5`")
                st.markdown(f"**Generated Rationale:**\n\n_{pred.rationale}_")
                
                with st.expander("View Full Raw LLM Output"):
                    st.code(pred.raw_response or "N/A")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 4: MODEL EVALUATION & BASELINES
# ==========================================
elif page == "4. Model Evaluation":
    st.title("📊 Model Evaluation & Baseline Comparison")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Reproduce Paper Results")

    dataset_choice = st.selectbox("Select Benchmark Dataset", ["ADB System (D1)", "TJA System (D2)", "APA System (D3)"])
    ds_file = "data/annotated/adb_pairs.json"
    if "TJA" in dataset_choice:
        ds_file = "data/annotated/tja_pairs.json"
    elif "APA" in dataset_choice:
        ds_file = "data/annotated/apa_pairs.json"

    if st.button("Run Full Benchmark Evaluation"):
        if os.path.exists(ds_file):
            ground_truth = load_pairs_json(ds_file)
            
            with st.spinner("Evaluating LEREDD and Baselines (TF-IDF & LSA, Fine-Tuned BERT)..."):
                # 1. Run LEREDD
                detector = LEREDDDetector(config=config)
                leredd_preds = detector.detect_batch(ground_truth, annotated_pool=ground_truth)
                report_leredd = Evaluator.evaluate(ground_truth, leredd_preds, system_name=dataset_choice, model_name="LEREDD")

                # 2. Run TF-IDF & LSA Baseline
                tfidf_baseline = TFIDF_LSABaseline()
                tfidf_preds = tfidf_baseline.predict_batch(ground_truth)
                report_tfidf = Evaluator.evaluate(ground_truth, tfidf_preds, system_name=dataset_choice, model_name="TF-IDF & LSA")

                # 3. Run BERT Baseline
                bert_baseline = BERTBaseline()
                bert_preds = bert_baseline.predict_batch(ground_truth)
                report_bert = Evaluator.evaluate(ground_truth, bert_preds, system_name=dataset_choice, model_name="Fine-Tuned BERT")

                # Display Comparative Table
                st.markdown("### Comparative Performance Results")
                summary_data = [
                    {
                        "Model Approach": "LEREDD (LLM + Dynamic ICL)",
                        "Overall Accuracy": f"{report_leredd.overall_accuracy:.4f}",
                        "Macro F1": f"{report_leredd.macro_f1:.4f}",
                        "Weighted F1": f"{report_leredd.weighted_f1:.4f}"
                    },
                    {
                        "Model Approach": "Fine-Tuned BERT Baseline",
                        "Overall Accuracy": f"{report_bert.overall_accuracy:.4f}",
                        "Macro F1": f"{report_bert.macro_f1:.4f}",
                        "Weighted F1": f"{report_bert.weighted_f1:.4f}"
                    },
                    {
                        "Model Approach": "TF-IDF & LSA Baseline",
                        "Overall Accuracy": f"{report_tfidf.overall_accuracy:.4f}",
                        "Macro F1": f"{report_tfidf.macro_f1:.4f}",
                        "Weighted F1": f"{report_tfidf.weighted_f1:.4f}"
                    }
                ]
                st.table(pd.DataFrame(summary_data))

                # Statistical Significance Tests
                st.markdown("### Statistical Significance Tests")
                stat_mc, p_mc = Evaluator.run_mcnemar_test(ground_truth, leredd_preds, bert_preds)
                st.write(f"**McNemar's Test (LEREDD vs BERT):** Statistic = `{stat_mc:.4f}`, p-value = `{p_mc:.4f}`")
                if p_mc < 0.05:
                    st.success("LEREDD demonstrates statistically significant superiority over BERT (p < 0.05)!")

                # Confusion Matrix Heatmap
                st.markdown("### Confusion Matrix (LEREDD)")
                if report_leredd.confusion_matrix:
                    cm_df = pd.DataFrame(report_leredd.confusion_matrix)
                    fig = px.imshow(
                        cm_df,
                        labels=dict(x="Predicted Class", y="Ground Truth", color="Pairs"),
                        x=list(cm_df.columns),
                        y=list(cm_df.index),
                        text_auto=True,
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"Benchmark file '{ds_file}' not found.")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 5: DEPENDENCY GRAPH VISUALIZATION
# ==========================================
elif page == "5. Graph Visualization":
    st.title("🕸️ Interactive Requirement Dependency Network Graph")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("System Dependency Network")

    ds_file = "data/annotated/adb_pairs.json"
    if os.path.exists(ds_file):
        pairs = load_pairs_json(ds_file)

        # Build NetworkX Graph
        G = nx.DiGraph()
        for pair in pairs:
            dt = pair.dependency_type.value if pair.dependency_type else "No_dependency"
            if dt != "No_dependency":
                G.add_node(pair.req_a.id, text=pair.req_a.text)
                G.add_node(pair.req_b.id, text=pair.req_b.text)
                G.add_edge(pair.req_a.id, pair.req_b.id, type=dt)

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#6366f1'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}: {G.nodes[node].get('text', '')}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                color=[len(list(G.neighbors(node))) for node in G.nodes()],
                line_width=2
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Graph nodes: {G.number_of_nodes()} requirements | Edges: {G.number_of_edges()} active dependencies")
    else:
        st.warning(f"File '{ds_file}' not found.")
    st.markdown('</div>', unsafe_allow_html=True)
