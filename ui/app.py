"""Streamlit Web UI Application for LEREDD."""

import os
import json
import datetime
import tempfile
import uuid
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from leredd.config import get_config
from leredd.data_models import (
    AnnotationProvenance, AnnotationSource, Requirement, RequirementPair,
    DependencyType, Prediction, VerificationStatus,
)
from leredd.utils import (
    extract_requirements_from_text,
    generate_requirement_pairs,
    load_pairs_json,
    save_pairs_json,
)
from leredd.detector import LEREDDDetector
from scripts.run_full_evaluation import run_evaluation
from leredd.review_queue import (
    ReviewQueueManager,
    QueueStatus,
    ReviewReasonCode,
    ReviewPriority,
    ReviewQueueConfig,
)

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


def validate_ui_output_path(output_path: str, overwrite: bool = False) -> Path:
    """Resolve an output path while protecting source datasets and existing files."""
    target = Path(output_path).expanduser().resolve()
    project_data = (Path.cwd() / "data").resolve()
    for protected_name in ("annotated", "raw", "replication", "splits"):
        protected = project_data / protected_name
        if target == protected or protected in target.parents:
            raise ValueError(f"Cannot write annotation output inside source data: {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; enable overwrite to replace it: {target}")
    return target


def persist_ui_pairs(pairs, target: Path) -> None:
    """Write pair JSON atomically so an interrupted save cannot truncate prior output."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
        save_pairs_json(pairs, temp_name)
        os.replace(temp_name, target)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def save_ui_annotation(
    pairs, pair_index: int, label: str, reviewer_id: str, output_path: str,
    overwrite: bool = False,
):
    """Persist one explicit human label to a separate, protected output file."""
    if not reviewer_id.strip():
        raise ValueError("Reviewer identifier must not be empty.")
    target = validate_ui_output_path(output_path, overwrite=overwrite)
    if not 0 <= pair_index < len(pairs):
        raise IndexError("Selected pair index is outside the annotation dataset.")

    updated_pairs = [pair.model_copy(deep=True) for pair in pairs]
    pair = updated_pairs[pair_index]
    prior_label = pair.dependency_type
    accepted_label = DependencyType.normalize(label)
    prior_provenance = pair.provenance or AnnotationProvenance()
    pair.dependency_type = accepted_label
    pair.provenance = prior_provenance.model_copy(update={
        "source": AnnotationSource.HUMAN,
        "status": VerificationStatus.VERIFIED,
        "annotator_id": reviewer_id.strip(),
        "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "predicted_label": prior_provenance.predicted_label or prior_label,
        "accepted_label": accepted_label,
    })

    persist_ui_pairs(updated_pairs, target)
    return updated_pairs


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
    ["1. Home Overview", "2. Requirement Annotation", "3. LEREDD Detection", "4. Model Evaluation", "5. Graph Visualization", "6. Human Review Queue"]
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
        st.markdown(r"""
        - The evaluation page reports only results generated from this checkout's fixed system-disjoint split.
        - The included legacy evaluation set has 20 pairs; its results are demonstrative and do not establish paper-level performance.
        - LEREDD evaluations in this local workflow use MockLLM unless separately configured and authorized.
        - Reference configuration: SBERT `all-MiniLM-L6-v2`, average similarity, and confidence thresholding at 4.
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
        uploaded_srs = st.file_uploader("Upload a text SRS (.txt or .md)", type=["txt", "md"])

        default_srs_file = "data/raw/adb_srs.txt"
        default_text = ""
        if os.path.exists(default_srs_file):
            with open(default_srs_file, "r", encoding="utf-8") as f:
                default_text = f.read()

        srs_input = st.text_area("SRS Content", value=default_text, height=200)
        candidates_output = st.text_input(
            "Generated Candidate JSON Output",
            value="data/provisional/ui_candidates.json",
        )
        replace_candidates = st.checkbox("Replace this existing candidate output file")

        if st.button("Extract Requirements & Generate Pairs"):
            try:
                document_text = uploaded_srs.getvalue().decode("utf-8") if uploaded_srs else srs_input
            except UnicodeDecodeError:
                st.error("The uploaded document is not valid UTF-8 text.")
                document_text = ""
            if document_text.strip():
                try:
                    reqs = extract_requirements_from_text(document_text, system_name=system_name)
                    pairs = generate_requirement_pairs(reqs)
                    safe_output = validate_ui_output_path(
                        candidates_output, overwrite=replace_candidates
                    )
                    persist_ui_pairs(pairs, safe_output)
                    st.session_state.extracted_reqs = reqs
                    st.session_state.generated_pairs = pairs
                    st.session_state.annotated_pairs = []
                    st.success(
                        f"Extracted {len(reqs)} requirements and generated {len(pairs)} "
                        f"provisional pairs. Saved to `{safe_output}`."
                    )
                except Exception as exc:
                    st.error(f"Could not generate or save candidate pairs: {exc}")
            else:
                st.warning("Provide non-empty SRS content or upload a UTF-8 .txt/.md document.")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.extracted_reqs:
            st.write(f"**Extracted Requirements ({len(st.session_state.extracted_reqs)}):**")
            st.dataframe(pd.DataFrame([r.model_dump() for r in st.session_state.extracted_reqs]))

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Interactive Pair Annotator")

        pairs_to_annotate = st.session_state.generated_pairs or st.session_state.annotated_pairs
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

            labels = [dt.value for dt in DependencyType]
            current_label = curr_pair.dependency_type.value if curr_pair.dependency_type else None
            selected_type = st.selectbox(
                "Select Dependency Type",
                labels,
                index=labels.index(current_label) if current_label in labels else None,
                placeholder="Choose a label",
            )
            reviewer_id = st.text_input("Annotator Identifier", value="")
            annotation_output = st.text_input(
                "Annotation Output File",
                value="data/verified/streamlit_annotations.json",
            )
            allow_replace = st.checkbox("Replace this existing annotation output file")

            if st.button("Save Annotation for Pair"):
                if not selected_type:
                    st.error("Choose a dependency label before saving.")
                else:
                    try:
                        updated_pairs = save_ui_annotation(
                            pairs_to_annotate, pair_idx, selected_type, reviewer_id,
                            annotation_output, overwrite=allow_replace,
                        )
                        st.session_state.annotated_pairs = updated_pairs
                        if st.session_state.generated_pairs:
                            st.session_state.generated_pairs = updated_pairs
                        st.success(
                            f"Saved verified annotation for pair {curr_pair.id} to `{annotation_output}`."
                        )
                    except Exception as exc:
                        st.error(f"Could not save annotation: {exc}")

            # Agreement requires an independent second annotation for each pair.
            st.divider()
            st.subheader("Inter-Annotator Agreement (Cohen's Kappa)")
            st.info(
                "Agreement is not calculated here because this workflow stores one annotation per pair. "
                "A second independently collected label vector is required for Cohen's kappa."
            )
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
        if provider == "openai":
            authorize_openai = st.checkbox(
                "I authorize sending requirement text to the configured OpenAI service"
            )
        else:
            authorize_openai = False

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
            if not req_a_input.strip() or not req_b_input.strip():
                st.error("Both requirement texts must be non-empty.")
            elif provider == "openai" and not authorize_openai:
                st.warning("Confirm the external OpenAI request before sending requirement text.")
            else:
                try:
                    with st.spinner("Executing LEREDD Pipeline (Knowledge Retrieval + Inference)..."):
                        detector = LEREDDDetector(config=config)
                        req_a = Requirement(id="REQ-A", text=req_a_input, system="ADB")
                        req_b = Requirement(id="REQ-B", text=req_b_input, system="ADB")
                        pair = RequirementPair(id="PAIR-DEMO-001", req_a=req_a, req_b=req_b)

                        examples_pool = []
                        default_file = "data/annotated/adb_pairs.json"
                        if os.path.exists(default_file):
                            examples_pool = load_pairs_json(default_file)

                        srs_text = ""
                        srs_file = "data/raw/adb_srs.txt"
                        if use_rag and os.path.exists(srs_file):
                            with open(srs_file, "r", encoding="utf-8") as f:
                                srs_text = f.read()

                        pred = detector.detect_pair(
                            target_pair=pair, annotated_pool=examples_pool, srs_text=srs_text
                        )
                    st.session_state.detection_results.append(pred)
                    st.markdown("### Detection Result")
                    st.markdown(f"**Predicted Type:** {render_badge(pred.dependency_type.value)}", unsafe_allow_html=True)
                    st.markdown(f"**Likert Confidence Score:** `{pred.confidence}/5`")
                    st.markdown(f"**Generated Rationale:**\n\n_{pred.rationale}_")
                    with st.expander("View Full Raw LLM Output"):
                        st.code(pred.raw_response or "N/A")
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 4: MODEL EVALUATION & BASELINES
# ==========================================
elif page == "4. Model Evaluation":
    st.title("📊 Model Evaluation & Baseline Comparison")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("System-Disjoint Benchmark")
    st.caption(
        "Uses the repository's fixed ADB train / APA development / TJA test split. "
        "LEREDD uses MockLLM; this is a small demonstrative benchmark, not a paper replication."
    )

    if st.button("Run System-Disjoint Evaluation"):
        run_id = f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        report_path = Path("data/evaluation/ui_runs") / run_id / "system_disjoint_report.json"
        try:
            with st.spinner("Running the leakage-checked system-disjoint evaluation..."):
                report = run_evaluation(
                    split_dir="data/splits/system_disjoint",
                    output_path=str(report_path),
                    strict_contamination=True,
                    strict_gt=True,
                )
            st.success(f"Evaluation completed. Report saved to `{report_path}`.")
            st.markdown("### Comparative Results")
            summary = []
            for result in report["model_results"].values():
                summary.append({
                    "Model": result["model"],
                    "Accuracy": f"{result['overall_accuracy']:.4f}",
                    "Macro F1": f"{result['macro_f1']:.4f}",
                    "Weighted F1": f"{result['weighted_f1']:.4f}",
                })
            st.table(pd.DataFrame(summary))
            notices = report.get("evaluation_limitations", [])
            st.warning("Exploratory results only: " + " ".join(notices))
            hybrid_matrix = report["model_results"]["LEREDD_Hybrid"]["confusion_matrix"]
            if hybrid_matrix:
                st.markdown("### LEREDD Hybrid Confusion Matrix")
                cm_df = pd.DataFrame(hybrid_matrix)
                fig = px.imshow(
                    cm_df, labels=dict(x="Predicted Class", y="Ground Truth", color="Pairs"),
                    x=list(cm_df.columns), y=list(cm_df.index), text_auto=True,
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 5: DEPENDENCY GRAPH VISUALIZATION
# ==========================================
elif page == "5. Graph Visualization":
    st.title("🕸️ Interactive Requirement Dependency Network Graph")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("System Dependency Network")
    st.caption(
        "Displays relations from the legacy ADB annotation file. This graph is illustrative; "
        "the source records have no verification provenance and are not certified ground truth."
    )

    ds_file = "data/annotated/adb_pairs.json"
    if os.path.exists(ds_file):
        pairs = load_pairs_json(ds_file)

        # Build NetworkX Graph
        G = nx.DiGraph()
        for pair in pairs:
            dt = pair.dependency_type.value if pair.dependency_type else "Unlabeled"
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


# ==========================================
# PAGE 6: HUMAN REVIEW QUEUE WORKSPACE
# ==========================================
elif page == "6. Human Review Queue":
    st.title("🛡️ Multi-Method Disagreement & Uncertainty Review Queue")
    st.caption("Inspect provisional predictions, resolve heuristic/LLM conflicts, record human decisions, and produce verified annotations.")
    st.info("Note: Imported replication annotations (pending_review) preserved original Implemented by label; direction unresolved per Eq 1/2.")

    # Workspace File Selection Bar
    col_path, col_btn = st.columns([3, 1])
    with col_path:
        default_q_file = "data/queue/adb_review_queue.json"
        if not os.path.exists(default_q_file) and os.path.exists("review_queue.json"):
            default_q_file = "review_queue.json"
        queue_path = st.text_input("Active Review Queue JSON File", value=default_q_file)
    with col_btn:
        st.write("")
        st.write("")
        reload_btn = st.button("🔄 Reload Queue File")

    manager = ReviewQueueManager()

    if not os.path.exists(queue_path):
        st.warning(f"Review queue file '{queue_path}' not found. You can create one using `leredd-cli queue-create --input data/provisional/adb_provisional_hybrid.json` or `scripts/build_review_queue.py`.")
    else:
        try:
            items = manager.load_queue(queue_path)
            manifest = manager.create_manifest(items, source_identity=queue_path)

            # Section A: Operational Queue Overview
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Operational Queue Summary")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Items", manifest.total_items)
            m2.metric("Pending", manifest.pending_count)
            m3.metric("Accepted", manifest.accepted_count)
            m4.metric("Corrected", manifest.corrected_count)
            m5.metric("Deferred", manifest.deferred_count)
            m6.metric("Rejected", manifest.rejected_count)

            st.markdown("---")
            st.write("**Queue Reason Code Breakdown:**")
            if manifest.reasons_breakdown:
                rb_cols = st.columns(len(manifest.reasons_breakdown))
                for idx, (code_name, count) in enumerate(manifest.reasons_breakdown.items()):
                    with rb_cols[idx]:
                        st.info(f"**{code_name.upper()}**: {count}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Section B: Review Queue Browser & Filters
            st.subheader("🔍 Review Queue Browser & Filter Controls")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)

            with f_col1:
                status_opts = ["ALL"] + [s.value.upper() for s in QueueStatus]
                sel_status = st.selectbox("Filter Queue Status", status_opts, index=1)  # Default to PENDING (index 1)
            with f_col2:
                reason_opts = ["ALL"] + [r.value for r in ReviewReasonCode]
                sel_reason = st.selectbox("Filter Review Reason Code", reason_opts)
            with f_col3:
                prio_opts = ["ALL", "1 - CRITICAL", "2 - HIGH", "3 - MEDIUM", "4 - LOW", "5 - INFO"]
                sel_prio = st.selectbox("Filter Priority Rank", prio_opts)
            with f_col4:
                search_q = st.text_input("Search Pair / Requirement Text", value="")

            status_enum = QueueStatus(sel_status.lower()) if sel_status != "ALL" else None
            reason_enum = ReviewReasonCode(sel_reason.lower()) if sel_reason != "ALL" else None
            prio_val = int(sel_prio[0]) if sel_prio != "ALL" else None

            filtered_items = manager.filter_queue(items, status=status_enum, reason=reason_enum, priority=prio_val)
            if search_q:
                sq = search_q.lower()
                filtered_items = [
                    i for i in filtered_items
                    if sq in i.pair_id.lower() or sq in i.req_a_id.lower() or sq in i.req_b_id.lower() or sq in i.req_a_text.lower() or sq in i.req_b_text.lower()
                ]

            st.write(f"Showing **{len(filtered_items)}** of **{len(items)}** items matching filters.")

            if not filtered_items:
                st.info("No review queue items match the selected filter criteria.")
            else:
                item_labels = [
                    f"[{i.queue_id}] P{i.priority} | {i.pair_id} ({i.req_a_id} → {i.req_b_id}) | Reasons: {','.join(r.value for r in i.review_reasons)} | Status: {i.status.value.upper()}"
                    for i in filtered_items
                ]
                selected_idx = st.selectbox("Select Queue Item to Inspect & Review", range(len(item_labels)), format_func=lambda idx: item_labels[idx])
                selected_item = filtered_items[selected_idx]

                # Section C: Requirement Pair Detail View
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader(f"📋 Inspection Workspace: {selected_item.queue_id} (Pair: {selected_item.pair_id})")

                det_c1, det_c2, det_c3, det_c4 = st.columns(4)
                det_c1.write(f"**Priority Rank:** Priority {selected_item.priority}")
                det_c2.write(f"**Queue Status:** `{selected_item.status.value.upper()}`")
                det_c3.write(f"**Preserved Orientation:** `{selected_item.orientation[0]} → {selected_item.orientation[1]}`")
                det_c4.write(f"**Source System:** `{selected_item.system_name or 'GenericSystem'}`")

                st.warning(f"⚠️ **Review Reasons:** {', '.join(r.value.upper() for r in selected_item.review_reasons)} | **Machine Notes:** {selected_item.review_notes}")

                st.markdown("---")
                # Side-by-side Requirements Display
                req_c1, req_c2 = st.columns(2)
                with req_c1:
                    st.markdown(f"#### 📄 Requirement A (`{selected_item.req_a_id}`)")
                    st.info(selected_item.req_a_text)
                with req_c2:
                    st.markdown(f"#### 📄 Requirement B (`{selected_item.req_b_id}`)")
                    st.info(selected_item.req_b_text)

                st.markdown("---")
                # Model Comparison View
                st.subheader("🤖 Method Predictions & Evidence Comparison")
                pred_c1, pred_c2 = st.columns(2)

                with pred_c1:
                    st.markdown("##### 🔬 Heuristic Baseline Prediction")
                    h_val = selected_item.heuristic_prediction.value if selected_item.heuristic_prediction else "Abstained (None)"
                    st.markdown(render_badge(h_val), unsafe_allow_html=True)
                    st.caption(f"**Supporting Rule / Evidence:** {selected_item.heuristic_evidence or 'No heuristic rule triggered.'}")

                with pred_c2:
                    st.markdown("##### 🧠 LLM Inference Prediction")
                    l_val = selected_item.llm_prediction.value if selected_item.llm_prediction else "Abstained / Malformed"
                    st.markdown(render_badge(l_val), unsafe_allow_html=True)
                    conf_text = f"{selected_item.llm_confidence}/5 (Likert score - *uncalibrated probability*)" if selected_item.llm_confidence is not None else "N/A"
                    st.caption(f"**Self-Reported Confidence:** {conf_text}")
                    st.write(f"**Rationale:** {selected_item.llm_rationale or 'No rationale text recorded.'}")

                st.markdown("---")
                p_val = selected_item.proposed_label.value if selected_item.proposed_label else "None (Unresolved Conflict / Abstention)"
                st.write(f"**Proposed Provisional Label:** {render_badge(p_val)}", unsafe_allow_html=True)

                if selected_item.human_decision:
                    st.success(f"**Recorded Decision:** {selected_item.human_decision.decision_type.value.upper()} by `{selected_item.human_decision.reviewer_id}` on {selected_item.human_decision.decided_at}")
                    f_val = selected_item.human_decision.final_label.value if selected_item.human_decision.final_label else "None"
                    st.write(f"**Final Assigned Label:** {render_badge(f_val)}", unsafe_allow_html=True)
                    st.write(f"**Reviewer Justification Notes:** {selected_item.human_decision.reviewer_notes}")

                st.markdown('</div>', unsafe_allow_html=True)

                # Section D: Human Decision Controls
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("✍️ Record Human Reviewer Decision")

                ctrl_c1, ctrl_c2 = st.columns(2)
                with ctrl_c1:
                    decision_choice = st.radio(
                        "Decision Action",
                        ["ACCEPT (Accept Proposed Label)", "CORRECT (Assign Different Label)", "DEFER (Decline / Need Info)", "REJECT (Invalid Candidate Pair)"],
                        index=0 if selected_item.proposed_label else 1
                    )
                with ctrl_c2:
                    canonical_classes = [d.value for d in DependencyType]
                    default_c_idx = 0
                    if selected_item.proposed_label and selected_item.proposed_label.value in canonical_classes:
                        default_c_idx = canonical_classes.index(selected_item.proposed_label.value)
                    elif selected_item.llm_prediction and selected_item.llm_prediction.value in canonical_classes:
                        default_c_idx = canonical_classes.index(selected_item.llm_prediction.value)

                    target_label_str = st.selectbox("Target Dependency Class (for ACCEPT or CORRECT)", canonical_classes, index=default_c_idx)

                user_c1, user_c2 = st.columns(2)
                with user_c1:
                    reviewer_id_in = st.text_input("Reviewer Identifier", value=selected_item.human_decision.reviewer_id if selected_item.human_decision else "human_reviewer_1")
                with user_c2:
                    reviewer_notes_in = st.text_area("Reviewer Decision Justification & Notes", value=selected_item.human_decision.reviewer_notes if selected_item.human_decision else "")

                confirm_check = st.checkbox("I confirm this human decision promotes/updates the verified annotation state in the queue.")

                if st.button("💾 Submit & Persist Human Decision"):
                    if not confirm_check:
                        st.error("Submission blocked: Please check the confirmation box before saving.")
                    else:
                        d_type_map = {
                            "ACCEPT (Accept Proposed Label)": QueueStatus.ACCEPTED,
                            "CORRECT (Assign Different Label)": QueueStatus.CORRECTED,
                            "DEFER (Decline / Need Info)": QueueStatus.DEFERRED,
                            "REJECT (Invalid Candidate Pair)": QueueStatus.REJECTED,
                        }
                        target_d_status = d_type_map[decision_choice]

                        try:
                            final_l = target_label_str if target_d_status in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) else None
                            updated_item, verified_pair = manager.record_decision(
                                item=selected_item,
                                decision_type=target_d_status,
                                final_label=final_l,
                                reviewer_id=reviewer_id_in,
                                reviewer_notes=reviewer_notes_in
                            )

                            manifest_updated = manager.create_manifest(items, source_identity=queue_path)
                            manager.save_queue(items, manifest_updated, queue_path)

                            st.success(f"Successfully saved decision '{target_d_status.value.upper()}' for item {selected_item.queue_id}!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Failed to record decision: {e}")

                st.markdown('</div>', unsafe_allow_html=True)

            # Section E: Verified Annotation Export
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📤 Export Verified Annotations")
            export_path = st.text_input("Verified Dataset Output File Path", value="data/verified/adb_verified.json")
            overwrite_export = st.checkbox("Replace this existing verified export file")

            verified_count = sum(1 for i in items if i.status in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) and i.human_decision and i.human_decision.final_label is not None)
            st.write(f"Ready to export **{verified_count}** human-verified annotations (provisional, deferred, rejected, and unresolved items are excluded).")

            if st.button("🚀 Export Verified Dataset"):
                if verified_count == 0:
                    st.warning("No human-verified items available in queue to export. Record ACCEPTED or CORRECTED decisions first.")
                else:
                    try:
                        verified_pairs = []
                        for i in items:
                            if i.status in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) and i.human_decision and i.human_decision.final_label:
                                _, v_pair = manager.record_decision(i, i.status, final_label=i.human_decision.final_label, reviewer_id=i.human_decision.reviewer_id, reviewer_notes=i.human_decision.reviewer_notes)
                                verified_pairs.append(v_pair)

                        safe_export_path = validate_ui_output_path(
                            export_path, overwrite=overwrite_export
                        )
                        safe_export_path.parent.mkdir(parents=True, exist_ok=True)
                        save_pairs_json(verified_pairs, str(safe_export_path))
                        st.success(f"Successfully exported {len(verified_pairs)} verified requirement pairs to '{safe_export_path}'.")
                    except Exception as e:
                        st.error(f"Error exporting verified dataset: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to load or parse review queue file '{queue_path}': {e}")
