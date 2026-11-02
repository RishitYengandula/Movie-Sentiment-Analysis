"""
Streamlit Web Application for Movie Review Sentiment Analysis.
Production-grade ML demonstration dashboard serving real-time inference
using the pre-trained, serialized scikit-learn Pipeline (TF-IDF + Logistic Regression).
Operates strictly in read-only inference mode with zero model retraining.
"""

import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path so 'src' modules resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import REPORTS_DIR
from src.predict import load_model, predict_sentiment

# Pre-defined sample reviews for rapid demonstration
SAMPLE_POSITIVE = (
    "I absolutely loved this movie. The acting was fantastic and the story was engaging."
)
SAMPLE_NEGATIVE = (
    "This was a terrible movie. The plot was boring and the acting was awful."
)
SAMPLE_AMBIGUOUS = "The movie was okay, but nothing particularly special."


@st.cache_resource(show_spinner="Loading trained sentiment pipeline...")
def get_model():
    """
    Loads and caches the serialized scikit-learn Pipeline.
    Cached across user sessions to ensure instantaneous inference.
    """
    return load_model()


def set_sample_review(text: str) -> None:
    """Callback to populate the text area with a selected sample review."""
    st.session_state["review_input"] = text


def inject_custom_css() -> None:
    """Injects carefully scoped CSS for a cohesive Dark Navy / Blue-Slate ML dashboard."""
    st.markdown(
        """
        <style>
            /* 1. Global Page Background & Base Text */
            .stApp {
                background-color: #0B1220 !important;
                color: #CBD5E1 !important;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 2.5rem !important;
                max-width: 1040px !important;
            }

            /* 2. Sidebar Styling */
            section[data-testid="stSidebar"] {
                background-color: #111C2E !important;
                border-right: 1px solid #263A56 !important;
            }
            section[data-testid="stSidebar"] h3 {
                color: #F1F5F9 !important;
                font-weight: 800 !important;
            }
            section[data-testid="stSidebar"] .sidebar-section {
                background: #16243A;
                border: 1px solid #263A56;
                border-radius: 9px;
                padding: 0.8rem 0.9rem;
                margin-bottom: 0.85rem;
            }
            section[data-testid="stSidebar"] .sidebar-section-title {
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                color: #94A3B8;
                margin-bottom: 0.5rem;
                border-bottom: 1px solid #263A56;
                padding-bottom: 0.3rem;
            }
            section[data-testid="stSidebar"] .sidebar-item {
                font-size: 0.8rem;
                color: #CBD5E1;
                margin: 0.28rem 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            section[data-testid="stSidebar"] .sidebar-item-strong {
                font-weight: 700;
                color: #F1F5F9;
                text-align: right;
            }

            /* 3. Hero Header */
            .hero-container {
                background: #111C2E;
                border: 1px solid #263A56;
                border-radius: 14px;
                padding: 1.4rem 1.8rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            }
            .hero-badge {
                display: inline-block;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #93C5FD;
                background: #172554;
                border: 1px solid #1D4ED8;
                padding: 0.24rem 0.65rem;
                border-radius: 20px;
                margin-bottom: 0.65rem;
            }
            .hero-title {
                font-size: 2rem;
                font-weight: 800;
                color: #F1F5F9;
                margin: 0 0 0.4rem 0;
                letter-spacing: -0.02em;
                line-height: 1.2;
            }
            .hero-subtitle {
                font-size: 1rem;
                color: #CBD5E1;
                margin: 0;
                line-height: 1.5;
            }

            /* 4. Top Performance Cards */
            .metric-card {
                background: #16243A;
                border: 1px solid #263A56;
                border-radius: 10px;
                padding: 0.9rem 1.1rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                transition: border-color 0.15s ease;
            }
            .metric-card:hover {
                border-color: #3B82F6;
            }
            .metric-label {
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #94A3B8;
                margin-bottom: 0.25rem;
            }
            .metric-value {
                font-size: 1.6rem;
                font-weight: 800;
                color: #F1F5F9;
                line-height: 1.1;
                margin-bottom: 0.2rem;
            }
            .metric-sub {
                font-size: 0.72rem;
                color: #94A3B8;
            }

            /* 5. Section Titles & Descriptions */
            .section-header {
                font-size: 1.15rem;
                font-weight: 700;
                color: #F1F5F9;
                margin-top: 0.6rem;
                margin-bottom: 0.2rem;
            }
            .section-desc {
                font-size: 0.88rem;
                color: #94A3B8;
                margin-bottom: 0.75rem;
            }

            /* 6. Example Prompt Cards */
            .sample-box {
                background: #16243A;
                border: 1px solid #263A56;
                border-radius: 10px;
                padding: 0.85rem 0.95rem;
                min-height: 125px;
                margin-bottom: 0.45rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
            }
            .sample-tag {
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                display: inline-block;
                padding: 0.18rem 0.5rem;
                border-radius: 4px;
                width: fit-content;
                margin-bottom: 0.4rem;
            }
            .tag-pos {
                background: #064E3B;
                color: #86EFAC;
                border: 1px solid #059669;
            }
            .tag-neg {
                background: #450A0A;
                color: #FCA5A5;
                border: 1px solid #DC2626;
            }
            .tag-amb {
                background: #1E293B;
                color: #CBD5E1;
                border: 1px solid #475569;
            }
            .sample-snippet {
                font-size: 0.82rem;
                color: #CBD5E1;
                line-height: 1.4;
                font-style: italic;
                margin: 0 0 0.35rem 0;
            }
            .sample-subdesc {
                font-size: 0.72rem;
                color: #94A3B8;
                margin: 0;
            }

            /* 7. Text Input Area */
            div[data-testid="stTextArea"] textarea {
                background-color: #111C2E !important;
                color: #F1F5F9 !important;
                border: 1.5px solid #263A56 !important;
                border-radius: 9px !important;
                font-size: 0.92rem !important;
            }
            div[data-testid="stTextArea"] textarea:focus {
                border-color: #3B82F6 !important;
                box-shadow: 0 0 0 1px #3B82F6 !important;
            }
            div[data-testid="stTextArea"] textarea::placeholder {
                color: #94A3B8 !important;
            }

            /* 8. Action Buttons */
            /* Primary Analyze Button */
            div[data-testid="stButton"] > button[kind="primary"] {
                background-color: #3B82F6 !important;
                color: #FFFFFF !important;
                border: none !important;
                font-weight: 700 !important;
                border-radius: 8px !important;
                padding: 0.55rem 1.3rem !important;
                font-size: 0.95rem !important;
                box-shadow: 0 2px 10px rgba(59, 130, 246, 0.3) !important;
                transition: all 0.18s ease-in-out !important;
            }
            div[data-testid="stButton"] > button[kind="primary"]:hover {
                background-color: #60A5FA !important;
                box-shadow: 0 4px 16px rgba(59, 130, 246, 0.45) !important;
                transform: translateY(-1px);
            }

            /* Secondary Load Buttons */
            div[data-testid="stButton"] > button[kind="secondary"] {
                background-color: #16243A !important;
                color: #CBD5E1 !important;
                border: 1px solid #263A56 !important;
                font-size: 0.82rem !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
                padding: 0.38rem 0.6rem !important;
                transition: all 0.15s ease !important;
            }
            div[data-testid="stButton"] > button[kind="secondary"]:hover {
                background-color: #1E324F !important;
                border-color: #3B82F6 !important;
                color: #F1F5F9 !important;
            }

            /* 9. Centerpiece Prediction Result Card */
            .result-card {
                border-radius: 12px;
                padding: 1.35rem 1.65rem;
                margin-top: 1.25rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            }
            .positive-card {
                background: linear-gradient(180deg, #07261a 0%, #0c1a24 100%);
                border: 1.5px solid #22C55E;
            }
            .negative-card {
                background: linear-gradient(180deg, #2b0e14 0%, #0c1a24 100%);
                border: 1.5px solid #F87171;
            }
            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.95rem;
            }
            .result-badge {
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                padding: 0.22rem 0.6rem;
                border-radius: 6px;
            }
            .pos-badge {
                background: #064E3B;
                color: #86EFAC;
                border: 1px solid #059669;
            }
            .neg-badge {
                background: #450A0A;
                color: #FCA5A5;
                border: 1px solid #DC2626;
            }
            .result-meta {
                font-size: 0.75rem;
                color: #94A3B8;
            }
            .result-body {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                margin-bottom: 0.95rem;
            }
            .result-sentiment-title {
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: #94A3B8;
                margin-bottom: 0.15rem;
            }
            .result-sentiment-text {
                font-size: 2.2rem;
                font-weight: 900;
                letter-spacing: -0.01em;
                line-height: 1;
            }
            .result-confidence-val {
                font-size: 2.2rem;
                font-weight: 900;
                line-height: 1;
                text-align: right;
            }
            .result-confidence-sub {
                font-size: 0.75rem;
                font-weight: 600;
                color: #94A3B8;
                text-align: right;
                margin-top: 0.2rem;
            }
            .pos-text {
                color: #22C55E;
            }
            .neg-text {
                color: #F87171;
            }
            .result-meter-track {
                height: 10px;
                background: #16243A;
                border: 1px solid #263A56;
                border-radius: 5px;
                overflow: hidden;
                margin-bottom: 0.55rem;
            }
            .result-meter-fill {
                height: 100%;
                border-radius: 5px;
            }
            .pos-fill {
                background: linear-gradient(90deg, #16A34A 0%, #22C55E 100%);
            }
            .neg-fill {
                background: linear-gradient(90deg, #DC2626 0%, #F87171 100%);
            }
            .result-footer {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: #94A3B8;
            }

            /* 10. Footer */
            .app-footer {
                text-align: center;
                font-size: 0.76rem;
                color: #64748B;
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid #1E293B;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Initialize review text state if not present
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""

    # -------------------------------------------------------------
    # Sidebar: Project Information Panel
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📊 Project Information")
        st.markdown(
            '<div style="font-size:0.8rem; color:#94A3B8; margin-bottom:0.8rem;">Architecture, dataset & evaluation metrics</div>',
            unsafe_allow_html=True,
        )

        # Section 1: MODEL
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-section-title">Model Architecture</div>
                <div class="sidebar-item"><span>Feature Extraction</span><span class="sidebar-item-strong">TF-IDF</span></div>
                <div class="sidebar-item"><span>N-Gram Range</span><span class="sidebar-item-strong">Unigrams + Bigrams (1, 2)</span></div>
                <div class="sidebar-item"><span>Vocabulary Cap</span><span class="sidebar-item-strong">50,000 features</span></div>
                <div class="sidebar-item"><span>Sublinear TF</span><span class="sidebar-item-strong">True (1 + log(tf))</span></div>
                <div class="sidebar-item"><span>Classifier</span><span class="sidebar-item-strong">Logistic Regression</span></div>
                <div class="sidebar-item"><span>Solver</span><span class="sidebar-item-strong">L-BFGS (L2, C=1.0)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Section 2: DATASET
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-section-title">Dataset Specs</div>
                <div class="sidebar-item"><span>Source</span><span class="sidebar-item-strong">IMDb Large Movie Reviews</span></div>
                <div class="sidebar-item"><span>Raw Total</span><span class="sidebar-item-strong">50,000 reviews</span></div>
                <div class="sidebar-item"><span>Deduplicated</span><span class="sidebar-item-strong">49,582 reviews (-418)</span></div>
                <div class="sidebar-item"><span>Split Ratio</span><span class="sidebar-item-strong">80/20 Stratified</span></div>
                <div class="sidebar-item"><span>Training Set</span><span class="sidebar-item-strong">39,665 reviews</span></div>
                <div class="sidebar-item"><span>Test Set</span><span class="sidebar-item-strong">9,917 reviews</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Section 3: PERFORMANCE
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-section-title">Evaluation Metrics</div>
                <div class="sidebar-item"><span>5-Fold CV Accuracy</span><span class="sidebar-item-strong">90.55% ± 0.19%</span></div>
                <div class="sidebar-item"><span>Test Accuracy</span><span class="sidebar-item-strong">90.69%</span></div>
                <div class="sidebar-item"><span>Precision (Positive)</span><span class="sidebar-item-strong">89.56%</span></div>
                <div class="sidebar-item"><span>Recall (Positive)</span><span class="sidebar-item-strong">92.20%</span></div>
                <div class="sidebar-item"><span>F1-Score</span><span class="sidebar-item-strong">90.86%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Section 4: CONFUSION MATRIX
        cm_image_path = REPORTS_DIR / "confusion_matrix.png"
        if cm_image_path.exists():
            st.markdown(
                """
                <div class="sidebar-section">
                    <div class="sidebar-section-title">Model Evaluation</div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:0.4rem;">
                        Confusion matrix from the held-out test set (9,917 reviews).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Open via PIL to ensure unique hash cache busting
            cm_img = Image.open(cm_image_path)
            st.image(
                cm_img,
                use_column_width=True,
            )

        st.markdown(
            '<div style="font-size:0.75rem; color:#94A3B8; margin-top:0.5rem;">🔒 <strong>Pipeline Safety</strong>: Read-only inference. Zero retraining.</div>',
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Hero / Header Section
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">IMDb 50K • NLP • Machine Learning</div>
            <h1 class="hero-title">🎬 Movie Review Sentiment Analysis</h1>
            <p class="hero-subtitle">Analyze movie reviews using TF-IDF and Logistic Regression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Model Performance Cards (Top Row)
    # -------------------------------------------------------------
    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">90.69%</div>
                <div class="metric-sub">Evaluated on 9,917 held-out reviews</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with met_col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">90.86%</div>
                <div class="metric-sub">Harmonic mean of precision & recall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with met_col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">5-Fold CV Accuracy</div>
                <div class="metric-value">90.55%</div>
                <div class="metric-sub">Std: ± 0.19% across training folds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load Model Artifact
    try:
        model = get_model()
    except FileNotFoundError as fnf_err:
        st.error(f"⚠️ **Model Artifact Missing**: {fnf_err}")
        st.info("Run `python -m src.train` in your terminal to train and serialize the model first.")
        return
    except Exception as exc:
        st.error(f"⚠️ **Failed to load model**: {exc}")
        return

    # -------------------------------------------------------------
    # Example Reviews Section
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">💡 Try an Example Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Select a sample review below to populate the text area and inspect model predictions:</div>',
        unsafe_allow_html=True,
    )

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    with sample_col1:
        st.markdown(
            """
            <div class="sample-box">
                <div>
                    <span class="sample-tag tag-pos">👍 Positive</span>
                    <p class="sample-snippet">"I absolutely loved this movie. The acting was fantastic and the story was engaging."</p>
                </div>
                <p class="sample-subdesc">Strongly positive sentiment cues</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Load Positive",
            key="btn_sample_pos",
            on_click=set_sample_review,
            args=(SAMPLE_POSITIVE,),
            use_container_width=True,
        )
    with sample_col2:
        st.markdown(
            """
            <div class="sample-box">
                <div>
                    <span class="sample-tag tag-neg">👎 Negative</span>
                    <p class="sample-snippet">"This was a terrible movie. The plot was boring and the acting was awful."</p>
                </div>
                <p class="sample-subdesc">Unambiguously critical review</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Load Negative",
            key="btn_sample_neg",
            on_click=set_sample_review,
            args=(SAMPLE_NEGATIVE,),
            use_container_width=True,
        )
    with sample_col3:
        st.markdown(
            """
            <div class="sample-box">
                <div>
                    <span class="sample-tag tag-amb">⚖️ Ambiguous</span>
                    <p class="sample-snippet">"The movie was okay, but nothing particularly special."</p>
                </div>
                <p class="sample-subdesc">Contains both positive and negative cues</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Load Ambiguous",
            key="btn_sample_amb",
            on_click=set_sample_review,
            args=(SAMPLE_AMBIGUOUS,),
            use_container_width=True,
        )

    st.write("")

    # -------------------------------------------------------------
    # Review Input Section
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">✍️ Analyze a Movie Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Enter a review below and let the trained model predict its sentiment:</div>',
        unsafe_allow_html=True,
    )

    review_text = st.text_area(
        label="Movie Review Input",
        key="review_input",
        height=145,
        placeholder="Enter a movie review here...",
        label_visibility="collapsed",
    )

    action_col1, action_col2 = st.columns([1, 3])
    with action_col1:
        analyze_clicked = st.button(
            "⚡ Analyze Sentiment",
            type="primary",
            use_container_width=True,
        )

    # -------------------------------------------------------------
    # Prediction Result Section (Centerpiece)
    # -------------------------------------------------------------
    if analyze_clicked:
        if not review_text or not review_text.strip():
            st.warning("⚠️ Please enter a movie review or select an example above before analyzing.")
        else:
            try:
                with st.spinner("Classifying review..."):
                    result = predict_sentiment(review_text, model=model)

                sentiment = result["sentiment"].upper()
                confidence = result["confidence"]
                confidence_pct = confidence * 100.0

                if result["sentiment"] == "positive":
                    st.markdown(
                        f"""
                        <div class="result-card positive-card">
                            <div class="result-header">
                                <span class="result-badge pos-badge">🎯 Sentiment Result</span>
                                <span class="result-meta">Pipeline: TF-IDF + Logistic Regression</span>
                            </div>
                            <div class="result-body">
                                <div>
                                    <div class="result-sentiment-title">Predicted Sentiment</div>
                                    <div class="result-sentiment-text pos-text">{sentiment}</div>
                                </div>
                                <div>
                                    <div class="result-confidence-val pos-text">{confidence_pct:.2f}%</div>
                                    <div class="result-confidence-sub">Model Confidence</div>
                                </div>
                            </div>
                            <div class="result-meter-track">
                                <div class="result-meter-fill pos-fill" style="width: {confidence_pct:.2f}%;"></div>
                            </div>
                            <div class="result-footer">
                                <span>Positive probability: <strong style="color:#86EFAC;">{confidence_pct:.2f}%</strong></span>
                                <span>Decision threshold: <strong style="color:#F1F5F9;">0.50</strong></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card negative-card">
                            <div class="result-header">
                                <span class="result-badge neg-badge">🎯 Sentiment Result</span>
                                <span class="result-meta">Pipeline: TF-IDF + Logistic Regression</span>
                            </div>
                            <div class="result-body">
                                <div>
                                    <div class="result-sentiment-title">Predicted Sentiment</div>
                                    <div class="result-sentiment-text neg-text">{sentiment}</div>
                                </div>
                                <div>
                                    <div class="result-confidence-val neg-text">{confidence_pct:.2f}%</div>
                                    <div class="result-confidence-sub">Model Confidence</div>
                                </div>
                            </div>
                            <div class="result-meter-track">
                                <div class="result-meter-fill neg-fill" style="width: {confidence_pct:.2f}%;"></div>
                            </div>
                            <div class="result-footer">
                                <span>Negative probability: <strong style="color:#FCA5A5;">{confidence_pct:.2f}%</strong></span>
                                <span>Decision threshold: <strong style="color:#F1F5F9;">0.50</strong></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except ValueError as val_err:
                st.warning(f"⚠️ Input Validation Warning: {val_err}")
            except Exception as exc:
                st.error(f"⚠️ An unexpected error occurred during prediction: {exc}")

    # -------------------------------------------------------------
    # Subtle Footer
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="app-footer">
            Built with Python • scikit-learn • Streamlit &nbsp;|&nbsp; IMDb 50K Movie Reviews
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
