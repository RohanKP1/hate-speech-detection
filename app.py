import streamlit as st
import os
from agents.hate_speech_agent import HateSpeechDetectionAgent
from agents.retriever_agent import HybridRetrieverAgent
from agents.reasoning_agent import PolicyReasoningAgent
from agents.action_agent import ActionRecommenderAgent
from agents.error_handler import ErrorHandlerAgent
from agents.audio_agent import AudioTranscriptionAgent
import pandas as pd
from datetime import datetime
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Page configuration with responsive layout
st.set_page_config(
    page_title="Content Moderator AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better responsiveness and modern design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    .hate { background-color: #dc3545; color: white; }
    .toxic { background-color: #fd7e14; color: white; }
    .offensive { background-color: #ffc107; color: black; }
    .neutral { background-color: #28a745; color: white; }
    .ambiguous { background-color: #6c757d; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'results_history': [],
        'agents_initialized': False,
        'current_analysis': None,
        'show_advanced': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def initialize_agents():
    """Cache agents to avoid re-initialization"""
    try:
        policy_docs_path = "data/policy_docs"
        if not os.path.exists(policy_docs_path):
            st.error(f"Policy documents directory not found: {policy_docs_path}")
            st.stop()
        
        agents = {
            'hate_speech': HateSpeechDetectionAgent(),
            'retriever': HybridRetrieverAgent(policy_docs_path),
            'reasoning': PolicyReasoningAgent(),
            'action': ActionRecommenderAgent(),
            'error_handler': ErrorHandlerAgent(),
            'audio': AudioTranscriptionAgent()
        }
        return agents
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        st.stop()

def get_classification_badge(classification, confidence):
    """Generate HTML badge for classification"""
    badge_class = classification.lower()
    return f'<span class="status-badge {badge_class}">{classification} ({confidence})</span>'

def create_metric_card(title, value, color="#667eea"):
    """Create a styled metric card"""
    return f"""
    <div class="metric-container" style="background: {color};">
        <h3 style="margin: 0; font-size: 2em;">{value}</h3>
        <p style="margin: 0; opacity: 0.8;">{title}</p>
    </div>
    """

async def analyze_content_async(agents, user_input):
    """Async content analysis for better performance"""
    with ThreadPoolExecutor() as executor:
        # Run classification and retrieval in parallel
        classification_future = executor.submit(agents['hate_speech'].classify_text, user_input)
        
        classification = classification_future.result()
        
        # Run remaining tasks in parallel
        retrieval_future = executor.submit(
            agents['retriever'].retrieve_relevant_policies, 
            user_input, 
            classification['classification']
        )
        
        retrieved = retrieval_future.result()
        
        explanation_future = executor.submit(
            agents['reasoning'].generate_explanation, 
            user_input, 
            classification, 
            retrieved
        )
        recommendation_future = executor.submit(
            agents['action'].recommend_action, 
            classification
        )
        
        explanation = explanation_future.result()
        recommendation = recommendation_future.result()
    
    return classification, retrieved, explanation, recommendation

def main():
    init_session_state()
    
    st.title("🛡️ Content Moderator AI")
    st.caption("AI-powered content analysis and moderation recommendations")

    # Initialize agents with progress
    if not st.session_state.agents_initialized:
        with st.spinner("Loading AI models..."):
            agents = initialize_agents()
            st.session_state.agents = agents
            st.session_state.agents_initialized = True
        st.success("Ready to analyze content!")
    else:
        agents = st.session_state.agents

    # Advanced settings sidebar (collapsible)
    with st.sidebar:
        st.header("Advanced Settings")
            
        # Policy documents status
        if os.path.exists("data/policy_docs"):
            policy_files = [f for f in os.listdir("data/policy_docs") if f.endswith('.txt')]
            st.success(f"{len(policy_files)} policies loaded")
                
            with st.expander("View Loaded Policies"):
                for file in policy_files:
                    st.text(f"• {file.replace('.txt', '').replace('_', ' ').title()}")
        else:
            st.error("Policy documents not found")
            
        st.divider()
            
        # Controls
        if st.button("Clear History", type="secondary"):
            st.session_state.results_history = []
            st.rerun()
        
    # Main content with responsive tabs
    tab1, tab2, tab3 = st.tabs(["Text", "Audio", "History"])

    # Tab 1: Streamlined Text Analysis
    with tab1:
        # Input section
        user_input = st.text_area(
            "**Enter content to analyze:**",
            height=120,
            placeholder="Paste user-generated content here for real-time analysis...",
            key="content_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            analyze_btn = st.button("Analyze Content", type="primary", use_container_width=True)
        with col2:
            quick_mode = st.checkbox("Quick Mode", help="Faster analysis with basic results")
        with col3:
            auto_analyze = st.checkbox("Auto-analyze", help="Analyze as you type")

        # Auto-analyze functionality
        if auto_analyze and user_input and len(user_input) > 10:
            analyze_btn = True

        # Analysis section
        if analyze_btn and user_input:
            # Input validation
            validation = agents['error_handler'].validate_input(user_input)
            if not validation['valid']:
                st.error(f"{validation['message']}")
                return

            # Progress tracking
            progress_container = st.container()
            results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            try:
                # Step 1: Classification
                status_text.text("Classifying content...")
                progress_bar.progress(25)
                classification = agents['hate_speech'].classify_text(user_input)
                
                # Step 2: Policy Retrieval (skip in quick mode)
                if not quick_mode:
                    status_text.text("Retrieving relevant policies...")
                    progress_bar.progress(50)
                    retrieved = agents['retriever'].retrieve_relevant_policies(
                        user_input, classification['classification']
                    )
                else:
                    retrieved = []
                
                # Step 3: Explanation
                status_text.text("Generating explanation...")
                progress_bar.progress(75)
                explanation = agents['reasoning'].generate_explanation(
                    user_input, classification, retrieved
                )
                
                # Step 4: Recommendation
                status_text.text("Recommending action...")
                progress_bar.progress(100)
                recommendation = agents['action'].recommend_action(classification)
                
                # Clear progress
                progress_container.empty()
                
                # Display results in cards
                with results_container:
                    # Main results row
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown("### 🎯 Classification")
                        badge_html = get_classification_badge(
                            classification['classification'], 
                            classification['confidence']
                        )
                        st.markdown(badge_html, unsafe_allow_html=True)
                        
                        with st.expander("📋 Details"):
                            st.write(f"**Reasoning:** {classification['reason']}")
                    
                    with col2:
                        st.markdown("### ⚡ Recommended Action")
                        st.info(f"**{recommendation['action']}**")
                        st.caption(f"Severity: {recommendation['severity']}")
                        
                        with st.expander("🤔 Why this action?"):
                            st.write(recommendation['reasoning'])
                    
                    with col3:
                        st.markdown("### 📊 Summary")
                        risk_level = "High" if classification['classification'] in ['Hate', 'Toxic'] else "Low"
                        st.metric("Risk Level", risk_level)
                        st.metric("Confidence", classification['confidence'])
                
                    # Additional information (expandable)
                    if not quick_mode:
                        st.markdown("---")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("### 📚 Policy References")
                            if retrieved:
                                for i, policy in enumerate(retrieved[:5]):  # Show top 5
                                    with st.expander(f"📄 {policy['source']} (Score: {policy['relevance_score']:.2f})"):
                                        st.write(policy['content'][:200] + "...")
                            else:
                                st.info("No specific policies matched this content.")
                        
                        with col2:
                            st.markdown("### 🔍 Detailed Analysis")
                            st.write(explanation)

                # Store results
                st.session_state.results_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'classification': classification['classification'],
                    'confidence': classification['confidence'],
                    'action': recommendation['action'],
                    'severity': recommendation['severity']
                })

            except Exception as e:
                progress_container.empty()
                error_info = agents['error_handler'].handle_error(e, "text analysis")
                st.error(f"{error_info['type']}: {error_info['message']}")
                st.info(f"{error_info['suggestion']}")

    # Tab 2: Streamlined Audio Analysis
    with tab2:
        st.markdown("### Audio Content Analysis")
        
        # Audio upload
        col1, col2 = st.columns([7, 1])
        with col1:
            audio_file = st.file_uploader(
                "Upload audio file", 
                type=["wav", "mp3", "flac", "m4a"],
                help="Supported formats: WAV, MP3, FLAC, M4A"
            )
        with col2:
            st.markdown("Record audio")
            record_btn = st.button("🎙️", type="secondary", use_container_width=True)

        # Process uploaded audio
        if audio_file:
            with st.spinner("Transcribing audio..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                        tmp.write(audio_file.read())
                        transcription = agents['audio'].transcribe_audio_file(tmp.name)
                        os.unlink(tmp.name)
                    
                    # Display transcription
                    st.success("Transcription complete!")
                    transcribed_text = st.text_area("Transcribed Text", transcription, height=150)
                    
                    # Option to analyze transcribed text
                    if st.button("Analyze Transcription", type="primary"):
                        st.session_state.content_input = transcribed_text
                        st.switch_page("analyze")  # Would switch to analyze tab
                        
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")

        # Live recording
        if record_btn:
            with st.spinner("Recording and transcribing..."):
                try:
                    rt_transcription = agents['audio'].transcribe_real_time_audio()
                    st.text_area("Live Transcription", rt_transcription, height=150)
                except Exception as e:
                    st.error(f"Live transcription failed: {str(e)}")

    # Tab 3: Enhanced History with Analytics
    with tab3:
        st.markdown("### Analysis History & Insights")
        
        if not st.session_state.results_history:
            st.info("No analysis history yet. Start analyzing content to see results here.")
        else:
            df = pd.DataFrame(st.session_state.results_history)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_metric_card("Total Analyses", len(df), "#667eea"), unsafe_allow_html=True)
            with col2:
                hate_count = (df['classification'] == "Hate").sum()
                st.markdown(create_metric_card("Hate Detected", hate_count, "#dc3545"), unsafe_allow_html=True)
            with col3:
                toxic_count = (df['classification'] == "Toxic").sum()
                st.markdown(create_metric_card("Toxic Content", toxic_count, "#fd7e14"), unsafe_allow_html=True)
            with col4:
                neutral_count = (df['classification'] == "Neutral").sum()
                st.markdown(create_metric_card("Clean Content", neutral_count, "#28a745"), unsafe_allow_html=True)
            
            # Data table with better formatting
            st.markdown("---")
            
            # Filters
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                filter_classification = st.selectbox(
                    "Filter by Classification", 
                    ["All"] + list(df['classification'].unique())
                )
            with col2:
                filter_action = st.selectbox(
                    "Filter by Action", 
                    ["All"] + list(df['action'].unique())
                )
            
            # Apply filters
            filtered_df = df.copy()
            if filter_classification != "All":
                filtered_df = filtered_df[filtered_df['classification'] == filter_classification]
            if filter_action != "All":
                filtered_df = filtered_df[filtered_df['action'] == filter_action]
            
            # Display filtered data
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, HH:mm"),
                    "text": st.column_config.TextColumn("Content", width="large"),
                    "classification": st.column_config.TextColumn("Class", width="small"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                }
            )
            
            # Export options
            col1, col2 = st.columns([1, 4])
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Export CSV",
                    csv,
                    f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()