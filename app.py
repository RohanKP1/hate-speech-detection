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
import time

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
    .nav-button {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.5rem;
        background: #f0f2f6;
        color: #262730;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-button:hover {
        background: #e6e9ef;
    }
    .nav-button.active {
        background: #ff4b4b;
        color: white;
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
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'results_history': [],
        'agents_initialized': False,
        'current_analysis': None,
        'show_advanced': False,
        'current_page': 'Text Analysis',  # Default page
        'pending_analysis_text': '',  # For transferring text between pages
        'auto_analyze_pending': False,  # Flag for auto-analysis
        'transcription_result': '',  # Store transcription results
        'live_transcription_result': '',  # Store live transcription results
        'last_auto_analyzed_text': '',  # Track last auto-analyzed text
        'auto_analyze_debounce': 0,  # Debounce timer for auto-analysis
        'quick_mode': False,  # Quick mode setting
        'auto_analyze': False  # Auto-analyze setting
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

def render_sidebar_navigation():
    """Render sidebar navigation and settings"""
    with st.sidebar:
        st.header("🛡️ Content Moderator AI")
        st.caption("AI-powered content analysis")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        pages = {
            "Text Analysis": "Text Analysis",
            "Audio Analysis": "Audio Analysis",
            "History & Analytics": "History"
        }
        
        for display_name, page_key in pages.items():
            if st.button(display_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.subheader("System Status")
        
        if st.session_state.agents_initialized:
            st.success("✅ AI Models Ready")
        else:
            st.warning("⏳ Loading Models...")
            
        # Policy documents status
        if os.path.exists("data/policy_docs"):
            policy_files = [f for f in os.listdir("data/policy_docs") if f.endswith('.txt')]
            st.success(f"✅ {len(policy_files)} Policies Loaded")
                
            with st.expander("View Loaded Policies"):
                for file in policy_files:
                    st.text(f"• {file.replace('.txt', '').replace('_', ' ').title()}")
        else:
            st.error("❌ Policy Documents Missing")
        
        st.markdown("---")
            
        # Settings & Controls
        st.subheader("Settings")
        
        # Quick settings - Store in session state
        st.session_state.quick_mode = st.checkbox("Quick Mode", 
            value=st.session_state.quick_mode,
            help="Faster analysis with basic results")
        
        st.session_state.auto_analyze = st.checkbox("Auto-analyze", 
            value=st.session_state.auto_analyze,
            help="Analyze as you type (with 2 second delay)")
        
        # Show auto-analyze status
        if st.session_state.auto_analyze:
            st.info("🤖 Auto-analyze is ON")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1, key="confidence_threshold")
            st.selectbox("Analysis Mode", ["Standard", "Strict", "Lenient"], key="analysis_mode")
            st.checkbox("Enable Detailed Logging", key="detailed_logging")
        
        st.markdown("---")
        
        # Action buttons
        st.subheader("Actions")
        
        if st.button("Refresh Models", type="secondary", use_container_width=True):
            st.session_state.agents_initialized = False
            st.rerun()
            
        if st.button("Clear History", type="primary", use_container_width=True):
            st.session_state.results_history = []
            st.success("History cleared!")
            st.rerun()

def perform_analysis(agents, user_input):
    """Centralized analysis function"""
    # Input validation
    validation = agents['error_handler'].validate_input(user_input)
    if not validation['valid']:
        st.error(f"❌ {validation['message']}")
        return None

    # Progress tracking
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Step 1: Classification
        status_text.text("🔍 Classifying content...")
        progress_bar.progress(25)
        classification = agents['hate_speech'].classify_text(user_input)
        
        # Step 2: Policy Retrieval (skip in quick mode)
        if not st.session_state.quick_mode:
            status_text.text("📚 Retrieving relevant policies...")
            progress_bar.progress(50)
            retrieved = agents['retriever'].retrieve_relevant_policies(
                user_input, classification['classification']
            )
        else:
            retrieved = []
        
        # Step 3: Explanation
        status_text.text("💭 Generating explanation...")
        progress_bar.progress(75)
        explanation = agents['reasoning'].generate_explanation(
            user_input, classification, retrieved
        )
        
        # Step 4: Recommendation
        status_text.text("⚡ Recommending action...")
        progress_bar.progress(100)
        recommendation = agents['action'].recommend_action(classification)
        
        # Clear progress
        progress_container.empty()
        
        return {
            'classification': classification,
            'retrieved': retrieved,
            'explanation': explanation,
            'recommendation': recommendation
        }
        
    except Exception as e:
        progress_container.empty()
        error_info = agents['error_handler'].handle_error(e, "text analysis")
        st.error(f"❌ {error_info['type']}: {error_info['message']}")
        st.info(f"💡 {error_info['suggestion']}")
        return None

def display_analysis_results(user_input, results):
    """Display analysis results in a consistent format"""
    classification = results['classification']
    retrieved = results['retrieved']
    explanation = results['explanation']
    recommendation = results['recommendation']
    
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
    if not st.session_state.quick_mode:
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

    # Store results in history
    st.session_state.results_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
        'classification': classification['classification'],
        'confidence': classification['confidence'],
        'action': recommendation['action'],
        'severity': recommendation['severity']
    })

def should_auto_analyze(user_input):
    """Check if auto-analysis should be triggered"""
    if not st.session_state.auto_analyze:
        return False
    
    if len(user_input.strip()) < 10:  # Minimum length
        return False
    
    # Check if text has changed
    if user_input == st.session_state.last_auto_analyzed_text:
        return False
    
    # Simple debouncing - check if enough time has passed
    current_time = time.time()
    if current_time - st.session_state.auto_analyze_debounce < 2:  # 2 second debounce
        return False
    
    return True

def render_text_analysis_page(agents):
    """Render the text analysis page"""
    st.title("Text Content Analysis")
    st.markdown("Analyze text content for hate speech, toxicity, and policy violations.")

    st.markdown("---")
    
    # Handle pending analysis from other tabs
    initial_value = ""
    if st.session_state.pending_analysis_text:
        initial_value = st.session_state.pending_analysis_text
        st.session_state.pending_analysis_text = ""  # Clear after using
    
    # Input section
    user_input = st.text_area(
        "**Enter content to analyze:**",
        value=initial_value,
        height=150,
        placeholder="Paste user-generated content here for real-time analysis...",
        key="content_input"
    )
    
    # Manual analyze button
    analyze_btn = st.button("Analyze Content", type="primary", use_container_width=True)
    
    # Auto-analyze functionality with improved logic
    should_auto = should_auto_analyze(user_input)
    
    # Handle auto-analysis flag from other tabs
    if st.session_state.auto_analyze_pending and user_input:
        should_auto = True
        st.session_state.auto_analyze_pending = False

    # Trigger analysis
    if (analyze_btn and user_input) or should_auto:
        if should_auto:
            # Update debounce timer and last analyzed text
            st.session_state.auto_analyze_debounce = time.time()
            st.session_state.last_auto_analyzed_text = user_input
            st.info("🤖 Auto-analyzing content...")
        
        results = perform_analysis(agents, user_input)
        if results:
            display_analysis_results(user_input, results)
    
    # Show auto-analyze status
    if st.session_state.auto_analyze and user_input:
        if len(user_input.strip()) < 10:
            st.caption("ℹ️ Auto-analyze will trigger when you have more than 10 characters")
        elif user_input == st.session_state.last_auto_analyzed_text:
            st.caption("✅ Content already analyzed")
        else:
            time_since_last = time.time() - st.session_state.auto_analyze_debounce
            if time_since_last < 2:
                remaining = 2 - time_since_last
                st.caption(f"⏱️ Auto-analyze in {remaining:.1f} seconds...")
                # Auto-refresh to trigger analysis when debounce is complete
                time.sleep(0.1)
                st.rerun()

def render_audio_analysis_page(agents):
    """Render the audio analysis page"""
    st.title("Audio Content Analysis")
    st.markdown("Upload or record audio content for transcription and analysis.")

    st.markdown("---")
    
    # Audio upload section
    st.markdown("#### Upload Audio File")
    
    audio_file = st.file_uploader(
        "Choose an audio file", 
        type=["wav", "mp3", "flac", "m4a"],
        help="Supported formats: WAV, MP3, FLAC, M4A (Max size: 200MB)"
    )
    
    # Audio processing
    if audio_file:
        # Display audio file info
        st.info(f"📄 File: {audio_file.name} ({round(audio_file.size/1024/1024, 2)} MB)")
        
        # Audio player
        st.audio(audio_file, format='audio/wav')
        
        # Transcription
        if st.button("Transcribe Audio", type="primary", use_container_width=True):
            with st.spinner("🎵 Transcribing audio... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_audio:
                        temp_audio.write(audio_file.getvalue())
                        audio_file_path = temp_audio.name

                        # Transcribe
                        transcription = agents['audio'].transcribe_audio_file(audio_file_path)

                    # Clean up
                    os.unlink(audio_file_path)
                    
                    # Store transcription in session state
                    st.session_state.transcription_result = transcription
                    
                    # Display results
                    st.success("✅ Transcription completed!")
                            
                except Exception as e:
                    st.error(f"❌ Transcription failed: {str(e)}")
                    st.info("💡 Please ensure the audio file is in a supported format and not corrupted.")
    
    # Show transcription result if available
    if st.session_state.transcription_result:
        st.markdown("##### Transcribed Text")
        transcribed_text = st.text_area(
            "Transcription Result:", 
            value=st.session_state.transcription_result,
            height=150,
            key="transcription_display"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Analyze transcribed text button
            if st.button("Analyze Transcribed Content", type="primary", use_container_width=True):
                st.session_state.pending_analysis_text = transcribed_text
                st.session_state.auto_analyze_pending = True
                st.session_state.current_page = "Text Analysis"
                st.rerun()
        
        with col2:
            # Clear transcription button
            if st.button("Clear Transcription", type="secondary", use_container_width=True):
                st.session_state.transcription_result = ""
                st.rerun()

    # Live recording section
    st.markdown("---")
    st.markdown("### Live Recording")

    record_btn = st.button("🎙️ Start Recording", type="primary", use_container_width=True)
    
    if record_btn:
        with st.spinner("🎵 Recording and transcribing... Speak now!"):
            try:
                rt_transcription = agents['audio'].transcribe_real_time_audio()
                
                # Store live transcription in session state
                st.session_state.live_transcription_result = rt_transcription
                
                st.success("✅ Live transcription completed!")
                
            except Exception as e:
                st.error(f"❌ Live transcription failed: {str(e)}")
                st.info("💡 Please check your microphone permissions and try again.")
    
    # Show live transcription result if available
    if st.session_state.live_transcription_result:
        st.markdown("### 📝 Live Transcription")
        live_text = st.text_area(
            "Live Transcription Result:", 
            value=st.session_state.live_transcription_result,
            height=150,
            key="live_transcription_display"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Analyze live transcription button
            if st.button("Analyze Live Content", type="primary", use_container_width=True):
                st.session_state.pending_analysis_text = live_text
                st.session_state.auto_analyze_pending = True
                st.session_state.current_page = "Text Analysis"
                st.rerun()
        
        with col2:
            # Clear live transcription button
            if st.button("Clear Live Transcription", type="secondary", use_container_width=True):
                st.session_state.live_transcription_result = ""
                st.rerun()

def render_history_page():
    """Render the history and analytics page"""
    st.title("Analysis History & Analytics")
    st.markdown("View past analyses and gain insights from your content moderation data.")

    st.markdown("---")
    
    if not st.session_state.results_history:
        st.info("No analysis history yet. Start analyzing content to see results here.")
        return
    
    df = pd.DataFrame(st.session_state.results_history)
    
    # Summary metrics
    st.markdown("### Overview Metrics")
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
    
    # Analytics section
    st.markdown("---")
    st.markdown("### Analytics Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Classification distribution
        st.markdown("#### Classification Distribution")
        classification_counts = df['classification'].value_counts()
        st.bar_chart(classification_counts)
        
    with col2:
        # Action distribution  
        st.markdown("#### Recommended Actions")
        action_counts = df['action'].value_counts()
        st.bar_chart(action_counts)
    
    # Data table with filters
    st.markdown("---")
    st.markdown("### Detailed History")
    
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
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=[],
            help="Select date range to filter results"
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
    
    # Bulk actions
    st.markdown("---")
    st.markdown("### Bulk Actions")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Export Filtered Data", use_container_width=True, type="primary"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"filtered_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                key="download_filtered"
            )
    

def main():
    init_session_state()
    
    # Initialize agents with progress
    if not st.session_state.agents_initialized:
        with st.spinner("🔄 Loading AI models..."):
            agents = initialize_agents()
            st.session_state.agents = agents
            st.session_state.agents_initialized = True
        st.success("✅ Ready to analyze content!")
    else:
        agents = st.session_state.agents

    # Render sidebar navigation
    render_sidebar_navigation()
    
    # Render main content based on current page
    if st.session_state.current_page == "Text Analysis":
        render_text_analysis_page(agents)
    elif st.session_state.current_page == "Audio Analysis":
        render_audio_analysis_page(agents)
    elif st.session_state.current_page == "History":
        render_history_page()

if __name__ == "__main__":
    main()