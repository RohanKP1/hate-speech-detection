import streamlit as st
import os
from agents.hate_speech_agent import HateSpeechDetectionAgent
from agents.retriever_agent import HybridRetrieverAgent
from agents.reasoning_agent import PolicyReasoningAgent
from agents.action_agent import ActionRecommenderAgent
from agents.error_handler import ErrorHandlerAgent
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection Assistant",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

def initialize_agents():
    """Initialize all agents"""
    try:
        policy_docs_path = "data/policy_docs"
        
        if not os.path.exists(policy_docs_path):
            st.error(f"Policy documents directory not found: {policy_docs_path}")
            st.stop()
        
        hate_speech_agent = HateSpeechDetectionAgent()
        retriever_agent = HybridRetrieverAgent(policy_docs_path)
        reasoning_agent = PolicyReasoningAgent()
        action_agent = ActionRecommenderAgent()
        error_handler = ErrorHandlerAgent()
        
        return hate_speech_agent, retriever_agent, reasoning_agent, action_agent, error_handler
        
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        st.stop()

def main():
    # Header
    st.title("🛡️ Hate Speech Detection Assistant")
    st.markdown("An AI-powered content moderation tool that classifies text, retrieves relevant policies, and recommends actions.")
    
    # Initialize agents
    with st.spinner("Initializing AI agents..."):
        hate_speech_agent, retriever_agent, reasoning_agent, action_agent, error_handler = initialize_agents()
    
    # Sidebar
    st.sidebar.header("📊 Analysis Controls")
    
    # Input section
    st.header("📝 Content Analysis")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste the content you want to analyze for hate speech, toxicity, or policy violations..."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Content", type="primary"):
        if user_input:
            # Validate input
            validation = error_handler.validate_input(user_input)
            
            if not validation['valid']:
                st.error(validation['message'])
                return
            
            # Create columns for results
            col1, col2 = st.columns([1, 1])
            
            with st.spinner("Analyzing content..."):
                try:
                    # Step 1: Classify text
                    with st.status("Step 1: Classifying content...", expanded=True) as status:
                        classification = hate_speech_agent.classify_text(user_input)
                        st.write(f"Classification: {classification['classification']}")
                        status.update(label="Classification complete!", state="complete", expanded=False)
                    
                    # Step 2: Retrieve relevant policies
                    with st.status("Step 2: Retrieving relevant policies...", expanded=True) as status:
                        retrieved_policies = retriever_agent.retrieve_relevant_policies(
                            user_input, classification['classification']
                        )
                        st.write(f"Found {len(retrieved_policies)} relevant policies")
                        status.update(label="Policy retrieval complete!", state="complete", expanded=False)
                    
                    # Step 3: Generate explanation
                    with st.status("Step 3: Generating detailed explanation...", expanded=True) as status:
                        explanation = reasoning_agent.generate_explanation(
                            user_input, classification, retrieved_policies
                        )
                        status.update(label="Explanation generated!", state="complete", expanded=False)
                    
                    # Step 4: Recommend action
                    with st.status("Step 4: Recommending moderation action...", expanded=True) as status:
                        action_recommendation = action_agent.recommend_action(classification)
                        status.update(label="Action recommendation complete!", state="complete", expanded=False)
                    
                    # Display results
                    st.success("Analysis completed successfully!")
                    
                    # Results display
                    with col1:
                        st.subheader("🎯 Classification Results")
                        
                        # Classification card
                        classification_color = {
                            'Hate': '🔴',
                            'Toxic': '🟠', 
                            'Offensive': '🟡',
                            'Neutral': '🟢',
                            'Ambiguous': '🔵'
                        }
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; border-left: 4px solid #1f77b4;">
                            <h4>{classification_color.get(classification['classification'], '❓')} {classification['classification']}</h4>
                            <p><strong>Confidence:</strong> {classification['confidence']}</p>
                            <p><strong>Initial Reason:</strong> {classification['reason']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Action recommendation
                        st.subheader("⚡ Recommended Action")
                        
                        action_color = {
                            'REMOVE_AND_BAN': '🔴',
                            'REMOVE_AND_WARN': '🟠',
                            'WARN_USER': '🟡',
                            'FLAG_FOR_REVIEW': '🔵',
                            'ALLOW': '🟢'
                        }
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; border-left: 4px solid #ff6b6b;">
                            <h4>{action_color.get(action_recommendation['action'], '❓')} {action_recommendation['action']}</h4>
                            <p><strong>Severity:</strong> {action_recommendation['severity']}</p>
                            <p><strong>Description:</strong> {action_recommendation['description']}</p>
                            <p><strong>Reasoning:</strong> {action_recommendation['reasoning']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("📋 Retrieved Policies")
                        
                        if retrieved_policies:
                            for i, policy in enumerate(retrieved_policies, 1):
                                with st.expander(f"📄 {policy['source']} (Relevance: {policy['relevance_score']})"):
                                    st.write(policy['content'])
                        else:
                            st.info("No specific policies retrieved for this content.")
                        
                        st.subheader("🔍 Detailed Analysis")
                        st.write(explanation)
                    
                    # Save to history
                    result_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'classification': classification['classification'],
                        'confidence': classification['confidence'],
                        'action': action_recommendation['action'],
                        'severity': action_recommendation['severity']
                    }
                    st.session_state.results_history.append(result_entry)
                    
                except Exception as e:
                    error_info = error_handler.handle_error(e, "content analysis")
                    st.error(f"**{error_info['type']}:** {error_info['message']}")
                    st.info(f"**Suggestion:** {error_info['suggestion']}")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Results history
    if st.session_state.results_history:
        st.header("📈 Analysis History")
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.results_history)
        
        # Display summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(df))
        
        with col2:
            hate_count = len(df[df['classification'] == 'Hate'])
            st.metric("Hate Speech Detected", hate_count)
        
        with col3:
            toxic_count = len(df[df['classification'] == 'Toxic'])
            st.metric("Toxic Content", toxic_count)
        
        with col4:
            neutral_count = len(df[df['classification'] == 'Neutral'])
            st.metric("Neutral Content", neutral_count)
        
        # Display history table
        st.dataframe(df, use_container_width=True)
        
        # Export option
        if st.button("📊 Export Results to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"hate_speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Sidebar information
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This tool uses AI to:
    - Classify content as Hate, Toxic, Offensive, Neutral, or Ambiguous
    - Retrieve relevant policy documents
    - Explain classification decisions
    - Recommend moderation actions
    
    **Classification Categories:**
    - **Hate:** Promotes hatred/violence against groups
    - **Toxic:** Harmful, abusive content
    - **Offensive:** Inappropriate, disrespectful content
    - **Neutral:** Policy-compliant content
    - **Ambiguous:** Unclear, needs review
    """)
    
    st.sidebar.header("🔧 System Status")
    if os.path.exists("data/policy_docs"):
        policy_files = [f for f in os.listdir("data/policy_docs") if f.endswith('.txt')]
        st.sidebar.success(f"✅ {len(policy_files)} policy documents loaded")
        
        with st.sidebar.expander("📋 Policy Documents"):
            for file in policy_files:
                st.sidebar.write(f"• {file.replace('.txt', '').replace('_', ' ').title()}")
    else:
        st.sidebar.error("❌ Policy documents not found")
    
    # Clear history button
    if st.session_state.results_history:
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.results_history = []
            st.rerun()

if __name__ == "__main__":
    main()