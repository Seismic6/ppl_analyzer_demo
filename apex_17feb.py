import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import pandas as pd
import numpy as np
import json
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Constants
COLOR_SCHEME = {
    'primary': '#1E88E5',
    'secondary': '#FFB300',
    'background': '#F5F7FA',
    'text': '#212121',
    'success': '#4CAF50'
}

# Utility Functions
def batch_responses(responses, batch_size):
    """Split responses into batches of specified size."""
    return [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]

def clean_and_parse_json(json_text):
    """Clean and parse JSON text into a list of dictionaries."""
    try:
        cleaned_text = re.sub(r'```json\s*|\s*```', '', json_text).strip()
        cleaned_text = cleaned_text.lstrip('json').lstrip('JSON').strip()
        parsed_json = json.loads(cleaned_text)
        return parsed_json if isinstance(parsed_json, list) else [parsed_json]
    except json.JSONDecodeError as e:
        st.warning(f"Failed to parse JSON: {e}")
        return []

# Unified Evaluation Function with Caching
@st.cache_data
def evaluate_responses(survey_question, responses, is_batch, batch_size, generation_config):
    """Evaluate survey responses based on multiple criteria."""
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    responses = responses if isinstance(responses, list) else [responses]
    batches = batch_responses(responses, batch_size) if is_batch else [[responses[0]]]
    all_evaluations = []
    
    evaluation_prompt = """
Evaluate each response based on these criteria:

1. Response: Full text of the answer (truncate after 100 characters if too long)

2. Relevance (0-10): How directly and accurately the response addresses the specific question asked
   - 0-3: Barely relevant or misses the point
   - 4-6: Partially addresses the question
   - 7-10: Directly addresses the core of the question

3. Completeness (0-10): How thoroughly the response answers all aspects of the question
   - 0-3: Highly incomplete, missing major elements
   - 4-6: Addresses some but not all aspects
   - 7-10: Comprehensively covers all aspects of the question

4. Specificity (0-10): Level of detail, precision, and concrete information provided
   - 0-3: Vague, general statements only
   - 4-6: Some specific details but lacks depth
   - 7-10: Rich in specific, precise details and examples

5. Language Quality (0-10): Coherence, clarity, grammar, and appropriate vocabulary
   - 0-3: Poor grammar, difficult to understand
   - 4-6: Generally understandable but with issues
   - 7-10: Well-written, clear, and grammatically sound

6. Sentiment Alignment (0-10): How well the tone matches what's appropriate for the question
   - 0-3: Completely mismatched tone
   - 4-6: Somewhat appropriate tone
   - 7-10: Perfectly matched tone and sentiment

7. Topic Status: "On-topic" or "Off-topic" 
   - Strictly evaluate whether the response addresses the question's subject matter
   - Be critical in assessment and mark as Off-topic if the response:
     
     * Mentions incorrect brand categories (e.g., if asked about luxury perfume brands and "Axe body spray" is mentioned)
     * Discusses wrong time periods (e.g., if asked about 1980s movies and response mentions 2010s films)
     * References incorrect geographic locations (e.g., if asked about Italian cuisine and response focuses on French dishes)
     * Addresses different price tiers (e.g., if asked about budget smartphones and response discusses premium $1000+ models)
     * Covers different industry segments (e.g., if asked about commercial aircraft and response discusses military jets)
     * Mentions incorrect professional fields (e.g., if asked about cardiologists and response discusses dermatologists)
     * Discusses different product categories (e.g., if asked about gaming laptops and response recommends tablets)
     * Answers a different question entirely (e.g., if asked "What makes a good leader?" and response discusses cooking techniques)

8. Sentiment Scores: Rate each on scale of 0-10
   - Positive: Degree of positive emotion/content
   - Negative: Degree of negative emotion/content
   - Neutral: Degree of neutral/objective content

9. Overall Score (0-100): Calculated as follows:
   - Sum the five main metrics (Relevance + Completeness + Specificity + Language Quality + Sentiment Alignment)
   - This gives a score out of 50
   - Multiply by 2 to convert to a 0-100 scale: Overall Score = (Sum of 5 metrics) × 2

10. Explanation: Provide a 1-2 sentence justification for the overall evaluation

Output as a JSON array with these keys:
"response", "relevance", "completeness", "specificity", "language_quality", 
"sentiment_alignment", "topic_status", "sentiment_positive", "sentiment_negative", 
"sentiment_neutral", "overall_score", "explanation"
"""
    
    if is_batch:
        progress_bar = st.progress(0)
    start_time = time.time()
    
    for i, batch in enumerate(batches):
        prompt = f"""
Survey Question: "{survey_question}"
{evaluation_prompt}
Responses: {json.dumps(batch)}
"""
        try:
            result = model.generate_content(prompt, generation_config=generation_config)
            parsed_batch = clean_and_parse_json(result.text)
            all_evaluations.extend(parsed_batch)
        except Exception as e:
            st.error(f"Error generating content for batch {i+1}: {e}")
            continue
        
        if is_batch:
            progress = (i + 1) / len(batches)
            progress_bar.progress(progress, text=f"{progress * 100:.1f}% completed")
            time.sleep(2)
    
    st.sidebar.write(f"Execution time: {time.time() - start_time:.2f}s")
    return pd.DataFrame(all_evaluations)

# Visualization Functions
def create_word_cloud(responses):
    """Generate a word cloud from responses."""
    text = ' '.join(responses)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def create_violin_plot(df, metric):
    """Create a violin plot for a specified metric."""
    if metric not in df.columns:
        st.error(f"Metric '{metric}' not available.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, y=metric, ax=ax, inner='stick')
    ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
    st.pyplot(fig)

# Clustering Function
def cluster_responses(responses, threshold):
    """Cluster responses based on similarity."""
    df = pd.DataFrame({'response': responses}).astype(str).apply(lambda x: x.str.lower())
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(df['response'])
    except ValueError:
        st.error("Error: Empty vocabulary.")
        return df
    
    similarity_matrix = cosine_similarity(X)
    distance_matrix = np.maximum(1 - similarity_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    df['Group'] = clusters

    # Calculate similarity scores
    similarity_scores = []
    for i, c in enumerate(clusters):
        same_cluster_indices = [j for j, g in enumerate(clusters) if g == c and j != i]
        if same_cluster_indices:
            score = np.max(similarity_matrix[i, same_cluster_indices])
        else:
            score = np.nan
        similarity_scores.append(score)
    
    df['Similarity Score'] = similarity_scores
    
    # Handle single-response clusters
    unique_groups = df['Group'].value_counts()[lambda x: x == 1].index
    df.loc[df['Group'].isin(unique_groups), 'Group'] = "Unique"
    df.loc[df['Group'] == "Unique", 'Similarity Score'] = np.nan
    
    return df

# Radar Chart Function
def create_radar_chart(values, categories):
    """Generate a radar chart for given values and categories."""
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values = list(values)
    values += values[:1]
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2)  # Start at top
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', color=COLOR_SCHEME['primary'])
    ax.fill(angles, values, color=COLOR_SCHEME['primary'], alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], color='grey', size=7)
    
    ax.set_ylim(0, 10)
    
    return fig

# Single Response Dashboard
def render_single_response_dashboard(df):
    """Render a dashboard for a single response with a radar chart."""
    if df.empty or len(df) == 0:
        return
    
    row = df.iloc[0]
    
    # Custom CSS for metric cards
    st.markdown("""
    <style>
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .metric-title {
        color: #757575;
        font-size: 0.9rem;
    }
    .metric-value {
        color: #1E88E5;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    with st.container():
        st.markdown(f"### Response Analysis: {row['response'][:30]}{'...' if len(row['response']) > 30 else ''}")
        
        # Metrics and Radar Chart
        with st.container():
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Relevance</div>
                            <div class="metric-value">{row['relevance']}/10</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Completeness</div>
                            <div class="metric-value">{row['completeness']}/10</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Specificity</div>
                            <div class="metric-value">{row['specificity']}/10</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with st.container():
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Language Quality</div>
                            <div class="metric-value">{row['language_quality']}/10</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col5:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Sentiment Alignment</div>
                            <div class="metric-value">{row['sentiment_alignment']}/10</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col6:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Overall Score</div>
                            <div class="metric-value">{row['overall_score']}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_right:
                st.subheader("Metric Radar Chart")
                radar_values = [row['relevance'], row['completeness'], row['specificity'], 
                                row['language_quality'], row['sentiment_alignment']]
                radar_categories = ['Relevance', 'Completeness', 'Specificity', 
                                    'Language Quality', 'Sentiment Alignment']
                fig = create_radar_chart(radar_values, radar_categories)
                st.pyplot(fig)
        
        # Sentiment and Explanation
        with st.container():
            st.markdown("#### Sentiment Analysis")
            sentiment_col, explanation_col = st.columns([1, 2])
            
            with sentiment_col:
                st.progress(row['sentiment_positive'] / 10)
                st.caption(f"Positive: {row['sentiment_positive']}")
                st.progress(row['sentiment_negative'] / 10)
                st.caption(f"Negative: {row['sentiment_negative']}")
                st.progress(row['sentiment_neutral'] / 10)
                st.caption(f"Neutral: {row['sentiment_neutral']}")
                
            with explanation_col:
                st.info(f"**Explanation:** {row['explanation']}")
                st.write(f"**Topic Status:** {row['topic_status']}")

# Main Application
st.set_page_config(layout="wide", page_title="Survey Response Analyzer")

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    st.sidebar.header("Initialization")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if st.sidebar.button("Initialize"):
        if api_key:
            try:
                genai.configure(api_key=api_key)
                st.session_state.initialized = True
                st.success("Initialized successfully.")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
        else:
            st.error("API Key required.")
else:
    st.sidebar.header("Settings")
    batch_size = st.sidebar.slider("Batch Size", 1, 50, 25)
    generation_config = GenerationConfig(
        temperature=st.sidebar.slider("Temperature", 0.0, 1.0, 0.2),
        top_k=st.sidebar.number_input("Top K", 1, value=40),
        top_p=st.sidebar.slider("Top P", 0.0, 1.0, 0.8),
        max_output_tokens=st.sidebar.number_input("Max Tokens", 100, 8192, 8192)
    )

if st.session_state.initialized:
    tabs = st.tabs(["Response Analysis", "Visualizations", "Clustering", "History"])
    
    # Response Analysis Tab
    with tabs[0]:
        st.header("Response Analysis")
        mode = st.radio("Mode", ["Single Response", "Batch Responses"])
        survey_question = st.text_area("Survey Question")
        responses_input = st.text_area("Response(s)", 
                                     placeholder="One response for Single mode, one per line for Batch mode")
        
        if st.button("Analyze"):
            if survey_question and responses_input:
                responses = [r.strip() for r in responses_input.split("\n")] if mode == "Batch Responses" else [responses_input.strip()]
                with st.spinner("Analyzing..."):
                    df = evaluate_responses(survey_question, responses, mode == "Batch Responses", batch_size, generation_config)
                    st.session_state.analysis_df = df
                    st.session_state.responses = responses
                    st.session_state.survey_question = survey_question
                
                if mode == "Single Response":
                    render_single_response_dashboard(df)
                else:
                    st.dataframe(df)
                    st.subheader("Summary Statistics")
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    summary = df[numerical_cols].describe().T
                    st.table(summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, "analysis_results.csv", "text/csv")
            else:
                st.error("Please provide question and response(s).")
    
    # Visualizations Tab
    with tabs[1]:
        st.header("Visualizations")
        if "analysis_df" in st.session_state:
            create_word_cloud(st.session_state.responses)
            numerical_cols = st.session_state.analysis_df.select_dtypes(include=[np.number]).columns
            metric = st.selectbox("Metric", options=numerical_cols)
            create_violin_plot(st.session_state.analysis_df, metric)
        else:
            st.info("Run an analysis first.")
    
    # Clustering Tab
    with tabs[2]:
        st.header("Response Clustering")
        if "responses" in st.session_state:
            threshold = 1 - (st.slider("Similarity Threshold (%)", 1, 100, 30) / 100)
            if st.button("Cluster"):
                with st.spinner("Clustering..."):
                    cluster_df = cluster_responses(st.session_state.responses, threshold)
                    st.session_state.clustering_df = cluster_df
                    st.dataframe(cluster_df)
        else:
            st.info("Run an analysis first.")
    
    # History Tab
    with tabs[3]:
        st.header("History")
        if "analysis_df" in st.session_state:
            st.subheader("Latest Analysis")
            st.dataframe(st.session_state.analysis_df)
        if "clustering_df" in st.session_state:
            st.subheader("Latest Clustering")
            st.dataframe(st.session_state.clustering_df)

if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()
