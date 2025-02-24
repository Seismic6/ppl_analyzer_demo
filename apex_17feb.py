import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig # Modified import statement
from google.generativeai.generative_models import GenerativeModel
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

###############################################
# Utility Functions
###############################################

def batch_responses(responses, batch_size):
    """
    Batch the given responses into chunks of size batch_size.
    """
    return [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]

def clean_and_parse_json(json_text):
    """
    Remove code block markers and common prefixes and parse a JSON string.
    Returns a list of JSON objects.
    """
    try:
        cleaned_text = re.sub(r'```json\s*|\s*```', '', json_text)
        cleaned_text = cleaned_text.strip().lstrip('json').lstrip('JSON').strip()
        parsed_json = json.loads(cleaned_text)
        if isinstance(parsed_json, dict):
            parsed_json = [parsed_json]
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {e}")
        st.error(f"Original text: {json_text}")
        try:
            fixed_text = cleaned_text.replace("'", '"')
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            return json.loads(fixed_text)
        except Exception as ex:
            st.error("Could not parse JSON even after attempted fixes.")
            return []

###############################################
# Evaluation Module Functions
###############################################

def generate_bulk_evaluations(survey_question, responses, batch_size, generation_config):
    """
    Evaluate survey responses in batches using Google AI Studio API.
    For each response, the evaluation includes:
      - response (text; you may truncate after three words if needed)
      - relevance (0-10)
      - completeness (0-10)
      - specificity (0-10)
      - language_quality (0-10)
      - sentiment_alignment (0-10)
      - overall_score (0-100) [weighted sum]
      - explanation (brief justification)
    The output is expected as a JSON array.
    """
    model = genai.GenerativeModel("gemini-2.0-flash-001") # Corrected model call
    batches = batch_responses(responses, batch_size)
    all_evaluations = []

    evaluation_instructions = f"""
Evaluate the survey question and each response according to these criteria:
1. Response: Provide the full text of the survey response (if too long, you may truncate after the first three words).
2. Relevance (0-10): To what extent does the response directly address the survey question?
3. Completeness (0-10): Does the response provide a full and meaningful answer?
4. Specificity (0-10): How much detail, examples, or personal insight does the response contain?
5. Language Quality (0-10): Is the response coherent, grammatically correct, and well-written?
6. Sentiment Alignment (0-10): Does the sentiment match the tone or context of the question?
7. Overall Score (0-100): Provide a weighted sum of the individual scores.
8. Explanation: Offer a short justification for the scores.

Output your analysis as a JSON array. Each object should have keys:
"response", "relevance", "completeness", "specificity", "language_quality", "sentiment_alignment", "overall_score", "explanation".

Respond only with the JSON.
    """

    progress_bar = st.progress(0)
    start_time = time.time()

    for i, batch in enumerate(batches):
        prompt = f"""
Survey Question: "{survey_question}"

{evaluation_instructions}

Responses: {json.dumps(batch)}
        """
        result = model.generate_content(prompt, generation_config=generation_config)
        st.write(f"Batch {i + 1} raw output:")
        st.code(result.text, language="json")

        parsed_batch = clean_and_parse_json(result.text)
        all_evaluations.extend(parsed_batch)

        # Update progress bar
        progress = (i + 1) / len(batches)
        progress_bar.progress(progress, text=f"{progress * 100:.1f}% completed")
        time.sleep(6)

    st.sidebar.write(f"Total execution time: {time.time() - start_time:.2f} seconds")
    df = pd.DataFrame(all_evaluations)
    return df

def generate_single_evaluation(survey_question, response, generation_config):
    """
    Evaluate a single survey response in detail.
    Returns a dictionary with keys:
    "response", "relevance", "completeness", "specificity", "language_quality",
    "sentiment_alignment", "overall_score", "explanation"
    """
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    prompt = f"""
Survey Question: "{survey_question}"

Evaluate the following response in detail based on these criteria:
1. Response (if necessary, truncate after three words)
2. Relevance (0-10)
3. Completeness (0-10)
4. Specificity (0-10)
5. Language Quality (0-10)
6. Sentiment Alignment (0-10)
7. Overall Score (0-100) [weighted sum]
8. Explanation ((avoid fluff and keep it brief and to the point, ONE SENTENCE ONLY))

Output your analysis as a JSON array with one object having keys:
"response", "relevance", "completeness", "specificity", "language_quality", "sentiment_alignment", "overall_score", "explanation".

Response: "{response}"
    """
    result = model.generate_content(prompt, generation_config=generation_config)
    with st.expander("Show raw json for single response evaluation"):
        st.code(result.text, language="json")
    parsed_output = clean_and_parse_json(result.text)
    if parsed_output:
        return parsed_output[0]
    else:
        return {}

def check_response_relevancy(survey_question, response, generation_config):
    """
    Check if the given response is On-topic or Off-topic relative to the survey question,
    and also return sentiment analysis scores (positive, negative, neutral on a 0-10 scale).
    Returns a JSON object with keys:
    "result": either "On-topic" or "Off-topic",
    "explanation": a brief justification,
    "positive": score (0-10),
    "negative": score (0-10),
    "neutral": score (0-10).
    """
    model = genai.GenerativeModel("gemini-2.0-flash-001") 
    prompt = f"""
Survey Question: "{survey_question}"

Response: "{response}"

Determine whether the response is On-topic or Off-topic in relation to the survey question.
Then, analyze the sentiment of the response and provide scores on a 0 to 10 scale for the following:
- "positive": the degree of positive sentiment,
- "negative": the degree of negative sentiment,
- "neutral": the degree of neutrality.
Provide only a JSON object with the following keys:
"result" (either "On-topic" or "Off-topic"),
"explanation" (short, maximum 2 sentences),
"positive", "negative", "neutral".
    """
    result = model.generate_content(prompt, generation_config=generation_config)
    with st.expander("Show raw json for question relevancy check"):
        st.code(result.text, language="json")
    parsed_output = clean_and_parse_json(result.text)
    if parsed_output and isinstance(parsed_output, list):
        return parsed_output[0]
    elif isinstance(parsed_output, dict):
        return parsed_output
    else:
        return {"result": "Unknown", "explanation": "Could not determine relevancy.", "positive": 0, "negative": 0, "neutral": 0}

###############################################
# Visualization Module Functions
###############################################

def create_word_cloud(responses):
    """
    Generate and display a word cloud based on the responses.
    """
    text = ' '.join(responses)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def create_violin_plot(df, metric):
    """
    Create a violin plot for the selected metric from the evaluation DataFrame.
    """
    if metric not in df.columns:
        st.error(f"Metric '{metric}' not available in evaluation data.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, y=metric, ax=ax, inner='stick')
    ax.set_title(f"Distribution of {metric.replace('_', ' ').title()}")
    ax.set_ylabel(metric.replace('_', ' ').title())
    st.pyplot(fig)

###############################################
# Similarity & Clustering Module
###############################################

def cluster_responses(responses, threshold):
    """
    Cluster responses based on cosine similarity (using TF-IDF).
    Returns a DataFrame with the original responses, assigned cluster groups,
    and similarity scores.
    """
    df = pd.DataFrame({'response': responses})
    df['response'] = df['response'].astype(str).str.lower()

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(df['response'])
    except ValueError:
        st.error("Error: Empty vocabulary. Responses may be empty or only contain stop words.")
        return df

    similarity_matrix = cosine_similarity(X)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.maximum(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distance, method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    df['Group'] = clusters

    similarity_scores = []
    for i, group in enumerate(clusters):
        same_group = [j for j, g in enumerate(clusters) if g == group and j != i]
        if same_group:
            max_sim = max(similarity_matrix[i, same_group])
            similarity_scores.append(max_sim)
        else:
            similarity_scores.append(np.nan)
    df['Similarity Score'] = similarity_scores

    group_counts = df['Group'].value_counts()
    unique_groups = group_counts[group_counts == 1].index.tolist()
    df.loc[df['Group'].isin(unique_groups), ['Group', 'Similarity Score']] = ""
    return df

###############################################
# Main UI and Application
###############################################

st.set_page_config(layout="wide", page_title="Survey Response Evaluation App")

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    st.sidebar.header("Google AI Studio Initialization")
    api_key = st.sidebar.text_input("Gemini API Key, generate at https://aistudio.google.com/apikey", type="password")
    if st.sidebar.button("Initialize App"):
        if api_key:
            try:
                genai.configure(api_key=api_key)
                st.session_state.initialized = True
                st.success("App initialized successfully.")
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
        else:
            st.error("Please provide your Google AI Studio API Key.")
else:
    st.sidebar.header("Generative Model Settings")
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=50, value=25)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)
    top_k = st.sidebar.number_input("Top K", min_value=1, value=40)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.8)
    max_output_tokens = st.sidebar.number_input("Max Output Tokens", min_value=100, value=8192, max_value=8192)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )

if st.session_state.initialized:

    tabs = st.tabs([
        "Bulk Evaluation",
        "Single Response Analysis",
        "Visualizations",
        "Similarity & Clustering",
        "History"
    ])

    ###############################################
    # Tab 1: Bulk Evaluation
    ###############################################
    with tabs[0]:
        st.header("Bulk Evaluation of Survey Responses")
        st.info("Enter a survey question and multiple responses (one per line) to perform a bulk evaluation.")
        survey_question_bulk = st.text_area("Survey Question")
        responses_text = st.text_area("Responses (one per line)")
        if st.button("Run Bulk Evaluation", key="bulk_eval"):
            if survey_question_bulk and responses_text:
                responses_list = [line.strip() for line in responses_text.split("\n") if line.strip()]
                st.session_state.survey_question = survey_question_bulk
                st.session_state.responses = responses_list
                with st.spinner("Evaluating responses with generative AI…"):
                    bulk_df = generate_bulk_evaluations(survey_question_bulk, responses_list, batch_size, generation_config)
                st.session_state.bulk_evaluation_df = bulk_df
                st.success("Bulk evaluation completed.")
                st.dataframe(bulk_df)
                csv = bulk_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Evaluation CSV", csv, "evaluation_results.csv", "text/csv")
            else:
                st.error("Please fill in both the survey question and responses.")

    ###############################################
    # Tab 2: Single Response Analysis (Combined)
    ###############################################
    with tabs[1]:
        st.header("Response Analysis")
        st.info("Enter a survey question and a single response. The system will provide a detailed evaluation and on-topic/sentiment analysis.")
        survey_question_single = st.text_area("Question", key="single_q")
        single_response = st.text_area("Response", key="single_r")
        if st.button("Analyze Single Response", key="single_eval"):
            if survey_question_single and single_response:
                with st.spinner("Running analysis…"):
                    single_eval = generate_single_evaluation(survey_question_single, single_response, generation_config)
                    relevancy_result = check_response_relevancy(survey_question_single, single_response, generation_config)
                st.session_state.single_evaluation = single_eval
                st.session_state.relevancy_result = relevancy_result
                st.success("Single response analysis completed.")

                # Create a visually appealing display for Evaluation Metrics
                st.markdown("## Evaluation Metrics")
                st.markdown(f"**Response:** {single_eval.get('response','')}")
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Relevance", single_eval.get("relevance", "N/A"))
                col2.metric("Completeness", single_eval.get("completeness", "N/A"))
                col3.metric("Specificity", single_eval.get("specificity", "N/A"))
                col4.metric("Language Quality", single_eval.get("language_quality", "N/A"))
                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment Alignment", single_eval.get("sentiment_alignment", "N/A"))
                col2.metric("Overall Score (Out of 100)", single_eval.get("overall_score", "N/A"))
                with st.expander("Explanation"):
                    st.markdown(single_eval.get("explanation", ""))

                st.markdown("---")
                st.markdown("## Relevancy & Sentiment Analysis")
                st.markdown(f"**Result:** {relevancy_result.get('result','N/A')}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Positive", relevancy_result.get("positive", "N/A"))
                col2.metric("Negative", relevancy_result.get("negative", "N/A"))
                col3.metric("Neutral", relevancy_result.get("neutral", "N/A"))
                with st.expander("Explanation"):
                    st.markdown(relevancy_result.get("explanation", ""))

            else:
                st.error("Please provide both a survey question and a single response.")

    ###############################################
    # Tab 3: Visualizations
    ###############################################
    with tabs[2]:
        st.header("Visualization of Evaluation Data")
        if "bulk_evaluation_df" in st.session_state:
            st.subheader("Word Cloud of Original Responses")
            create_word_cloud(st.session_state.responses)
            st.subheader("Violin Plot of Evaluation Metrics")
            metric = st.selectbox("Select Metric", options=[
                "relevance", "completeness", "specificity", "language_quality", "sentiment_alignment", "overall_score"
            ])
            create_violin_plot(st.session_state.bulk_evaluation_df, metric)
        else:
            st.info("No bulk evaluation data available yet. Please run a bulk evaluation first.")

    ###############################################
    # Tab 4: Similarity & Clustering
    ###############################################
    with tabs[3]:
        st.header("Similarity & Clustering of Responses")
        if "responses" in st.session_state:
            similarity_percent = st.slider("Similarity Threshold (%)", min_value=1, max_value=100, value=30)
            threshold = 1 - (similarity_percent / 100)
            if st.button("Run Similarity Check", key="sim_check"):
                with st.spinner("Clustering responses…"):
                    cluster_df = cluster_responses(st.session_state.responses, threshold)
                st.session_state.clustering_df = cluster_df
                st.dataframe(cluster_df)
        else:
            st.info("Please run a bulk evaluation first to have responses to cluster.")

    ###############################################
    # Tab 5: History
    ###############################################
    with tabs[4]:
        st.header("History")
        if "bulk_evaluation_df" in st.session_state:
            st.subheader("Bulk Evaluation Results")
            st.dataframe(st.session_state.bulk_evaluation_df)
        else:
            st.info("No bulk evaluation results available.")
        if "single_evaluation" in st.session_state and "relevancy_result" in st.session_state:
            st.subheader("Single Response Analysis")
            st.markdown("### Evaluation Metrics")
            st.table(pd.DataFrame([st.session_state.single_evaluation]))
            st.markdown("### Relevancy & Sentiment Analysis")
            st.table(pd.DataFrame([st.session_state.relevancy_result]))
        else:
            st.info("No single response analysis available.")
        if "clustering_df" in st.session_state:
            st.subheader("Similarity & Clustering Results")
            st.dataframe(st.session_state.clustering_df)
        else:
            st.info("No similarity clustering results available.")

###############################################
# Reset Application
###############################################
if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()
