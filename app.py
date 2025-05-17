import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import sys
import tempfile
from bubble_sheet_processor import BubbleSheetProcessor
from cheating_detection import CheatingDetection as cd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

st.set_page_config(page_title="Bubble Sheet Analyzer", layout="wide")

st.title("📝 Bubble Sheet Analyzer")
st.write("Upload bubble sheets to analyze scores and detect potential cheating patterns")

# Create a two-column layout for input section
col1, col2 = st.columns([1, 1])

# Function to save uploaded files to temp directory
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

# Model answer upload in first column
with col1:
    st.header("Upload Model Answer Sheet")
    model_file = st.file_uploader("Upload the model answer bubble sheet", type=["jpg", "jpeg", "png"])
    if model_file:
        st.image(model_file, caption="Model Answer Sheet", width=300)

# Student answers upload in second column
with col2:
    st.header("Upload Student Answer Sheets")
    student_files = st.file_uploader("Upload student bubble sheets", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if student_files:
        st.write(f"Uploaded {len(student_files)} student answer sheets")
        with st.expander("Preview Uploaded Sheets"):
            for i, file in enumerate(student_files):
                st.image(file, caption=f"Student Sheet {i+1}", width=200)

# Process data when both model and student files are uploaded
if model_file and student_files and st.button("Process Bubble Sheets"):
    with st.spinner("Processing bubble sheets..."):
        try:
            # Save uploaded files to temp directory
            model_path = save_uploaded_file(model_file)
            student_paths = [save_uploaded_file(f) for f in student_files]
            
            # Process bubble sheets
            @st.cache_data
            def process_bubble_sheets(model_path, student_paths):
                # Initialize processor for model answer
                model_processor = BubbleSheetProcessor(model_path)
                _, model_answer_matrix = model_processor.run_pipeline()
                
                student_ids = []
                scores = []
                student_answer_matrices = []
                
                for student_path in student_paths:
                    student_processor = BubbleSheetProcessor(student_path)
                    student_id, student_matrix = student_processor.run_pipeline()
                    student_answer_matrices.append(student_matrix)
                    student_ids.append(student_id)
                    # Score is number of fully matched rows (i.e., correct answers)
                    score = np.sum(np.all(student_matrix == model_answer_matrix, axis=1))
                    scores.append(score)
                
                return {
                    'model_answer_matrix': model_answer_matrix,
                    'students_answers_matrices': student_answer_matrices,
                    'students_ids': student_ids,
                    'scores': scores
                }
            
            processed_data = process_bubble_sheets(model_path, student_paths)
            
            # Clean up temporary files
            os.unlink(model_path)
            for path in student_paths:
                os.unlink(path)
            
            # Store the processed data in session state
            st.session_state.processed_data = processed_data
            
            st.success("Bubble sheets processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing bubble sheets: {e}")
            st.exception(e)

# Display results if data has been processed
if hasattr(st.session_state, 'processed_data'):
    st.header("Analysis Results")
    
    processed_data = st.session_state.processed_data
    students = processed_data['students_answers_matrices']
    model = processed_data['model_answer_matrix']
    scores = processed_data['scores']
    student_ids = processed_data['students_ids']
    
    # Format student IDs for display
    display_ids = []
    for i, sid in enumerate(student_ids):
        if sid.strip():  # If ID was successfully extracted
            display_ids.append(f"ID: {sid}")
        else:
            display_ids.append(f"Student {i+1}")
    
    # Create tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "Results Table", "Cheating Detection", "Cluster Analysis"])
    
    with tab1:
        st.subheader("Score Distribution")
        
        # Calculate statistics
        total_questions = model.shape[0]
        avg_score = np.mean(scores)
        median_score = np.median(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Create columns for stats and histogram
        stat_col, hist_col = st.columns([1, 2])
        
        with stat_col:
            st.metric("Average Score", f"{avg_score:.2f}/{total_questions}")
            st.metric("Median Score", f"{median_score:.2f}/{total_questions}")
            st.metric("Highest Score", f"{max_score}/{total_questions}")
            st.metric("Lowest Score", f"{min_score}/{total_questions}")
        
        with hist_col:
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot histogram with different bins based on number of questions
            bins = min(total_questions // 2, 20)  # Adjust bins based on number of questions
            bins = max(bins, 5)  # Ensure at least 5 bins
            
            ax.hist(scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Score')
            ax.set_ylabel('Number of Students')
            ax.set_title('Distribution of Student Scores')
            ax.grid(axis='y', alpha=0.75)
            
            # Add mean line
            ax.axvline(avg_score, color='red', linestyle='dashed', linewidth=1)
            ax.text(avg_score*1.02, ax.get_ylim()[1]*0.9, f'Mean: {avg_score:.2f}', color='red')
            
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Individual Results")
        
        # Create dataframe for displaying results
        results_df = pd.DataFrame({
            'Student ID': display_ids,
            'Raw ID': student_ids,
            'Score': scores,
            'Percentage': [score/total_questions*100 for score in scores]
        })
        
        # Display the table
        st.dataframe(results_df[['Student ID', 'Score', 'Percentage']], 
                    use_container_width=True,
                    hide_index=True)
        
        # Export to Excel
        results_excel = io.BytesIO()
        results_df.to_excel(results_excel, index=False)
        results_excel.seek(0)
        
        st.download_button(
            label="Download Results as Excel",
            data=results_excel,
            file_name="bubble_sheet_results.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    with tab3:
        st.subheader("Cheating Detection")
        
        # Get parameters for analysis
        n_choices = students[0].shape[1]
        n_questions = students[0].shape[0]
        n_students = len(students)
        accuracy = np.median(scores)/n_questions
        
        # Number of suspicious pairs to display
        num_pairs = st.slider("Number of suspicious pairs to display", 1, 10, 5)
        
        # Get top suspicious pairs
        top_pairs = cd.count_same_wrong_answers_vectorized(model, students, top_n=num_pairs)
        
        if top_pairs:
            # Create a dataframe for the suspicious pairs
            suspicious_data = []
            
            for (i, j), count in top_pairs:
                p, significance = cd.prob_matching_wrong_answers(
                    k=count, 
                    m=n_students, 
                    n=n_questions, 
                    c=n_choices, 
                    accuracy=accuracy,
                    alpha=.001
                )
                
                suspicious_data.append({
                    'Student 1': display_ids[i],
                    'Student 2': display_ids[j],
                    'Identical Wrong Answers': count,
                    'p-value': p,
                    'Significant': significance
                })
            
            suspicious_df = pd.DataFrame(suspicious_data)
            
            # Display the suspicious pairs
            st.dataframe(
                suspicious_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "p-value": st.column_config.NumberColumn(format="%.6f"),
                    "Significant": st.column_config.CheckboxColumn()
                }
            )
            
            # Add explanations
            st.info("""
            **Understanding the Results**:
            - **Identical Wrong Answers**: The number of questions where both students selected the same incorrect answer
            - **p-value**: The probability of this level of similarity occurring by chance
            - **Significant**: Whether the similarity is statistically significant (unlikely to occur by chance)
            """)
        else:
            st.write("No suspicious pairs detected.")
    
    with tab4:
        st.subheader("Cluster Analysis")
        
        # Apply clustering to detect groups of similar answer patterns
        labels = cd.get_clusters(students)
        
        # Apply PCA
        X = np.array([student.flatten() for student in students])
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        # Calculate the explained variance
        explained_variance = pca.explained_variance_ratio_
        total_variance = sum(explained_variance)
        
        colors_scheme = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22']
        colors = []
        
        for label in labels:
            if label == -1:  # Noise points in DBSCAN
                colors.append('gray')
            else:
                colors.append(colors_scheme[label % len(colors_scheme)])
        
        # Convert to DataFrame for plotly
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'Student_ID': display_ids,
            'Score': scores,
            'Cluster': ["Cluster " + str(l) if l >= 0 else "Outlier" for l in labels],
            'Label': [str(l) for l in labels]  # <--- add this line
        })

        import plotly.express as px

        fig = px.scatter_3d(
            df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Label',
            hover_data=['Student_ID', 'Score', 'Cluster']
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({explained_variance[0]:.2%})',
                yaxis_title=f'PC2 ({explained_variance[1]:.2%})',
                zaxis_title=f'PC3 ({explained_variance[2]:.2%})',
                aspectmode='data'
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=0)
        )


        # Show the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Add variance explanation
        st.write(f"**Total variance explained by first 3 PCs**: {total_variance:.2%}")
        
        # Show variance breakdown
        variance_data = [{"PC": f"PC{i+1}", "Variance Explained": var} 
                        for i, var in enumerate(explained_variance)]
        
        variance_df = pd.DataFrame(variance_data)
        
        st.dataframe(
            variance_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Variance Explained": st.column_config.ProgressColumn(
                    "Variance Explained",
                    help="Proportion of variance explained by each principal component",
                    format="%.2f%%",
                    min_value=0,
                    max_value=max(explained_variance) * 1.1  # To give some headroom in the progress bar
                )
            }
        )
        
        # Add explanation of the plot
        st.info("""
        **Understanding the Cluster Plot**:
        - Each point represents a student's answer pattern
        - Students with similar answer patterns appear closer together
        - Different colors represent different clusters detected by the algorithm
        - Outliers (gray points) don't fit well into any cluster
        - Students in the same cluster may have collaborated or have similar learning patterns
        """)
