import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb


def clean_date(date_raw):
    date_raw = date_raw.replace("сарын", "").strip()
    if len(date_raw.split()) == 1:
        date_raw = f"{date_raw} 1"
    try:
        month, day = map(int, date_raw.split())
        current_year = 2024
        return f"{current_year}-{month:02d}-{day:02d}"
    except ValueError:
        return "Invalid date"

@st.cache_data
def load_data():
    df = pd.read_csv("listings.csv")
    return df

def preprocess_data(df):
    df['Salary'] = df['Salary'].str.replace('[₮,]', '', regex=True).str.strip().astype(int)
    df = df[df['Salary'] <= 100_000_000]
    df = df[df['Salary'] >= 1_000_000]
    return df

st.markdown("<h1>Mongolian Job Salary Analysis 💼 </h1>", unsafe_allow_html=True)
st.markdown("""
    This app allows you to analyze job salary data for various positions in Mongolia. 
    You can explore salary insights, visualize salary distributions, predict salaries for specific job titles, 
    and compare job positions based on salary data.
""", unsafe_allow_html=True)

job_title_filter = st.text_input("Enter Job Title (or part of it):", "")

df = load_data()
df = preprocess_data(df)

filtered_df = df[df['Job Title'].str.contains(job_title_filter, case=False, na=False)]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "📈 Insights", "🤖 Prediction", "🔗 Recommendations", "📋 Comparisons"])

# Tab 1 - Displaying Filtered Data
with tab1:
    st.subheader("Filtered Data", anchor="data")
    with st.expander("Description of Data Tab"):
        st.markdown("""
            In this tab, you can see a table of job listings that match the job title you entered. 
            The data includes details such as job title, company, salary, and category.
        """, unsafe_allow_html=True)
    st.write(f"Displaying **{len(filtered_df)}** results for jobs matching '{job_title_filter}'")
    st.dataframe(filtered_df, use_container_width=True)

# Tab 2 - Salary Insights
with tab2:
    if not filtered_df.empty:
        st.subheader("Salary Insights")
        with st.expander("Description of Insights Tab"):
            st.markdown("""
                This tab provides insights into the salary data, including average salaries, 
                top-paying companies, and the distribution of salaries across different job categories.
            """, unsafe_allow_html=True)
        avg_salary = filtered_df['Salary'].mean()
        st.write(f"**Average Salary:** {avg_salary:,.0f} ₮")
        st.write(f"**Max Salary:** {filtered_df['Salary'].max():,.0f} ₮")
        st.write(f"**Min Salary:** {filtered_df['Salary'].min():,.0f} ₮")

        top_company = filtered_df.groupby('Company')['Salary'].mean().idxmax()
        st.write(f"**Top Paying Company:** {top_company}")

        st.write("### Category Average Salaries")
        selected_categories = filtered_df['Category'].unique()
        for category in selected_categories:
            category_avg = df[df['Category'] == category]['Salary'].mean()
            st.write(f"- **{category}:** {category_avg:,.0f} ₮")

        # Visualizations using Streamlit's charting functions
        st.write("### Salary Distribution & Job Title vs Salary")
        
        col1, col2 = st.columns(2)

        with col1:
            # Salary Distribution (using Streamlit's st.bar_chart)
            salary_data = filtered_df['Salary']
            salary_distribution = salary_data.value_counts().sort_index()  # Create a distribution
            st.bar_chart(salary_distribution)  # Display the bar chart

        with col2:
            # Salary by Job Title (using Streamlit's st.bar_chart)
            salary_by_job = filtered_df.groupby('Job Title')['Salary'].mean().sort_values()
            st.bar_chart(salary_by_job)  # Display the bar chart


# Tab 3 - Salary Prediction with XGBoost
with tab3:
    if not filtered_df.empty:
        st.subheader("Salary Prediction with XGBoost")
        with st.expander("Description of Prediction Tab"):
            st.markdown("""
                In this tab, you can predict the salary for a specific job title using machine learning models like XGBoost. 
                You can select a job title, and the app will predict the salary based on historical data.
            """, unsafe_allow_html=True)
    
        title = filtered_df['Job Title']
        selected_job = st.selectbox("Select a Job Title for Prediction", title)
    
        # Prepare data for prediction
        X = filtered_df.drop(columns=['Salary', 'Date'])
        y = filtered_df['Salary']
    
        # Handle categorical variables using one-hot encoding
        categorical_features = X.select_dtypes(include=['object']).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough')
    
        # Transform features
        X_transformed = preprocessor.fit_transform(X)
    
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.5, random_state=42)
    
        # Train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
    
        # Predict salary for the selected job
        selected_job_data = filtered_df[filtered_df['Job Title'] == selected_job].iloc[0:1].drop(columns=['Salary'])
        selected_job_transformed = preprocessor.transform(selected_job_data)
        predicted_salary = model.predict(selected_job_transformed)[0]
    
        st.write(f"The predicted salary for **'{selected_job}'** is: **{predicted_salary:,.0f} ₮**")

# Tab 4 - Job Recommendations
with tab4:
    if not filtered_df.empty:
        st.subheader("Job Recommendations")
        with st.expander("Description of Recommendations Tab"):
            st.markdown("""
                In this tab, you can get job recommendations based on your salary preferences. 
                Use the salary range slider to filter job listings within your desired salary range.
            """, unsafe_allow_html=True)
        
        salary_min = int(filtered_df['Salary'].min() / 10000) * 10000
        salary_max = int(filtered_df['Salary'].max() / 10000) * 10000
        
        salary_range = st.slider(
            "Filter by Salary Range",
            min_value=salary_min,
            max_value=salary_max,
            value=(salary_min, salary_max),
            step=100000
        )

        recommended_jobs = filtered_df[
            (filtered_df['Salary'] >= salary_range[0]) &
            (filtered_df['Salary'] <= salary_range[1])
        ]

        st.write("### Recommended Jobs")
        
        if not recommended_jobs.empty:
            recommended_jobs['Job Title'] = recommended_jobs.apply(
                lambda row: f'<a href="{row["Link"]}" target="_blank">{row["Job Title"]}</a>', axis=1
            )
            recommended_jobs['Salary'] = recommended_jobs['Salary'].apply(lambda x: f"{x:,}")
            display_df = recommended_jobs[['Job Title', 'Company', 'Salary']]
            
            st.markdown(
                display_df.to_html(escape=False, index=False), 
                unsafe_allow_html=True
            )
        else:
            st.write("No jobs found in the selected salary range.")

# Tab 5 - Comparisons
with tab5:
    st.subheader("Compare Two Job Searches")
    with st.expander("Description of Comparison tab"):
            st.markdown("""
                This tab helps you compare salaries between different job positions. 
            You can compare multiple job titles and see how their salaries differ.
             """, unsafe_allow_html=True)
    
    search_query1 = st.text_input("Search Query 1:", "")
    search_query2 = st.text_input("Search Query 2:", "")
    
    if search_query1 and search_query2:
        search_results1 = filtered_df[filtered_df['Job Title'].str.contains(search_query1, case=False, na=False)]
        search_results2 = filtered_df[filtered_df['Job Title'].str.contains(search_query2, case=False, na=False)]
        
        def compute_metrics(search_results, query):
            if not search_results.empty:
                avg_salary = search_results['Salary'].mean()
                min_salary = search_results['Salary'].min()
                max_salary = search_results['Salary'].max()
                job_count = len(search_results)
                return {
                    "Query": query,
                    "Average Salary": avg_salary,
                    "Minimum Salary": min_salary,
                    "Maximum Salary": max_salary,
                    "Job Count": job_count
                }
            else:
                return {
                    "Query": query,
                    "Average Salary": 0,
                    "Minimum Salary": 0,
                    "Maximum Salary": 0,
                    "Job Count": 0
                }

        metrics1 = compute_metrics(search_results1, search_query1)
        metrics2 = compute_metrics(search_results2, search_query2)

        comparison_df = pd.DataFrame([metrics1, metrics2])

        st.write("### Comparison Results")
        st.dataframe(comparison_df, use_container_width=True)

        st.write("### Comparison Visualizations")

        # Creating columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for salary comparison
            salary_data = comparison_df[["Query", "Average Salary", "Minimum Salary", "Maximum Salary"]].set_index("Query")
            st.bar_chart(salary_data, stack= False)

        with col2:
            # Line chart for job count comparison
            job_count_data = comparison_df[["Query", "Job Count"]].set_index("Query")
            st.bar_chart(job_count_data)
    
    else:
        st.write("Please enter both search queries to compare.")

# Benchmarking
if not filtered_df.empty:
    st.write("### Salary Benchmark")
    industry_avg_salary = df['Salary'].mean()
    difference = avg_salary - industry_avg_salary

    comparison = "higher" if difference > 0 else "lower"
    st.write(f"Compared to the industry average, this job's salary is **{comparison}** by **{abs(difference):,.0f} ₮**.")
