import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("job_listings_big.csv")
    return df

def preprocess_data(df):
    df['Salary'] = df['Salary'].str.replace('[₮,]', '', regex=True).str.strip().astype(int)
    return df

# App Title
st.markdown("<h1>Mongolian Job Salary Analysis </h1>", unsafe_allow_html=True)

# Sidebar for Search
st.sidebar.title("🔍 Filter Jobs")
job_title_filter = st.sidebar.text_input("Enter Job Title (or part of it):")

# Load and Process Data
df = load_data()
df = preprocess_data(df)

# Filter Based on Job Title
filtered_df = df[df['Job Title'].str.contains(job_title_filter, case=False, na=False)]

# Tabs for Organized Navigation
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Insights", "🤖 Prediction", "🔗 Recommendations"])

with tab1:
    # Show Filtered Data
    st.subheader("Filtered Data")
    st.write(f"Displaying **{len(filtered_df)}** results for jobs matching '{job_title_filter}'")
    st.dataframe(filtered_df)

with tab2:
    # Salary Insights
    if not filtered_df.empty:
        st.subheader("Salary Insights")
        avg_salary = filtered_df['Salary'].mean()
        st.write(f"**Average Salary:** {avg_salary:,.0f} ₮")
        st.write(f"**Max Salary:** {filtered_df['Salary'].max():,.0f} ₮")
        st.write(f"**Min Salary:** {filtered_df['Salary'].min():,.0f} ₮")

        top_company = filtered_df.groupby('Company')['Salary'].mean().idxmax()
        st.write(f"**Top Paying Company:** {top_company}")

        # Category Average Salary
        st.write("### Category Average Salaries")
        selected_categories = filtered_df['Category'].unique()
        for category in selected_categories:
            category_avg = df[df['Category'] == category]['Salary'].mean()
            st.write(f"- **{category}:** {category_avg:,.0f} ₮")

        # Visualizations
        st.write("### Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['Salary'], bins=10, kde=True, ax=ax)
        st.pyplot(fig)

        st.write("### Salary by Job Title")
        fig, ax = plt.subplots()
        filtered_df.groupby('Job Title')['Salary'].mean().sort_values().plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab3:
    # Salary Prediction
    if not filtered_df.empty:
        st.subheader("Salary Prediction")

        unique_titles = filtered_df['Job Title'].unique()
        selected_job = st.selectbox("Select a Job Title for Prediction", unique_titles)
        filtered_df['Job Title Length'] = filtered_df['Job Title'].apply(len)  # Example feature
        X = filtered_df[['Job Title Length']]
        y = filtered_df['Salary']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        job_title_length = len(selected_job)
        predicted_salary = model.predict([[job_title_length]])[0]

        st.write(f"**Predicted Salary for '{selected_job}':** {predicted_salary:,.0f} ₮")

with tab4:
    # Job Recommendations
    if not filtered_df.empty:
        st.subheader("Job Recommendations")
        salary_range = st.slider("Filter by Salary Range",
                                 int(filtered_df['Salary'].min()),
                                 int(filtered_df['Salary'].max()),
                                 (int(filtered_df['Salary'].min()), int(filtered_df['Salary'].max())))

        recommended_jobs = filtered_df[(filtered_df['Salary'] >= salary_range[0]) &
                                       (filtered_df['Salary'] <= salary_range[1])]
        st.write("### Recommended Jobs")
        st.dataframe(recommended_jobs[['Job Title', 'Company', 'Salary']])

# Benchmarking
if not filtered_df.empty:
    st.write("### Salary Benchmarking")
    industry_avg_salary = df['Salary'].mean()
    difference = avg_salary - industry_avg_salary

    comparison = "higher" if difference > 0 else "lower"
    st.write(f"Compared to the industry average, this job's salary is **{comparison}** by **{abs(difference):,.0f} ₮**.")
