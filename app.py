
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Social Media & Mental Health Analysis",
    page_icon="📱",
    layout="wide"
)

# Create sample data based on the images
def create_sample_data():
    # Based on the images, creating a comprehensive dataset
    data = {
        'timestamp': ['4/18/2022 19:18:47', '4/18/2022 19:19:28'] + ['4/18/2022'] * 5,
        'age': [21.0, 21.0] + [22.0] * 5,
        'gender': ['Male', 'Female'] + ['Female', 'Male', 'Female', 'Male', 'Female'],
        'relationship_status': ['In a relationship', 'Single'] + ['Single', 'In a relationship', 'Single', 'Single', 'In a relationship'],
        'occupation_status': ['University Student', 'University Student'] + ['University Student'] * 5,
        'affiliated_organizations': ['University', 'University'] + ['University'] * 5,
        'use_social_media': ['Yes', 'Yes'] + ['Yes'] * 5,
        'social_media_platforms': [
            'Facebook, Twitter, Instagram, YouTube, Discord...',
            'Facebook, Twitter, Instagram, YouTube, Discord...'
        ] + ['Facebook, Instagram, TikTok'] * 5,
        'daily_social_media_time': ['Between 2 and 3 hours', 'More than 5 hours'] + ['More than 5 hours', 'Between 3 and 4 hours', 'Between 2 and 3 hours', 'Between 4 and 5 hours', 'More than 5 hours'],
        'distractibility_scale': [5, 4, 2, 3, 4, 5, 4],
        'worry_level_scale': [2, 5, 5, 5, 5, 2, 5],
        'difficulty_concentrating': [5, 4, 4, 3, 5, 4, 3],
        'compare_to_successful_people_scale': [2, 5, 3, 5, 3, 4, 5],
        'feelings_about_comparisons': [3, 1, 3, 1, 3, 3, 2],
        'frequency_seeking_validation': [2, 1, 1, 2, 3, 4, 2],
        'frequency_feeling_depressed': [5, 5, 4, 4, 4, 5, 5],
        'interest_fluctuation_scale': [4, 4, 2, 3, 4, 3, 5],
        'sleep_issues_scale': [5, 5, 5, 2, 1, 4, 5],
        'cluster': [1, 1, 0, 0, 1, 0, 1]
    }
    
    return pd.DataFrame(data)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/users/HP/Desktop/FINAL/Socialmedia-_impact_clustering/smmh.csv")
    # In a real app, you might load from a file
    # df = pd.read_csv('social_media_mental_health.csv')
    
    # Using our sample data for demonstration
    df = create_sample_data()
    return df

df = load_data()

# Main app header
st.title("📱 Social Media Usage & Mental Health Analysis")
st.markdown("""
This dashboard analyzes the relationship between social media usage and various mental health metrics.
The data includes demographics, social media habits, and psychological well-being indicators.
""")

# Sidebar for filtering
st.sidebar.header("Filters")

# Add filters
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df['gender'].unique(),
    default=df['gender'].unique()
)

relationship_filter = st.sidebar.multiselect(
    "Select Relationship Status",
    options=df['relationship_status'].unique(),
    default=df['relationship_status'].unique()
)

social_media_time_filter = st.sidebar.multiselect(
    "Daily Social Media Usage",
    options=df['daily_social_media_time'].unique(),
    default=df['daily_social_media_time'].unique()
)

cluster_filter = st.sidebar.multiselect(
    "Cluster Group",
    options=df['cluster'].unique(),
    default=df['cluster'].unique()
)

# Apply filters
filtered_df = df[
    df['gender'].isin(gender_filter) &
    df['relationship_status'].isin(relationship_filter) &
    df['daily_social_media_time'].isin(social_media_time_filter) &
    df['cluster'].isin(cluster_filter)
]

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Mental Health Metrics", "Cluster Analysis", "Raw Data"])

with tab1:
    st.header("Dataset Overview")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Participants", len(filtered_df))
    with col2:
        high_usage = filtered_df[filtered_df['daily_social_media_time'] == 'More than 5 hours'].shape[0]
        st.metric("Heavy Social Media Users", f"{high_usage} ({high_usage/len(filtered_df)*100:.1f}%)")
    with col3:
        avg_worry = filtered_df['worry_level_scale'].mean()
        st.metric("Average Worry Level", f"{avg_worry:.2f}/5")
    
    # Gender distribution
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        gender_counts = filtered_df['gender'].value_counts()
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Gender Distribution')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        relationship_counts = filtered_df['relationship_status'].value_counts()
        ax.bar(relationship_counts.index, relationship_counts.values)
        plt.title('Relationship Status')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Social media usage
    st.subheader("Social Media Usage")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        time_counts = filtered_df['daily_social_media_time'].value_counts().sort_index()
        ax.bar(time_counts.index, time_counts.values)
        plt.title('Daily Social Media Usage Time')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Create a column for each platform and count
        all_platforms = []
        for platforms in filtered_df['social_media_platforms']:
            platforms_list = [p.strip() for p in platforms.split(',')]
            all_platforms.extend(platforms_list)
        platform_counts = pd.Series(all_platforms).value_counts().head(5)
        
        fig, ax = plt.subplots()
        ax.bar(platform_counts.index, platform_counts.values)
        plt.title('Top Social Media Platforms')
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab2:
    st.header("Mental Health Metrics Analysis")
    
    # Select metrics to compare
    mental_health_cols = [
        'distractibility_scale', 'worry_level_scale', 'difficulty_concentrating',
        'compare_to_successful_people_scale', 'feelings_about_comparisons',
        'frequency_seeking_validation', 'frequency_feeling_depressed',
        'interest_fluctuation_scale', 'sleep_issues_scale'
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        x_metric = st.selectbox("X-axis Metric", mental_health_cols, index=0)
    with col2:
        y_metric = st.selectbox("Y-axis Metric", mental_health_cols, index=1)
    
    # Scatter plot of selected metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x=x_metric, y=y_metric, hue='gender', style='cluster', s=100)
    plt.title(f'{x_metric} vs {y_metric}')
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    corr = filtered_df[mental_health_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Between Mental Health Metrics')
    st.pyplot(fig)
    
    # Average metrics by social media usage time
    st.subheader("Mental Health Metrics by Social Media Usage")
    
    # Create ordered categories
    usage_order = [
        'Less than 1 hour',
        'Between 1 and 2 hours',
        'Between 2 and 3 hours',
        'Between 3 and 4 hours',
        'Between 4 and 5 hours',
        'More than 5 hours'
    ]
    
    # Filter for only usage categories that exist in data
    existing_usage = [u for u in usage_order if u in filtered_df['daily_social_media_time'].unique()]
    
    # Group by social media usage time and calculate mean for mental health metrics
    grouped_df = filtered_df.groupby('daily_social_media_time')[mental_health_cols].mean().reindex(existing_usage)
    
    # Plot
    selected_metric = st.selectbox("Select Metric to Analyze", mental_health_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_df[selected_metric].plot(kind='bar', ax=ax)
    plt.title(f'Average {selected_metric} by Social Media Usage Time')
    plt.ylabel('Average Score (1-5)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab3:
    st.header("Cluster Analysis")
    
    # Description of clusters
    st.write("""
    The dataset contains cluster assignments that group participants based on their patterns of social media usage and mental health indicators.
    
    - **Cluster 0**: Users with moderate social media usage and better mental health indicators
    - **Cluster 1**: Users with heavy social media usage and more concerning mental health indicators
    """)
    
    # Compare clusters
    cluster_means = filtered_df.groupby('cluster')[mental_health_cols].mean()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cluster_means.T.plot(kind='bar', ax=ax)
    plt.title('Mental Health Metrics by Cluster')
    plt.ylabel('Average Score (1-5)')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster')
    st.pyplot(fig)
    
    # Radar chart for clusters
    st.subheader("Cluster Profiles (Radar Chart)")
    
    # Prepare data for radar chart
    categories = mental_health_cols
    
    # Function to create radar chart
    def create_radar_chart(cluster_means):
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for cluster in cluster_means.index:
            values = cluster_means.loc[cluster].tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
            ax.fill(angles, values, alpha=0.1)
        
        plt.xticks(angles[:-1], categories, size=10)
        plt.yticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'], color='grey', size=8)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        return fig
    
    radar_fig = create_radar_chart(cluster_means)
    st.pyplot(radar_fig)
    
    # Social media usage by cluster
    st.subheader("Social Media Usage by Cluster")
    
    # Count social media usage time by cluster
    usage_by_cluster = pd.crosstab(filtered_df['cluster'], filtered_df['daily_social_media_time'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    usage_by_cluster.plot(kind='bar', stacked=True, ax=ax)
    plt.title('Social Media Usage Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Daily Usage Time')
    st.pyplot(fig)

with tab4:
    st.header("Raw Data")
    st.write("This is the filtered dataset based on your selections:")
    st.dataframe(filtered_df)
    
    # Allow download of filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="social_media_mental_health_filtered.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown("Social Media & Mental Health Dashboard | Created with Streamlit")