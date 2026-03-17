import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

model = joblib.load("backend/best_model.pkl")
scaler = joblib.load("backend/scaler.pkl")

st.set_page_config(page_title="Cybersecurity Threat Detection", layout="wide")
st.title("🚨 Cybersecurity Threat Detection Dashboard")
st.markdown("Upload a CSV file to detect network attacks and explore interactive visualizations.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:

    data = pd.read_csv(uploaded_file)
    
    if 'status' in data.columns:
        data = data.drop('status', axis=1)

    cat_cols = data.select_dtypes(include='object').columns
    for col in cat_cols:
        data[col] = data[col].astype('category').cat.codes

    data_scaled = scaler.transform(data)
    
    predictions = model.predict(data_scaled)
    data['Predicted_Status'] = ["Normal" if x==0 else "Attack" for x in predictions]

    tabs = st.tabs(["Summary & KPIs", "Visualizations", "Detailed Table"])
    
    with tabs[0]:
        st.subheader("Summary Metrics")
        total = len(data)
        attacks = sum(data['Predicted_Status']=="Attack")
        normal = total - attacks

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Normal Records", normal, delta_color="normal")
        col3.metric("Attacks Detected", attacks, delta_color="inverse")

        st.subheader("Top 5 Most Suspicious Records")
        if attacks > 0:
            top_attack = data[data['Predicted_Status']=="Attack"]
            top_attack = top_attack.assign(Total_Bytes=top_attack['src_bytes'] + top_attack['dst_bytes'])
            top_attack = top_attack.sort_values(by='Total_Bytes', ascending=False).head(5)
            st.dataframe(top_attack)
        else:
            st.info("No attacks detected in the uploaded data.")

    with tabs[1]:
        st.subheader("Interactive Charts")
        
        st.sidebar.header("Filters for Visualizations")
        protocol_filter = st.sidebar.multiselect(
            "Filter by Protocol", options=data['protocol_type'].unique(), default=data['protocol_type'].unique()
        )
        service_filter = st.sidebar.multiselect(
            "Filter by Service", options=data['service'].unique(), default=data['service'].unique()
        )
        status_filter = st.sidebar.multiselect(
            "Filter by Status", options=data['Predicted_Status'].unique(), default=data['Predicted_Status'].unique()
        )

        filtered_data = data[
            (data['protocol_type'].isin(protocol_filter)) &
            (data['service'].isin(service_filter)) &
            (data['Predicted_Status'].isin(status_filter))
        ]

        status_counts = filtered_data['Predicted_Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        fig_bar = px.bar(
            status_counts, x='Status', y='Count', color='Status',
            color_discrete_map={'Normal':'green','Attack':'red'},
            title="Attack vs Normal Records",
            text='Count'
        )
        fig_bar.update_layout(height=400, width=500)
        
        fig_pie = px.pie(
            status_counts, names='Status', values='Count',
            color='Status', color_discrete_map={'Normal':'green','Attack':'red'},
            title="Proportion of Normal vs Attack"
        )
        fig_pie.update_layout(height=400, width=500)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Cumulative Attack Trend")
        trend_df = filtered_data.reset_index().copy()
        trend_df['Cumulative_Attacks'] = (trend_df['Predicted_Status']=="Attack").cumsum()
        fig_trend = px.line(trend_df, x='index', y='Cumulative_Attacks', title="Cumulative Attacks Over Records")
        st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[2]:
        st.subheader("Detailed Predictions Table")
        st.dataframe(filtered_data)

        st.download_button(
            "Download Predictions CSV",
            data=filtered_data.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to start predictions.")