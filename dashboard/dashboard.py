import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Set styling for seaborn and streamlit options
sns.set_theme(style='dark')

 # Define your Streamlit app
st.title("Data Analysis on Brazilian E-Commerce Dataset")

# Load the datasets
df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\customers_dataset.csv') 
geo_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\geolocation_dataset.csv')  
payment_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\order_payments_dataset.csv')  
order_reviews_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\order_reviews_dataset.csv')  
orders_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\orders_dataset.csv')  
product_category_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\orders_dataset.csv')  
product_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\products_dataset.csv')  
sellers_df = pd.read_csv('D:\Sampoerna University\Boothcamp\Bangkit_Academy_Machine_Learning\Submission_DicodingxBangkit_Data_Analytics\Data\sellers_dataset.csv')  

# Customer count by city
customers_by_city = df.groupby('customer_city')['customer_id'].count().reset_index()
customers_by_city = customers_by_city.rename(columns={'customer_id': 'customer_count'})
top_cities = customers_by_city.sort_values(by='customer_count', ascending=False).head(10)

# Customer count by state
customers_by_state = df.groupby('customer_state')['customer_id'].count().reset_index()
customers_by_state = customers_by_state.rename(columns={'customer_id': 'customer_count'})
top_states = customers_by_state.sort_values(by='customer_count', ascending=False)

# Visualization: Top Cities by Customer Count
st.subheader("Top 10 Cities by Customer Count")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_cities, x='customer_city', y='customer_count', hue='customer_city', palette='viridis', ax=ax1, dodge=False)
ax1.set_title('Top 10 Cities by Customer Count', fontsize=16)
ax1.set_xlabel('City', fontsize=12)
ax1.set_ylabel('Customer Count', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Visualization: Customer Distribution by State
st.subheader("Customer Distribution by State")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_states, x='customer_state', y='customer_count', hue='customer_state', palette='coolwarm', ax=ax2, dodge=False)
ax2.set_title('Customer Distribution by State', fontsize=16)
ax2.set_xlabel('State', fontsize=12)
ax2.set_ylabel('Customer Count', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Count occurrences of each unique customer ID
customer_repeat = df.groupby('customer_unique_id')['customer_id'].count().reset_index()
customer_repeat = customer_repeat.rename(columns={'customer_id': 'purchase_count'})

# Identify repeat customers
repeat_customers = customer_repeat[customer_repeat['purchase_count'] > 1]

# Percentage of repeat customers
total_customers = len(customer_repeat)
repeat_customer_percentage = (len(repeat_customers) / total_customers) * 100

# Data for visualization: Repeat vs Non-Repeat Customers
repeat_data = pd.DataFrame({
    'Category': ['Repeat Customers', 'Non-Repeat Customers'],
    'Count': [len(repeat_customers), total_customers - len(repeat_customers)]
})

# Visualization: Repeat vs Non-Repeat Customers
st.subheader("Percentage of Repeated Customers")
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.pie(repeat_data['Count'], labels=repeat_data['Category'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107'])
ax3.set_title('Percentage of Repeated Customers', fontsize=16)
st.pyplot(fig3)

# Repeat Customers by State
repeat_by_state = df[df['customer_unique_id'].isin(repeat_customers['customer_unique_id'])]
repeat_by_state = repeat_by_state.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
repeat_by_state = repeat_by_state.rename(columns={'customer_unique_id': 'repeat_customers'})

# Visualization: Repeat Customers by State
st.subheader("Repeat Customers by State")
fig4, ax4 = plt.subplots(figsize=(12, 6))
sns.barplot(data=repeat_by_state.sort_values(by='repeat_customers', ascending=False),
            x='customer_state', y='repeat_customers', hue='customer_state', palette='Blues_d', ax=ax4, dodge=False)
ax4.set_title('Repeated Customers by State', fontsize=16)
ax4.set_xlabel('State', fontsize=12)
ax4.set_ylabel('Repeat Customers', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig4)

# Data cleaning for geolocation dataset
geo_df.dropna(inplace=True)
geo_df.columns = geo_df.columns.str.lower().str.replace(" ", "_")

# Selecting relevant columns for clustering
geo_data = geo_df[['geolocation_lat', 'geolocation_lng']]

# Clustering using KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
geo_df['cluster'] = kmeans.fit_predict(geo_data)

# Visualizing clusters
st.subheader("Customer Clustering Based on Geolocation")
fig5, ax5 = plt.subplots(figsize=(12, 8))
for cluster in range(5):
    cluster_data = geo_df[geo_df['cluster'] == cluster]
    ax5.scatter(cluster_data['geolocation_lng'], cluster_data['geolocation_lat'], label=f'Cluster {cluster}', alpha=0.6)

ax5.set_title('Customer Clustering Based on Geolocation', fontsize=16)
ax5.set_xlabel('Longitude', fontsize=12)
ax5.set_ylabel('Latitude', fontsize=12)
ax5.legend()
st.pyplot(fig5)

# Analysis of cluster sizes
cluster_sizes = geo_df['cluster'].value_counts().reset_index()
cluster_sizes.columns = ['cluster', 'customer_count']

# Visualization: Cluster size comparison
st.subheader("Cluster Size Comparison")
fig6, ax6 = plt.subplots(figsize=(10, 6))
sns.barplot(data=cluster_sizes, x='cluster', y='customer_count', hue='cluster', palette='coolwarm', ax=ax6, dodge=False)
ax6.set_title('Cluster Size Comparison', fontsize=16)
ax6.set_xlabel('Cluster', fontsize=12)
ax6.set_ylabel('Number of Customers', fontsize=12)
st.pyplot(fig6)

# Clean and process the payment dataset
payment_df.dropna(inplace=True)
payment_df['payment_value'] = payment_df['payment_value'].astype(float)
payment_df['payment_installments'] = payment_df['payment_installments'].astype(int)

# Calculate average payment value by payment type
avg_payment_by_type = payment_df.groupby('payment_type')['payment_value'].mean().reset_index()

# Calculate average number of installments by payment type
avg_installments_by_type = payment_df.groupby('payment_type')['payment_installments'].mean().reset_index()

# Visualization: Payment Value by Payment Type
st.subheader("Average Payment Value by Payment Type")
fig7, ax7 = plt.subplots(figsize=(12, 6))
sns.barplot(data=avg_payment_by_type, x='payment_type', y='payment_value', hue='payment_type', palette='viridis', ax=ax7, dodge=False)
ax7.set_title('Average Payment Value by Payment Type', fontsize=16)
ax7.set_xlabel('Payment Type', fontsize=12)
ax7.set_ylabel('Average Payment Value', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig7)

# Visualization: Average Installments by Payment Type
st.subheader("Average Number of Installments by Payment Type")
fig8, ax8 = plt.subplots(figsize=(12, 6))
sns.barplot(data=avg_installments_by_type, x='payment_type', y='payment_installments', hue='payment_type', palette='coolwarm', ax=ax8, dodge=False)
ax8.set_title('Average Number of Installments by Payment Type', fontsize=16)
ax8.set_xlabel('Payment Type', fontsize=12)
ax8.set_ylabel('Average Number of Installments', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig8)

# Scatter plot of payment value vs installments to visualize the relationship
st.subheader("Payment Value vs. Payment Installments by Payment Type")
fig9, ax9 = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=payment_df, x='payment_installments', y='payment_value', hue='payment_type', palette='Set2', alpha=0.6, ax=ax9)
ax9.set_title('Payment Value vs. Payment Installments by Payment Type', fontsize=16)
ax9.set_xlabel('Number of Installments', fontsize=12)
ax9.set_ylabel('Payment Value', fontsize=12)
st.pyplot(fig9)

# Boxplot to show the distribution of payment value across different installments
st.subheader("Distribution of Payment Value by Number of Installments")
fig10, ax10 = plt.subplots(figsize=(12, 6))
sns.boxplot(data=payment_df, x='payment_installments', y='payment_value', hue='payment_installments', palette='coolwarm', ax=ax10, dodge=False)
ax10.set_title('Distribution of Payment Value by Number of Installments', fontsize=16)
ax10.set_xlabel('Number of Installments', fontsize=12)
ax10.set_ylabel('Payment Value', fontsize=12)
st.pyplot(fig10)

# Clean the review_comment_message
order_reviews_df['review_comment_message'] = order_reviews_df['review_comment_message'].fillna('')
order_reviews_df['review_comment_message'] = order_reviews_df['review_comment_message'].str.lower()

# Generate wordcloud from review comments
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(' '.join(order_reviews_df['review_comment_message']))

st.subheader("Most Frequent Themes in Customer Reviews")

# Display the wordcloud in Streamlit
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Themes in Customer Reviews', fontsize=16)

# Instead of plt.show(), use st.pyplot
st.pyplot(plt)

# Use CountVectorizer to identify the top n-grams in reviews
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20)
X = vectorizer.fit_transform(order_reviews_df['review_comment_message'])

# Get the most common n-grams
n_grams = vectorizer.get_feature_names_out()
n_gram_freq = X.toarray().sum(axis=0)

# Convert to DataFrame for easier visualization
ngram_df = pd.DataFrame({'ngram': n_grams, 'frequency': n_gram_freq})
ngram_df = ngram_df.sort_values(by='frequency', ascending=False).head(10)

st.subheader("Top 10 Common Themes (N-Grams) in Customer Reviews")

# Visualize top n-grams
plt.figure(figsize=(12, 6))
sns.barplot(x='frequency', y='ngram', data=ngram_df, hue='ngram', palette='viridis', legend=False)
plt.title('Top 10 Common Themes (N-Grams) in Customer Reviews', fontsize=16)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('N-Gram', fontsize=12)

# Instead of plt.show(), use st.pyplot
st.pyplot(plt)

# Load dataset (assuming it's already loaded as 'orders_df')
orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
orders_df['order_delivered_carrier_date'] = pd.to_datetime(orders_df['order_delivered_carrier_date'])
orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])

# Calculate delivery time and delay (difference in days)
# Convert columns to datetime
orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
orders_df['order_delivered_carrier_date'] = pd.to_datetime(orders_df['order_delivered_carrier_date'])
orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])

    # Calculate delivery time and delay (difference in days)
orders_df['delivery_time'] = (orders_df['order_delivered_customer_date'] - orders_df['order_delivered_carrier_date']).dt.days
orders_df['delivery_delay'] = (orders_df['order_delivered_customer_date'] - orders_df['order_estimated_delivery_date']).dt.days

    # Distribution of delivery times
st.subheader("Distribution of Delivery Time")
fig1, ax1 = plt.subplots(figsize=(14, 7))
sns.histplot(orders_df['delivery_time'], kde=True, color='skyblue', bins=30, ax=ax1)
ax1.set_title('Distribution of Delivery Time', fontsize=16)
ax1.set_xlabel('Delivery Time (days)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
st.pyplot(fig1)

    # Delivery delay by order status
st.subheader("Delivery Delay by Order Status")
fig2, ax2 = plt.subplots(figsize=(14, 7))
sns.boxplot(x='order_status', y='delivery_delay', data=orders_df, ax=ax2)
ax2.set_title('Delivery Delay by Order Status', fontsize=16)
ax2.set_xlabel('Order Status', fontsize=12)
ax2.set_ylabel('Delivery Delay (days)', fontsize=12)
st.pyplot(fig2)

    # Delivery time by purchase month
orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month
st.subheader("Delivery Time by Purchase Month")
fig3, ax3 = plt.subplots(figsize=(14, 7))
sns.boxplot(x='purchase_month', y='delivery_time', data=orders_df, ax=ax3)
ax3.set_title('Delivery Time by Purchase Month', fontsize=16)
ax3.set_xlabel('Purchase Month', fontsize=12)
ax3.set_ylabel('Delivery Time (days)', fontsize=12)
st.pyplot(fig3)

    # Correlation between delivery time/delay
st.subheader("Correlation between Delivery Time and Delay")
corr_matrix = orders_df[['delivery_time', 'delivery_delay']].corr()
fig4, ax4 = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax4)
ax4.set_title('Correlation between Delivery Time and Delay', fontsize=16)
st.pyplot(fig4)

    # Distribution of delivery delays
st.subheader("Distribution of Delivery Delay")
fig5, ax5 = plt.subplots(figsize=(14, 7))
sns.histplot(orders_df['delivery_delay'], kde=True, color='salmon', bins=30, ax=ax5)
ax5.set_title('Distribution of Delivery Delay', fontsize=16)
ax5.set_xlabel('Delivery Delay (days)', fontsize=12)
ax5.set_ylabel('Frequency', fontsize=12)
st.pyplot(fig5)

# Ensure purchase hour, day of week, and month are extracted correctly
orders_df['purchase_hour'] = orders_df['order_purchase_timestamp'].dt.hour
orders_df['purchase_day_of_week'] = orders_df['order_purchase_timestamp'].dt.dayofweek
orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month

# Visualize the purchase patterns by hour of the day
st.subheader('Number of Orders by Hour of the Day')
fig1, ax1 = plt.subplots(figsize=(14, 7))
sns.countplot(x='purchase_hour', data=orders_df, color='skyblue', ax=ax1)
ax1.set_title('Number of Orders by Hour of the Day', fontsize=16)
ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig1)

# Visualize the purchase patterns by day of the week
st.subheader('Number of Orders by Day of the Week')
fig2, ax2 = plt.subplots(figsize=(14, 7))
sns.countplot(x='purchase_day_of_week', data=orders_df, color='lightcoral', ax=ax2)
ax2.set_title('Number of Orders by Day of the Week', fontsize=16)
ax2.set_xlabel('Day of the Week (0=Monday, 6=Sunday)', fontsize=12)
ax2.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig2)

# Visualize the purchase patterns by month
st.subheader('Number of Orders by Month')
fig3, ax3 = plt.subplots(figsize=(14, 7))
sns.countplot(x='purchase_month', data=orders_df, color='lightgreen', ax=ax3)
ax3.set_title('Number of Orders by Month', fontsize=16)
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig3)

# Analyze the relationship between order purchase time and customer purchase behavior
st.subheader('Delivery Time by Purchase Month')
fig4, ax4 = plt.subplots(figsize=(14, 7))
sns.boxplot(x='purchase_month', y='delivery_time', data=orders_df, color='lightblue', ax=ax4)
ax4.set_title('Delivery Time by Purchase Month', fontsize=16)
ax4.set_xlabel('Purchase Month', fontsize=12)
ax4.set_ylabel('Delivery Time (days)', fontsize=12)
st.pyplot(fig4)

# Visualize the seasonal trends in product demand (using order counts per month)
st.subheader('Monthly Order Volume')
monthly_order_count = orders_df.groupby('purchase_month').size()
fig5, ax5 = plt.subplots(figsize=(14, 7))
monthly_order_count.plot(kind='bar', color='teal', ax=ax5)
ax5.set_title('Monthly Order Volume', fontsize=16)
ax5.set_xlabel('Month', fontsize=12)
ax5.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig5)

# Analyze the number of sellers per state to understand regional distribution
seller_distribution_state = sellers_df['seller_state'].value_counts().reset_index()
seller_distribution_state.columns = ['State', 'Seller Count']

st.subheader('Seller Distribution by State')

# Visualizing the distribution of sellers by state
plt.figure(figsize=(14, 7))
sns.barplot(data=seller_distribution_state, x='State', y='Seller Count', hue='State', legend=False, palette="viridis")
plt.title('Seller Distribution by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Sellers', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Analyze the relationship between zip code prefix and seller location
sellers_df['zip_prefix'] = sellers_df['seller_zip_code_prefix'].astype(str).str[:2]
zip_prefix_distribution = sellers_df['zip_prefix'].value_counts().reset_index()
zip_prefix_distribution.columns = ['Zip Prefix', 'Seller Count']

st.subheader('Seller Distribution by Zip Code Prefix')

# Visualize the seller distribution by zip code prefix
plt.figure(figsize=(14, 7))
sns.barplot(data=zip_prefix_distribution, x='Zip Prefix', y='Seller Count', hue='Zip Prefix', legend=False, palette="coolwarm")
plt.title('Seller Distribution by Zip Code Prefix', fontsize=16)
plt.xlabel('Zip Code Prefix', fontsize=12)
plt.ylabel('Number of Sellers', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Regional shipping time analysis (Assumption: Seller data influences shipping time)
sellers_df['shipping_time'] = sellers_df['seller_zip_code_prefix'].apply(
    lambda x: 5 + int(x) % 10  # Hypothetical function to simulate shipping time
)

st.subheader('Shipping Time by State (Hypothetical)')

# Visualize shipping time by state
plt.figure(figsize=(14, 7))
sns.boxplot(data=sellers_df, x='seller_state', y='shipping_time', hue='seller_state', legend=False, palette="muted")
plt.title('Shipping Time by State (Hypothetical)', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Shipping Time (Days)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Calculate average shipping time by state
state_shipping_time = sellers_df.groupby('seller_state')['shipping_time'].mean().reset_index()
state_shipping_time.columns = ['State', 'Average Shipping Time']

st.subheader('Average Shipping Time by State (Hypothetical)')

# Visualize average shipping time by state
plt.figure(figsize=(14, 7))
sns.barplot(data=state_shipping_time, x='State', y='Average Shipping Time', hue='State', legend=False, palette="Blues_d")
plt.title('Average Shipping Time by State (Hypothetical)', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Average Shipping Time (Days)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Conclusion Section
st.subheader("Conclusion")
st.write(""" 

The analysis of the regional distribution of sellers and customers highlights the importance of geographic factors in optimizing marketing and logistics strategies. Sao Paulo (SP), Rio de Janeiro (RJ), and Minas Gerais (MG) emerge as the key regions for targeted marketing efforts due to their high customer concentrations. Tailored marketing campaigns, location-based promotions, and streamlined delivery logistics can significantly enhance business performance in these areas. Conversely, less populated regions like Acre, Roraima, Amapá, and Tocantins may require a more centralized delivery model, possibly through partnerships with neighboring states to improve operational efficiency.

Customer loyalty remains a challenge, with only '3.12%' of customers returning for multiple purchases. However, higher repeat customer rates in states like São Paulo and Rio de Janeiro suggest that targeted loyalty programs could be effective in retaining customers. Implementing region-specific loyalty incentives, discounts, and personalized offers can encourage repeat purchases, especially in areas with lower customer loyalty. Businesses can also leverage customer clusters to optimize delivery routes, focusing on high-density regions for frequent deliveries while reducing resources in less dense areas.

The payment analysis reveals a correlation between payment installments and purchase value, indicating that offering flexible installment plans for higher-value items can boost sales and cash flow. For customers preferring single payments, like those using boleto or debit cards, businesses could introduce bundled offers to encourage repeat purchases. Additionally, improving communication about delivery times and product expectations is crucial to manage customer satisfaction, especially with concerns around delivery delays and product quality.

Geographic factors, such as the concentration of sellers, play a critical role in product availability and delivery efficiency. High-density seller regions experience faster delivery times and better product availability, while remote areas face challenges that could impact customer satisfaction. By aligning marketing efforts and operational strategies with regional dynamics, businesses can optimize their marketing campaigns, enhance customer loyalty, and streamline logistics to improve overall performance.

""")

# Copyright Section
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
st.markdown("<p style='text-align: center; font-size: 12px;'>Copyright (C) Ady Syamsuri. 2024</p>", unsafe_allow_html=True)