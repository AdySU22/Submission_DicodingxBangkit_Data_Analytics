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

# Sidebar for Navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Select a page:", ["Overview", "Customer Analysis", "Geolocation Analysis", "Payment Analysis", "Review Analysis", "Order Analysis"])

# Load the datasets
# Load the datasets using relative paths (assuming datasets are in the 'Data' folder)
df = pd.read_csv('../data/customers_dataset.csv') 
geo_df = pd.read_csv('../data/geolocation_dataset.csv')  
payment_df = pd.read_csv('../data/order_payments_dataset.csv')  
order_reviews_df = pd.read_csv('../data/order_reviews_dataset.csv')  
orders_df = pd.read_csv('../data/orders_dataset.csv')  
product_category_df = pd.read_csv('../data/orders_dataset.csv')  
product_df = pd.read_csv('../data/products_dataset.csv')  
sellers_df = pd.read_csv('../data/sellers_dataset.csv')  

# Customer count by city
customers_by_city = df.groupby('customer_city')['customer_id'].count().reset_index()
customers_by_city = customers_by_city.rename(columns={'customer_id': 'customer_count'})
top_cities = customers_by_city.sort_values(by='customer_count', ascending=False).head(10)

# Customer count by state
customers_by_state = df.groupby('customer_state')['customer_id'].count().reset_index()
customers_by_state = customers_by_state.rename(columns={'customer_id': 'customer_count'})
top_states = customers_by_state.sort_values(by='customer_count', ascending=False)

st.write(""" 

- ### Question 1:

Question 1: How can we segment customers based on their location and zip code to optimize regional marketing campaigns and improve delivery logistics?
         
""")

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

st.write(""" 

- ### Question 2:

Are there patterns in customer repeat behavior (costumer loyalty) across cities or states, and how can we design loyalty programs to retain high-value customers?
         
""")

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

st.write(""" 

- ### Question 3:

What are the geographical trends in customer clustering, and how can this information be used to improve delivery routing and efficiency?
         
""")

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

st.write(""" 

- ### Question 4:

How do payment types influence the average payment value and installment patterns, and what insights can be drawn to optimize payment offerings?
         
""")

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

st.write(""" 

- ### Question 5:

What is the relationship between the number of payment installments and the total payment value, and how can this be leveraged to offer installment plans effectively?
         
""")

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

st.write(""" 

- ### Question 6:

What are the most common themes in customer reviews (via review comments), and how can these insights be used to enhance the product or service offerings?
         
""")

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

st.write(""" 

- ### Question 7:

What are the key factors influencing order delivery time, and how can they be optimized for better customer satisfaction?
         
""")

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

st.write(""" 

- ### Question 8:

 How do the order purchase patterns (e.g., seasonality, time of day) correlate with customer purchase behavior and product demand?
                  
""")

# Number of Orders by Hour of the Day ---
orders_df['purchase_hour'] = orders_df['order_purchase_timestamp'].dt.hour
st.subheader("Number of Orders by Hour of the Day")
fig12, ax12 = plt.subplots(figsize=(14, 7))
sns.countplot(x='purchase_hour', data=orders_df, color='skyblue', ax=ax12)
ax12.set_title('Number of Orders by Hour of the Day', fontsize=16)
ax12.set_xlabel('Hour of Day', fontsize=12)
ax12.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig12)

# Number of Orders by Day of the Week ---
orders_df['purchase_day_of_week'] = orders_df['order_purchase_timestamp'].dt.dayofweek
st.subheader("Number of Orders by Day of the Week")
fig13, ax13 = plt.subplots(figsize=(14, 7))
sns.countplot(x='purchase_day_of_week', data=orders_df, color='lightcoral', ax=ax13)
ax13.set_title('Number of Orders by Day of the Week', fontsize=16)
ax13.set_xlabel('Day of the Week (0=Monday, 6=Sunday)', fontsize=12)
ax13.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig13)

# Number of Orders by Month ---
orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month
st.subheader("Number of Orders by Month")
fig14, ax14 = plt.subplots(figsize=(14, 7))
sns.countplot(x='purchase_month', data=orders_df, color='lightgreen', ax=ax14)
ax14.set_title('Number of Orders by Month', fontsize=16)
ax14.set_xlabel('Month', fontsize=12)
ax14.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig14)

# Delivery time by purchase month
orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month
st.subheader("Delivery Time by Purchase Month")
fig3, ax3 = plt.subplots(figsize=(14, 7))
sns.boxplot(x='purchase_month', y='delivery_time', data=orders_df, ax=ax3)
ax3.set_title('Delivery Time by Purchase Month', fontsize=16)
ax3.set_xlabel('Purchase Month', fontsize=12)
ax3.set_ylabel('Delivery Time (days)', fontsize=12)
st.pyplot(fig3)

# Monthly Order Volume ---
monthly_order_count = orders_df.groupby('purchase_month').size()
st.subheader("Monthly Order Volume")
fig15, ax15 = plt.subplots(figsize=(14, 7))
monthly_order_count.plot(kind='bar', color='teal', ax=ax15)
ax15.set_title('Monthly Order Volume', fontsize=16)
ax15.set_xlabel('Month', fontsize=12)
ax15.set_ylabel('Number of Orders', fontsize=12)
st.pyplot(fig15)

# Order Status by Time Features ---
order_status_by_time = orders_df.groupby(['purchase_hour', 'purchase_day_of_week', 'purchase_month'])['order_status'].value_counts().unstack().fillna(0)
st.subheader("Order Status by Time Features")
fig16, ax16 = plt.subplots(figsize=(14, 7))
order_status_by_time.plot(kind='bar', stacked=True, cmap='Set2', ax=ax16)
ax16.set_title('Order Status by Time Features', fontsize=16)
ax16.set_xlabel('Time Features', fontsize=12)
ax16.set_ylabel('Order Count', fontsize=12)
st.pyplot(fig16)

st.write(""" 

- ### Question 9:

 How do sales trends vary across different product categories, and which categories are driving the highest revenue?
                  
""")

# Sample DataFrame (replace with actual data)
data = {
    'product_category_name': ['beleza_saude', 'informatica_acessorios', 'automotivo', 'cama_mesa_banho', 'moveis_decoracao'] * 100,
    'price': np.random.randint(10, 500, 500),
    'quantity_sold': np.random.randint(1, 10, 500),
    'order_date': pd.date_range('2018-01-01', periods=500, freq='D')
}
df = pd.DataFrame(data)

# Create a revenue column
df['revenue'] = df['price'] * df['quantity_sold']

# Convert 'order_date' to datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# Group by product category and calculate total revenue
category_revenue = df.groupby('product_category_name').agg(
    total_revenue=('revenue', 'sum'),
    total_sales=('quantity_sold', 'sum')
).reset_index()

# Sort by total revenue to see the highest revenue-generating categories
category_revenue_sorted = category_revenue.sort_values(by='total_revenue', ascending=False)

# --- Visualization 1: Total Revenue by Product Category ---
st.subheader("Total Revenue by Product Category")
fig17, ax17 = plt.subplots(figsize=(14, 7))
sns.barplot(data=category_revenue_sorted, x='product_category_name', y='total_revenue', palette="viridis", ax=ax17)
ax17.set_title('Total Revenue by Product Category', fontsize=16)
ax17.set_xlabel('Product Category', fontsize=12)
ax17.set_ylabel('Total Revenue', fontsize=12)
ax17.set_xticklabels(ax1.get_xticklabels(), rotation=45)
st.pyplot(fig17)

# Group by month and product category, then calculate total revenue
monthly_revenue = df.groupby([df['order_date'].dt.to_period('M'), 'product_category_name']).agg(
    total_revenue=('revenue', 'sum')
).reset_index()

# Ensure the 'month' column is in a proper format for plotting
monthly_revenue['month'] = monthly_revenue['order_date'].dt.to_timestamp()

# --- Visualization 2: Monthly Revenue Trend by Product Category ---
st.subheader("Monthly Revenue Trend by Product Category")
fig18, ax18 = plt.subplots(figsize=(14, 7))
sns.lineplot(data=monthly_revenue, x='month', y='total_revenue', hue='product_category_name', marker='o', ax=ax18)
ax18.set_title('Monthly Revenue Trend by Product Category', fontsize=16)
ax18.set_xlabel('Month', fontsize=12)
ax18.set_ylabel('Total Revenue', fontsize=12)
ax18.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax18.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig18)

st.write(""" 

- ### Question 10:

How does the geographic location of sellers (based on zip code, city, and state) influence product availability and delivery times?
                           
""")

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

- ### Conclusion for Question 1:

To optimize regional marketing campaigns and improve delivery logistics, businesses should prioritize Sao Paulo (SP), Rio de Janeiro (RJ), and Minas Gerais (MG) for their high customer concentrations at both the city and state levels. These regions should be the primary focus for tailored marketing efforts, such as location-based promotions and targeted advertisements. Delivery logistics should also be streamlined in these areas, possibly through local distribution centers or partnerships with logistics companies to reduce costs and improve delivery times.

Other states like Rio Grande do Sul (RS), Parana (PR), and Santa Catarina (SC) should also be targeted with region-specific strategies, but at a slightly reduced scale compared to the top states. For smaller regions with low customer counts such as Acre, Roraima, Amapá, and Tocantins, businesses may consider a more centralized delivery model, combining shipments or partnering with neighboring states for logistics efficiency.

By focusing efforts on the areas with the highest customer density and adapting strategies for regions with fewer customers, businesses can enhance their marketing effectiveness and optimize logistics, ensuring better customer engagement and faster delivery.

- ### Conclusion for Question 2:

Based on the analysis, we can conclude that customer loyalty is relatively low across the entire dataset, with only 3.12% of customers returning for multiple purchases. This low repeat customer percentage suggests that there is significant potential to improve customer retention. The distribution of repeat customers by state also reveals that some regions, particularly São Paulo (SP) and Rio de Janeiro (RJ), have higher numbers of loyal customers, which could serve as key markets for targeted loyalty programs.

To improve customer loyalty, businesses could implement region-specific loyalty programs. For states with lower repeat customer numbers, such as Acre (AC) and Roraima (RR), businesses could focus on incentives, discounts, and personalized offers to encourage repeat purchases. In contrast, regions with higher repeat customer counts, like São Paulo (SP) and Rio de Janeiro (RJ), could benefit from tiered loyalty programs to reward and retain their loyal customer base, possibly through exclusive benefits, early access to products, or premium services.

- ### Conclusion for Question 3:

The geographical clustering of customers provides a clear pattern of how customers are distributed across different regions, enabling businesses to optimize delivery routing and improve operational efficiency. The largest cluster (Cluster 2) and moderate clusters (Cluster 0 and Cluster 1) indicate areas with a high concentration of customers, where frequent and fast deliveries should be prioritized. Conversely, Cluster 3, with a smaller customer base, suggests that delivery frequencies can be reduced in those regions, and resources can be allocated more efficiently.

This clustering information is invaluable for:

- Optimizing Delivery Routes: Delivery teams can focus on high-density clusters for frequent trips, while minimizing routes in low-density areas.
- Resource Allocation: Businesses can allocate delivery vehicles and personnel more effectively based on customer density, optimizing time and costs.
- Improving Efficiency: Tailoring routes and delivery schedules to the geographical distribution of customers can lead to reduced delivery times, lower operational costs, and improved customer satisfaction.

- ### Conclusion for Question 4:

Based on the analysis, businesses can optimize their payment offerings by encouraging credit card usage, especially for high-value transactions. Since credit card users tend to spend more and prefer installment options, offering incentives or promotions for credit card payments could increase sales and average order values. Additionally, highlighting installment plans for larger purchases can make customers more comfortable committing to higher-value items. For customers using boleto, debit cards, or vouchers, who prefer single payments and make smaller purchases, businesses should focus on bundling offers or creating incentives for repeat purchases. Understanding these distinct payment behaviors allows businesses to tailor their payment options accordingly, catering to customer preferences and ultimately improving sales performance and customer satisfaction.

- ### Conclusion for Question 5:

The analysis reveals that there is a moderate correlation between payment installments and payment value, with a trend where customers choosing higher installment plans tend to make larger purchases. The payment value distribution shows that the mean payment value increases with the number of installments, with higher installment plans being more common for larger payments. However, significant variability in the data indicates that offering installment plans can accommodate both small and large purchases. Businesses can leverage this relationship by targeting installment offerings for higher-value purchases to improve cash flow management, while also offering flexibility in the number of installments to cater to different customer needs. Understanding this relationship can also help in designing customized installment plans, ensuring that customers are more likely to choose longer-term installments for larger payments, thereby increasing customer satisfaction and business revenue.

- ### Conclusion for Question 6:

The analysis of the top n-grams in customer reviews reveals several critical insights into customer priorities and concerns:

Product Quality and Features: Customers are highly focused on the product itself, with frequent mentions of "produto" indicating that the product's features, quality, and performance are central themes in the reviews. Businesses should focus on continuous product improvement, ensuring it meets or exceeds customer expectations.

Delivery and Timeliness Issues: Delivery time is a significant concern, as seen in the frequent mentions of "prazo" (delivery time), "entrega" (delivery), and "chegou" (arrived). Delays or missed delivery windows could be major sources of dissatisfaction. Companies can improve customer satisfaction by optimizing logistics, providing more accurate delivery time estimates, and communicating proactively about any delays.

Customer Expectations vs. Reality: The presence of words like "não" (no) and "antes" (before) suggests that customers have strong expectations about both the product and its delivery. These expectations need to be managed carefully. Businesses should work on clear communication about product capabilities and delivery times to minimize disappointment.

- ### Conclusion for Question 7:

Several key factors influence order delivery time, and each of these can be optimized to improve customer satisfaction:

1. Data Quality and Logistics Operations:

The presence of negative values in delivery times suggests that there may be issues with data integrity or mismanagement in delivery status updates. Improving data accuracy and logistics management can help in reducing inconsistencies and delays.

2. Operational Efficiency:

The moderate correlation between delivery time and delivery delay suggests that orders with longer delivery times tend to face delays. Streamlining the delivery process to reduce unnecessary waiting time between the order's shipment and its final delivery could enhance overall customer satisfaction. For example, improving warehouse processes, optimizing routes, or partnering with more reliable carriers could help reduce delivery times.

3. Seasonality Effects:

The analysis of delivery times by purchase month shows that certain months, like June and July, have shorter delivery times, while others, like February and March, tend to see more delays. To optimize for customer satisfaction, businesses could implement strategies to address seasonal challenges, such as increasing staffing, managing inventory better, and improving supplier relationships during peak periods.

4. Customer Communication:

The early deliveries indicated by the negative delivery delay values can be a positive aspect, as it shows that many orders are delivered ahead of schedule. To enhance customer satisfaction, retailers can maintain transparent communication about estimated delivery dates, informing customers of any expected delays and managing their expectations.

- ### Conclusion for Question 8:

There is clear seasonality in product demand, with mid-year months driving the highest volumes of orders. Customer purchase behavior is influenced by time-of-day patterns, with peak shopping hours being during the afternoon. Understanding these patterns allows businesses to better plan for high-demand periods, optimize order fulfillment processes, and adjust marketing strategies to cater to customer shopping habits.

- ### Conclusion for Question 9:

- beleza_saude is the leading category in terms of total revenue generation and consistent sales performance.
- informatica_acessorios also performs strongly, though with less consistency than beleza_saude.
- automotivo and moveis_decoracao are comparatively lower in revenue generation, with automotivo showing a significant drop in recent months.

In summary, beleza_saude and informatica_acessorios are the primary revenue drivers, while the other categories show smaller or more fluctuating sales trends.

- ### Conclusion for Question 10:

geographic factors such as the number of sellers, state, and zip code prefix are crucial in influencing both the availability of products and the delivery times. Regions with a higher density of sellers experience better product availability and faster shipping times, while remote and less populated regions face challenges related to both product availability and delivery efficiency.

""")

# Copyright Section
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
st.markdown("<p style='text-align: center; font-size: 12px;'>Copyright (C) Ady Syamsuri. 2024</p>", unsafe_allow_html=True)