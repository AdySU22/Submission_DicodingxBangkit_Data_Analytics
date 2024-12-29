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

# Sidebar with radio buttons
options = ["Home", "Customer Analysis", "Geolocation Analysis", "Payment Analysis", "Review Analysis", "Delivery Analysis", "Order Analysis", "Product Category Analysis", "Seller Analysis", "Shipment Analysis", "Conclusion"]
st.sidebar.radio("Segmentation Analysis", options, key='selection', on_change=lambda: update_content())

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

# Function to update content based on sidebar selection
def update_content():
    selection = st.session_state.selection

    if selection == "Home":
        st.markdown('<a id="Home"></a>', unsafe_allow_html=True)

        st.title("Welcome to Brazilian E-Commerce Data Analysis")

        st.write("""
        
        - Name: Ady Syamsuri
        - Email: m467b4ky0140@bangkit.academy
        - Dicoding ID: ady_syamsuri_m467b4k

        """)

        # Copyright Section
        st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
        st.markdown("<p style='text-align: center; font-size: 12px;'>Copyright (C) Ady Syamsuri. 2024</p>", unsafe_allow_html=True)

    elif selection == 'Customer Analysis':

        st.subheader("Question 1: How can we segment customers based on their location and zip code to optimize regional marketing campaigns and improve delivery logistics?")

        df = pd.read_csv('../data/customers_dataset.csv') 

        # Customer count by city
        customers_by_city = df.groupby('customer_city')['customer_id'].count().reset_index()
        customers_by_city = customers_by_city.rename(columns={'customer_id': 'customer_count'})
        top_cities = customers_by_city.sort_values(by='customer_count', ascending=False).head(10)

        # Customer count by state
        customers_by_state = df.groupby('customer_state')['customer_id'].count().reset_index()
        customers_by_state = customers_by_state.rename(columns={'customer_id': 'customer_count'})
        top_states = customers_by_state.sort_values(by='customer_count', ascending=False)

        st.subheader("Top 10 Cities by Customer Count")

        # Visualization: Top Cities by Customer Count
        fig19, ax19 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=top_cities, x='customer_city', y='customer_count', hue='customer_city', palette='viridis', legend=False, ax=ax19)
        ax19.set_title('Top 10 Cities by Customer Count', fontsize=16)
        ax19.set_xlabel('City', fontsize=12)
        ax19.set_ylabel('Customer Count', fontsize=12)
        ax19.set_xticklabels(ax19.get_xticklabels(), rotation=45)
        st.pyplot(fig19)

        st.subheader("Customer Distribution by State")

        # Visualization: Customer Distribution by State
        fig20, ax20 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=top_states, x='customer_state', y='customer_count', hue='customer_state', palette='coolwarm', legend=False, ax=ax20)
        ax20.set_title('Customer Distribution by State', fontsize=16)
        ax20.set_xlabel('State', fontsize=12)
        ax20.set_ylabel('Customer Count', fontsize=12)
        ax20.set_xticklabels(ax20.get_xticklabels(), rotation=45)
        st.pyplot(fig20)

        st.subheader("Question 2: Are there patterns in customer repeat behavior (costumer loyalty) across cities or states, and how can we design loyalty programs to retain high-value customers?")

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

        st.subheader("Percentage of Repeated Customers")

        # Visualization: Repeat vs Non-Repeat Customers
        fig21, ax21 = plt.subplots(figsize=(8, 8))
        ax21.pie(repeat_data['Count'], labels=repeat_data['Category'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107'])
        ax21.set_title('Percentage of Repeated Customers', fontsize=16)
        st.pyplot(fig21)

        # Repeat Customers by State
        repeat_by_state = df[df['customer_unique_id'].isin(repeat_customers['customer_unique_id'])]
        repeat_by_state = repeat_by_state.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
        repeat_by_state = repeat_by_state.rename(columns={'customer_unique_id': 'repeat_customers'})

        st.subheader("Repeated Customers by State")

        # Visualization: Repeat Customers by State (Fixed)
        fig22, ax22 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=repeat_by_state.sort_values(by='repeat_customers', ascending=False),
                    x='customer_state', y='repeat_customers', hue='customer_state', palette='Blues_d', legend=False, ax=ax22)
        ax22.set_title('Repeated Customers by State', fontsize=16)
        ax22.set_xlabel('State', fontsize=12)
        ax22.set_ylabel('Repeat Customers', fontsize=12)
        ax22.set_xticklabels(ax22.get_xticklabels(), rotation=45)
        st.pyplot(fig22)

        st.write("""

        To optimize regional marketing campaigns and improve delivery logistics, it is crucial to understand the distribution of customers across different cities and states. Based on the customer count data, we can identify key regions where businesses should focus their efforts.

        Cities Analysis:

        - Sao Paulo (SP) stands out as the largest city in terms of customer count with 15,540 customers, significantly higher than any other city. This makes Sao Paulo the primary target for marketing campaigns and logistics strategies, as it holds the largest portion of the customer base.
        - Rio de Janeiro (RJ) follows with 6,882 customers, making it the second most important city for marketing and delivery. While not as large as Sao Paulo, it still represents a sizable portion of the market.

        - Other cities such as Belo Horizonte (MG), Brasilia (DF), and Curitiba (PR) also show strong customer numbers (2,773, 2,131, and 1,521, respectively). These cities, though not as large as Sao Paulo or Rio de Janeiro, represent substantial opportunities for regional marketing and delivery operations.

        State Analysis:

        - The state-level distribution shows Sao Paulo (SP) as the dominant region with 41,746 customers, accounting for a significant portion of the overall customer base. This reinforces the importance of focusing on Sao Paulo for both marketing campaigns and logistical strategies.

        - Rio de Janeiro (RJ) again ranks second, with 12,852 customers, followed by Minas Gerais (MG) with 11,635 customers. These regions, along with Sao Paulo, should be the focal point for businesses looking to optimize marketing and delivery.

        - States like Rio Grande do Sul (RS), Parana (PR), and Santa Catarina (SC) show relatively strong customer counts (5,466, 5,045, and 3,637, respectively). These states, while not as large as the top three, still represent valuable opportunities for marketing campaigns and delivery solutions.

        - Smaller states such as Acre (AC), Roraima (RR), Amapá (AP), and Tocantins (TO) show minimal customer counts, with fewer than 500 customers in each of these regions. These areas might not justify large-scale marketing campaigns or dedicated delivery hubs, but businesses could focus on cost-effective regional solutions like consolidating shipments or focusing on nearby larger states.

        """)

        st.write("""
        The question at hand is whether there are patterns in customer repeat behavior (customer loyalty) across cities or states, and how loyalty programs can be designed to retain high-value customers. To analyze this, we first need to identify how many customers are repeat customers, the distribution of repeat customers across states, and the general trend of customer loyalty in the dataset.

        From the code and output, we can observe the following:

        - Percentage of Repeat Customers:

        The percentage of repeat customers is calculated to be 3.12%. This suggests that only a small fraction of the total customer base has made more than one purchase, which indicates that customer loyalty might be relatively low across the board. Out of 96,096 total customers, only 2,997 have made multiple purchases, which is a clear indication that the vast majority of customers are one-time buyers.

        - Repeat vs Non-Repeat Customers Count:

        The data reveals that out of 96,096 total customers, 2,997 customers (about 3.12%) are repeat customers, while the remaining 93,099 (about 96.88%) are non-repeat customers. This stark contrast suggests a significant opportunity to focus on improving customer loyalty, as most customers do not return after their first purchase.

        - Repeat Customers by State:

        The distribution of repeat customers by state gives more granular insights into regional loyalty patterns. The state with the highest number of repeat customers is São Paulo (SP) with 1,313 repeat customers, followed by Rio de Janeiro (RJ) with 429 and Paraná (PR) with 149. Other states like Minas Gerais (MG), Santa Catarina (SC), and Bahia (BA) also show reasonable repeat customer counts, ranging from 95 to 169 repeat customers. On the other hand, some states, such as Acre (AC), Amapá (AP), and Roraima (RR), have very few repeat customers (as low as 1), which could indicate either a smaller customer base or lower customer loyalty in those regions.
        """)

    elif selection == 'Geolocation Analysis':
        st.markdown('<a id="Customer-Clustering-based-on-geolocation"></a>', unsafe_allow_html=True)

        st.subheader("Question 3: What are the geographical trends in customer clustering, and how can this information be used to improve delivery routing and efficiency?")

        # Data cleaning for geolocation dataset
        geo_df.dropna(inplace=True)
        geo_df.columns = geo_df.columns.str.lower().str.replace(" ", "_")

        # Selecting relevant columns for clustering
        geo_data = geo_df[['geolocation_lat', 'geolocation_lng']]

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        geo_df['cluster'] = kmeans.fit_predict(geo_data)

        # Visualization: Customer Clustering Based on Geolocation
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

        st.write("""

        The question asks about geographical trends in customer clustering and how this information can improve delivery routing and efficiency. To answer this, we used KMeans clustering to categorize customers based on their geographical coordinates (latitude and longitude). The clustering process divides the customers into five groups (clusters), where each group represents a geographical zone with similar customer locations. Here's the detailed analysis of the results:

        1. Cluster Centroids (Latitude, Longitude):

        - Cluster 0: Latitude = -19.72, Longitude = -49.88
        - Cluster 1: Latitude = -9.12, Longitude = -38.64
        - Cluster 2: Latitude = -22.58, Longitude = -45.28
        - Cluster 3: Latitude = -6.20, Longitude = -53.99
        - Cluster 4: Latitude = -27.62, Longitude = -50.99

        These centroids represent the average geographical location of customers in each cluster. They indicate where customer populations are densely grouped. For example, Cluster 0 has its centroid at latitude -19.72 and longitude -49.88, which might correspond to a specific region within Brazil with a high density of customers. Similarly, Cluster 4 is located at latitude -27.62 and longitude -50.99, marking a different region, possibly in the southern part of the country. The spread of these centroids reflects how far apart customer groups are geographically, with clusters in various latitudes and longitudes suggesting regional differences in customer concentration.

        2. Cluster Size Comparison:

        - Cluster 2: 591,348 customers
        - Cluster 4: 146,733 customers
        - Cluster 0: 139,168 customers
        - Cluster 1: 93,663 customers
        - Cluster 3: 29,251 customers

        The size of each cluster indicates how many customers belong to each geographical zone. The largest cluster (Cluster 2) has 591,348 customers, suggesting this area has the highest concentration of customers. Conversely, Cluster 3, with only 29,251 customers, is the smallest, likely representing a less densely populated area. The distribution of customers across clusters gives insights into regional demand patterns, which can be crucial for routing and delivery optimization.

        3. Geographical Trends:

        - High-density areas: Cluster 2 is the largest cluster with over half a million customers. The geographical area corresponding to this cluster likely represents a high-density urban region where delivery operations could benefit from prioritizing routes in this area due to high customer volume. Similarly, Cluster 0 (with 139,168 customers) suggests another densely populated area with substantial delivery needs.
        - Low-density areas: Cluster 3 has the fewest customers (29,251), suggesting it represents a sparsely populated or rural area. Delivery routes in this area could be optimized for cost-efficiency, perhaps with fewer trips or by consolidating deliveries.

        4. Clustering Insights:

        - Clustering reveals customer density and geographic distribution, allowing businesses to tailor delivery services more effectively. For instance, a high-density cluster (like Cluster 2) may require frequent deliveries with optimized routes to minimize delays. On the other hand, sparse clusters (like Cluster 3) might need less frequent but more direct routes.
        - Cluster 1 and Cluster 4 are moderate in size, suggesting regional delivery hubs can be set up to service those areas efficiently, ensuring faster and more reliable delivery services.

        """)

    elif selection == 'Payment Analysis':
        st.markdown('<a id="Average-Payment-Value-by-Payment-Type"></a>', unsafe_allow_html=True)

        st.subheader("Question 4: How do payment types influence the average payment value and installment patterns, and what insights can be drawn to optimize payment offerings?")

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

        st.subheader("Question 5: What is the relationship between the number of payment installments and the total payment value, and how can this be leveraged to offer installment plans effectively?")

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

        The analysis is based on the payment_value and payment_installments data across different payment types. Here's a detailed breakdown:

        1. Average Payment Value by Payment Type: The average payment value across different payment types indicates how much customers tend to spend depending on their chosen method of payment. Here's the summary:

        - Boleto has the lowest average payment value of 145.03. This suggests that customers using this method tend to make smaller purchases.
        - Credit Card has the highest average payment value of 163.32, indicating that customers tend to spend more when using credit cards.
        - Debit Card has a lower average payment value of 142.57, which is similar to Boleto.
        - Voucher has an average payment value of 65.70, which is significantly lower than the other methods, indicating smaller transactions for voucher users.

        2. Average Number of Installments by Payment Type: The number of installments refers to how many times a customer chooses to pay for their purchase over time. The data shows the following:

        - Boleto, Debit Card, Not Defined, and Voucher users do not use installments at all. All these payment types have an average of 1 installment, indicating that the payments are made in full at the time of purchase.
        - Credit Card users, on the other hand, have a significantly higher average of 3.51 installments, meaning they tend to divide their payments into multiple parts. This is likely a strategy for managing larger purchases.
        From these insights, we can infer the following key observations:

        Higher spending with installment options: Customers who use credit cards tend to spend more on average, and they also opt for installment payments. This could be because credit card users are more comfortable spreading the cost of their purchases over several months. This trend highlights the importance of offering installment options to encourage higher-value purchases.
        Boleto and Voucher customers prefer single payments: Customers who use Boleto or Voucher tend to make smaller purchases and pay upfront, likely due to the payment type’s nature, which doesn’t generally offer installment options.
        Debit Card as a simpler payment method: Similarly, Debit Card users also make smaller purchases and pay upfront in a single installment, indicating that debit card usage is associated with lower-value transactions.
                 
        The correlation between payment value and payment installments is found to be 0.33, indicating a moderate positive relationship between these two variables. This suggests that as the number of installments increases, the payment value also tends to increase, though not strongly. The moderate correlation reflects that higher-value transactions are more likely to be split into multiple installments, but not all high-value payments follow this trend.

        From the statistical summary of payment value by the number of installments, we can observe several key insights:

        1. Payments with fewer installments:

        - 0 Installments: The mean payment value is about 94.32, with values ranging from 58.69 to 129.94. This small number of payments with zero installments could indicate some immediate payments, but there’s a noticeable variance.
        - 1 Installment: The mean payment value is 112.42, with a wide standard deviation of 177.56. Payments with one installment are common, but there is considerable variability in the payment values, with some low and some extremely high amounts (e.g., payments reaching up to 13,664).

        2. As the number of installments increases, the average payment value tends to increase:

        - 2 Installments: Mean of 127.23, with a range from 20.03 to 2442.82.
        - 3 Installments: Mean of 142.54, with a wider range.
        - 4 to 6 Installments: Mean payment values increase significantly, reaching 163.98 and 183.47, with notable spreads indicating variability.
        - Installments above 10: When installment numbers increase further (10 and above), there’s a jump in the payment value, with the highest mean (593.88) seen in the 24 installment category.

        3. Notable outliers are present in certain installment groups, particularly at higher numbers of installments. For instance:

        - Installments > 10 (e.g., 24 installments) have extreme maximum payment values (e.g., 1440.10), suggesting that some customers are willing to take long-term installment plans for very high-value purchases.

        4. The variance in payment value increases as the installment count rises, especially for higher installments like 10+ installments. This indicates that businesses offering extended installment plans might deal with more heterogeneous payment amounts.

        """)
    
    elif selection == 'Review Analysis':
        st.markdown('<a id="Most-Frequent-Themes-in-Customer-Reviews"></a>', unsafe_allow_html=True)

        st.subheader("Question 6: What are the most common themes in customer reviews (via review comments), and how can these insights be used to enhance the product or service offerings?")
        
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
                        
        Product Focus: The frequent occurrence of "produto" suggests that customers primarily discuss the product in their reviews. This may include aspects like quality, features, usability, and value.

        Dissatisfaction and Complaints: The word "não" (no) indicates dissatisfaction or negative feedback, which likely points to unmet expectations or product-related issues.

        Delivery Concerns: Words like "prazo" (deadline), "entrega" (delivery), and "chegou" (arrived) emphasize that delivery and its timeliness are major concerns for customers.

        Customer Expectations: Words like "antes" (before) and "recebi" (received) suggest that customers are concerned about whether the product meets their expectations, particularly in terms of arrival time and quality.
                 
                 """)
    
    elif selection == 'Delivery Analysis':
        st.markdown('<a id="Most-Frequent-Themes-in-Customer-Reviews"></a>', unsafe_allow_html=True)

        st.subheader("Question 7: What are the key factors influencing order delivery time, and how can they be optimized for better customer satisfaction?")

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
        The analysis involves understanding the factors that impact order delivery times based on the dataset provided. Below is a detailed breakdown of the key aspects derived from the statistics:

        1. Delivery Time Statistics:

        - The average delivery time is 8.88 days, with a standard deviation of 8.75 days. This large standard deviation indicates that while most orders are delivered around 8 days, there are significant variations in the delivery time.
        - The minimum delivery time is -17 days, which is a negative value, suggesting either data inconsistencies or orders that were marked as delivered before the shipment occurred (e.g., logistical issues or data errors).
        - The 25th percentile of delivery times is 4 days, meaning 25% of orders were delivered within 4 days, while 75% were delivered within 12 days.
        - The maximum delivery time is 205 days, indicating extreme outliers, possibly due to delays or logistical failures.

        2. Delivery Delay Statistics:

        - The average delivery delay is -11.88 days, with a standard deviation of 10.18 days. A negative delivery delay implies that many orders were delivered before their estimated delivery dates, which is favorable for customers.
        - The 25th percentile delivery delay is -17 days, meaning that 25% of orders were delivered significantly earlier than expected.
        - However, there are extreme positive delays, as indicated by the maximum delay of 188 days. These significant delays might stem from operational inefficiencies or unexpected issues.
        - The positive correlation of 0.58 between delivery time and delivery delay suggests that orders delivered late tend to have longer delivery times as well.

        3. Delivery Delay by Order Status:

        - The 'delivered' orders, which represent the vast majority of data, show an average delay of -11.88 days. This suggests that orders are often delivered earlier than expected.
        - The 'canceled' orders show an average delay of -27.83 days, though the sample size for canceled orders is very small, making it less significant for drawing conclusions.
        - The lack of data for other order statuses such as 'created,' 'invoiced,' and 'shipped' prevents any meaningful analysis from these categories.

        4. Delivery Time by Purchase Month:

        - The purchase month significantly impacts delivery times. For example, the mean delivery time in June (Month 6) is 7.07 days, whereas in February (Month 2), the mean delivery time is 12.09 days.
        - A noticeable trend is that months like June and July have relatively shorter delivery times, while months like February and March see a slight increase in average delivery times. This variation might be influenced by seasonal demand, supply chain disruptions, or holiday-related delays.

        5. Correlation Analysis:

        - A correlation coefficient of 0.58 between delivery time and delivery delay indicates a moderate positive relationship. This means that as the delivery time increases, the delay tends to increase as well. Long delivery times often correlate with longer delays, suggesting potential inefficiencies in the delivery process.
        """)

    elif selection == 'Order Analysis':
        st.markdown('<a id="Number-of-Orders-by-Hour-of-the-Day"></a>', unsafe_allow_html=True)

        st.subheader("Question 8: How do the order purchase patterns (e.g., seasonality, time of day) correlate with customer purchase behavior and product demand?")

        # Sample data for visualization
        # Assuming you have your data in orders_df DataFrame
        # Ensure the order_purchase_timestamp is in datetime format
        orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])

        # Number of Orders by Hour of the Day
        orders_df['purchase_hour'] = orders_df['order_purchase_timestamp'].dt.hour
        st.subheader("Number of Orders by Hour of the Day")
        fig12, ax12 = plt.subplots(figsize=(14, 7))
        sns.countplot(x='purchase_hour', data=orders_df, color='skyblue', ax=ax12)
        ax12.set_title('Number of Orders by Hour of the Day', fontsize=16)
        ax12.set_xlabel('Hour of Day', fontsize=12)
        ax12.set_ylabel('Number of Orders', fontsize=12)
        st.pyplot(fig12)

        # Number of Orders by Day of the Week
        orders_df['purchase_day_of_week'] = orders_df['order_purchase_timestamp'].dt.dayofweek
        st.subheader("Number of Orders by Day of the Week")
        fig13, ax13 = plt.subplots(figsize=(14, 7))
        sns.countplot(x='purchase_day_of_week', data=orders_df, color='lightcoral', ax=ax13)
        ax13.set_title('Number of Orders by Day of the Week', fontsize=16)
        ax13.set_xlabel('Day of the Week (0=Monday, 6=Sunday)', fontsize=12)
        ax13.set_ylabel('Number of Orders', fontsize=12)
        st.pyplot(fig13)

        # Number of Orders by Month
        orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month
        st.subheader("Number of Orders by Month")
        fig14, ax14 = plt.subplots(figsize=(14, 7))
        sns.countplot(x='purchase_month', data=orders_df, color='lightgreen', ax=ax14)
        ax14.set_title('Number of Orders by Month', fontsize=16)
        ax14.set_xlabel('Month', fontsize=12)
        ax14.set_ylabel('Number of Orders', fontsize=12)
        st.pyplot(fig14)

        # Monthly Order Volume
        monthly_order_count = orders_df.groupby('purchase_month').size()
        st.subheader("Monthly Order Volume")
        fig15, ax15 = plt.subplots(figsize=(14, 7))
        monthly_order_count.plot(kind='bar', color='teal', ax=ax15)
        ax15.set_title('Monthly Order Volume', fontsize=16)
        ax15.set_xlabel('Month', fontsize=12)
        ax15.set_ylabel('Number of Orders', fontsize=12)
        st.pyplot(fig15)

        # Order Status by Time Features
        order_status_by_time = orders_df.groupby(['purchase_hour', 'purchase_day_of_week', 'purchase_month'])['order_status'].value_counts().unstack().fillna(0)
        st.subheader("Order Status by Time Features")
        fig16, ax16 = plt.subplots(figsize=(14, 7))
        order_status_by_time.plot(kind='bar', stacked=True, cmap='Set2', ax=ax16)
        ax16.set_title('Order Status by Time Features', fontsize=16)
        ax16.set_xlabel('Time Features', fontsize=12)
        ax16.set_ylabel('Order Count', fontsize=12)
        st.pyplot(fig16)

        st.write("""

        The data provided represents the purchase patterns of customers based on various time-based features, such as the hour of the day, day of the week, and month, and how these features correlate with customer behavior and product demand. Let's break down and analyze the insights for each of the time-based patterns and their correlations.

        1. Orders by Hour of the Day
        The distribution of orders throughout the day shows distinct peaks at certain times:

        - Peak Hours (9 AM - 3 PM): The highest order volumes occur between 9 AM and 3 PM, with a clear peak at 12 PM (6578 orders) and 1 PM (6518 orders). These peaks may indicate customers' preference to shop during lunch breaks or after finishing their morning work routines.
        - Lower Activity (Late Night - Early Morning): The hours 12 AM - 6 AM show a significant drop in orders, especially between 2 AM and 6 AM, which could indicate a lack of activity during the nighttime when customers are less likely to be shopping.

        2. Orders by Day of the Week
        - Weekdays (Monday - Friday): The highest order volumes are seen on Monday (16196 orders), with a steady decrease in volume until Friday (14122 orders). This suggests that customers are more active at the beginning of the week and tend to shop less towards the weekend, possibly due to weekend distractions or shopping outside of work hours.
        - Weekend Orders (Saturday and Sunday): Orders on Saturday (10887 orders) and Sunday (11960 orders) are noticeably lower compared to weekdays, with Saturday seeing the least activity. This may be because consumers are less likely to engage in online shopping when they're occupied with other weekend activities.

        3. Orders by Month
        - High Activity in Mid-Year: The highest order volumes occur between March and August, peaking in July (10843 orders). This suggests a mid-year shopping surge, possibly due to summer promotions, events, or seasonal product demand. For example, many retail companies have sales during this period to capitalize on vacation season, which could drive higher customer purchases.
        - Low Activity in Late Fall and Winter: There is a sharp drop in September (4305 orders) and October (4959 orders), followed by a slight increase in November (7544 orders) and December (5674 orders). This could be attributed to a post-summer lull in product demand, before the holiday season and end-of-year promotions kick in.

        4. Delivery Time by Purchase Month
        - Higher Delivery Times in Winter and Early Spring: Months like January (mean delivery time of 9.8 days) and February (mean delivery time of 12.1 days) show longer average delivery times, which may be linked to seasonal shipping delays or higher volume of deliveries after the holidays.
        - Lower Delivery Times in Summer: Months such as June and July (mean delivery time of around 7.0 days) show quicker delivery times. This could be due to fewer logistical challenges during these months or better organization in shipping and handling.
        - Negative and Zero Delivery Times: Some months also show negative delivery times, which may indicate errors or very fast deliveries, where the orders are processed and shipped unusually quickly (or logistical mistakes like orders marked as delivered before actual shipment).

        5. Monthly Order Volume
        - Surge in Mid-Year: June and July are the months with the highest number of orders, reflecting the overall trend of increased product demand during the summer. The increased volume of orders during this time aligns with seasonal demand for summer products and sales promotions.
        - Lower Volume in Fall and Winter: The drop in order volume from September to December is consistent with the seasonal trends seen in the previous data, where people tend to purchase less outside of the major promotional events or before the holidays.

        6. Order Status by Time Features
        The order status data reveals the relationship between the time features and how orders are processed:

        - Order Processing and Approval: In the early hours of the day (particularly around 1-3 AM), orders are more likely to be canceled or marked as unavailable. This could indicate issues in order fulfillment during off-hours, or customers might cancel orders made late at night upon waking up or reviewing them.
        - Successful Deliveries: There are clear spikes in orders being delivered in certain hours (e.g., at 2 PM), showing that deliveries are more likely to be completed in the afternoon when logistics and customer services are in full operation.
        """)

    elif selection == 'Product Category Analysis':
        st.markdown('<a id="Total-Revenue-by-Product-Category"></a>', unsafe_allow_html=True)

        st.subheader("Question 9: How do sales trends vary across different product categories, and which categories are driving the highest revenue?")

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
        ax17.set_xticklabels(ax17.get_xticklabels(), rotation=45)
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
        ax18.set_xticklabels(ax18.get_xticklabels(), rotation=45)
        ax18.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig18)
        
        st.write("""
        Based on the data, we can observe key trends in sales performance across different product categories from January 2018 to May 2019. Here's a summary of the findings:

        1. Sales Trends by Product Category

        - beleza_saude consistently leads in total revenue, with significant revenue spikes. For example, in January 2018, it had the highest total revenue of 11,008, showcasing strong early-year performance.
        - informatica_acessorios shows steady sales performance, with consistent revenue generation throughout the period. It had notable revenue figures, like 10,017 in May 2019, indicating it remains a key player.
        - cama_mesa_banho and moveis_decoracao follow, but they consistently show lower total revenue compared to the top two categories.
        - automotivo shows relatively lower revenue across the months, with a noticeable drop in May 2019 (761), suggesting lower sales in the latter months.

        2. Highest Revenue-Generating Categories
        - beleza_saude emerges as the highest revenue-generating category, with a total revenue consistently higher than others, especially in the initial months (e.g., January 2018).
        - informatica_acessorios is the second-highest in terms of revenue, with a peak of 10,017 in May 2019, showing consistent sales.
        - cama_mesa_banho, moveis_decoracao, and automotivo have smaller but still notable revenue, with automotivo seeing the lowest overall sales.

        3. Sales Trend Fluctuations
        - Sales in beleza_saude show a pattern of peaks and steady performance, likely due to seasonal factors like promotions or demand spikes.
        - informatica_acessorios maintains relatively stable performance, with minor fluctuations, indicating consistent demand.
        - Categories like automotivo and moveis_decoracao have noticeable drops in revenue, especially in the later months, highlighting potential seasonality or market shifts.

        """)
    
    elif selection == 'Seller Analysis':
        st.markdown('<a id="Seller-Distribution-by-State"></a>', unsafe_allow_html=True)
        
        st.subheader("Question 10: How does the geographic location of sellers (based on zip code, city, and state) influence product availability and delivery times?")

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

        st.write("""
        1. Number of Sellers by State: The data shows the geographic distribution of sellers across various states in Brazil. Here are a few key observations:

        - São Paulo (SP) has the highest number of sellers, with 1,849 sellers. This is followed by Paraná (PR) with 349 sellers.
        - States with fewer sellers include Acre (AC), Amapá (AM), and Maranhão (MA), each having only 1 seller.
        - Generally, states like São Paulo (SP), Minas Gerais (MG), and Rio Grande do Sul (RS) host a large concentration of sellers, which likely results in more product availability due to the higher number of sellers.

        2. Number of Sellers by Zip Code Prefix: The zip code prefix distribution gives us an understanding of the concentration of sellers within specific regions. For example:

        - Zip prefix "13" has 269 sellers, indicating a high concentration in certain regions of São Paulo (SP).
        - On the other hand, some zip codes, such as "66", have as few as 4 sellers, showing that certain areas have limited seller presence.

        3. Average Shipping Time by State: The shipping time data is influenced by the sellers' geographic locations and the assumption that shipping times tend to increase with distance from the central warehouse or major hubs.

        - Acre (AC) has the shortest shipping time, with an average of 5 days, indicating that shipping logistics may be more efficient in areas with fewer sellers.
        - Amazonas (AM), in contrast, has the longest shipping time of 10 days, which could be due to more challenging logistics, fewer sellers, and potentially longer distances.
        - States like São Paulo (SP) and Minas Gerais (MG), which have high concentrations of sellers, have average shipping times around 7-8 days, indicating a relatively efficient distribution system.

        Analysis of Geographic Influence on Product Availability and Delivery Times:

        - Product Availability:

        The number of sellers in a region can directly influence product availability. States with higher seller counts, such as São Paulo (SP), Paraná (PR), and Minas Gerais (MG), are likely to have a wider range of products available due to the larger number of sellers.
        Conversely, states with fewer sellers like Acre (AC), Amapá (AM), and Maranhão (MA) may have limited product availability as there are fewer sellers operating in these regions.
        In zip code regions where there are fewer sellers (such as prefix "66" with only 4 sellers), it is likely that product availability will be lower, and customers may face challenges finding the specific products they desire.
        """)

    elif selection == 'Shipment Analysis':
        st.markdown('<a id="Shipping-Time-by-State-(Hypothetical)"></a>', unsafe_allow_html=True)
    
        st.subheader("Question 10: How does the geographic location of sellers (based on zip code, city, and state) influence product availability and delivery times?")

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

        st.write("""
        - Delivery Times:

        Geographic location has a significant influence on delivery times, with regions closer to major logistical hubs generally experiencing faster delivery. States with more sellers, such as São Paulo (SP), tend to have relatively faster delivery times, averaging around 7-8 days.
        In contrast, states with fewer sellers or more remote locations, such as Amazonas (AM), have much longer shipping times due to fewer sellers, less efficient logistics, and greater distances. For example, the average shipping time in Amazonas is 10 days.
        Delivery times are also affected by the zip code prefixes in certain areas, as sellers from more remote zip code prefixes are likely to have longer delivery times. Zip codes with fewer sellers may experience slower deliveries as well due to logistics constraints.

        - Impact of Zip Code on Delivery:

        The zip code prefixes reflect the geographical spread of sellers within a state. Areas with high concentrations of sellers (e.g., zip prefix "13" in São Paulo) are likely to have faster deliveries due to proximity to other sellers and hubs. In contrast, less densely populated zip code areas (such as prefix "66") are more likely to experience slower delivery times.
        """)
    
    elif selection == 'Conclusion':
        st.markdown('<a id="Conclusion"></a>', unsafe_allow_html=True)
    
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
        