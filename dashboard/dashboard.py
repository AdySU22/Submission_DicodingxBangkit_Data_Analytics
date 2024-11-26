import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import streamlit as st
import urllib

# Set styling for seaborn and streamlit options
sns.set_theme(style='dark')

# DataAnalyzer class
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def create_daily_orders_df(self):
        daily_orders_df = self.df.resample(rule='D', on='order_approved_at').agg({
            "order_id": "nunique",
            "payment_value": "sum"
        })
        daily_orders_df = daily_orders_df.reset_index()
        daily_orders_df.rename(columns={
            "order_id": "order_count",
            "payment_value": "revenue"
        }, inplace=True)
        
        return daily_orders_df
    
    def create_daily_spend_df(self):
        daily_spend_df = self.df.resample(rule='D', on='order_approved_at').agg({
            "payment_value": "sum"
        })
        daily_spend_df = daily_spend_df.reset_index()
        daily_spend_df.rename(columns={
            "payment_value": "total_spend"
        }, inplace=True)

        return daily_spend_df

    def create_sum_order_items_df(self):
        sum_order_items_df = self.df.groupby("product_category_name_english")["product_id"].count().reset_index()
        sum_order_items_df.rename(columns={
            "product_id": "product_count"
        }, inplace=True)
        sum_order_items_df = sum_order_items_df.sort_values(by='product_count', ascending=False)

        return sum_order_items_df

    def review_score_df(self):
        review_scores = self.df['review_score'].value_counts().sort_values(ascending=False)
        most_common_score = review_scores.idxmax() if not review_scores.empty else None

        return review_scores, most_common_score

    def create_bystate_df(self):
        bystate_df = self.df.groupby(by="customer_state").customer_id.nunique().reset_index()
        bystate_df.rename(columns={
            "customer_id": "customer_count"
        }, inplace=True)
        most_common_state = bystate_df.loc[bystate_df['customer_count'].idxmax(), 'customer_state'] if not bystate_df.empty else None
        bystate_df = bystate_df.sort_values(by='customer_count', ascending=False)

        return bystate_df, most_common_state

    def create_order_status(self):
        order_status_df = self.df["order_status"].value_counts().sort_values(ascending=False)
        most_common_status = order_status_df.idxmax() if not order_status_df.empty else None

        return order_status_df, most_common_status

    def create_customer_spend_df(self):
        customer_spend_df = self.df.groupby("customer_id")["payment_value"].sum().reset_index()
        customer_spend_df.rename(columns={"payment_value": "total_spend"}, inplace=True)
        return customer_spend_df


# BrazilMapPlotter class
class BrazilMapPlotter:
    def __init__(self, data, plt, mpimg, urllib, st):
        self.data = data
        self.plt = plt
        self.mpimg = mpimg
        self.urllib = urllib
        self.st = st

    def plot(self):
        brazil = self.mpimg.imread(self.urllib.request.urlopen('https://i.pinimg.com/originals/3a/0c/e1/3a0ce18b3c842748c255bc0aa445ad41.jpg'), 'jpg')
        
        fig, ax = self.plt.subplots(figsize=(10, 10))
        self.data.plot(kind="scatter", x="geolocation_lng", y="geolocation_lat", ax=ax, alpha=0.3, s=0.3, color='maroon')
        
        ax.axis('off')
        ax.imshow(brazil, extent=[-73.98283055, -33.8, -33.75116944, 5.4])
        self.st.pyplot(fig)


# Dashboard implementation starts here
# Dataset
datetime_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "shipping_limit_date"]
all_df = pd.read_csv(r"C:/Users/ASUS/Downloads/Submission_DicodingxBangkit_Data_Analytics/dashboard/df.csv")

print(all_df.head())

all_df.sort_values(by="order_approved_at", inplace=True)
all_df.reset_index(inplace=True)

# Geolocation Dataset
geolocation = pd.read_csv(r"C:/Users/ASUS/Downloads/Submission_DicodingxBangkit_Data_Analytics/dashboard/geolocation_dataset.csv")

print(geolocation.head())

customer_data = None
# Fixing the missing columns issue: Check if 'customer_unique_id' or 'customer_id' exists
if 'customer_unique_id' in all_df.columns:
    print("Option 1")
    customer_data = all_df.drop_duplicates(subset='customer_unique_id')
elif 'customer_id' in all_df.columns:
    print("Option 2")
    customer_data = all_df.drop_duplicates(subset='customer_id')
else:
    print("Option 3")
    st.error("Neither 'customer_unique_id' nor 'customer_id' exist in the all_df customer_data.")
    st.stop()  # This will stop the script execution

geolocation_data = None
# Fixing the missing columns issue: Check if 'customer_unique_id' or 'customer_id' exists
if 'customer_unique_id' in all_df.columns:
    print("Option 1")
    geolocation_data = all_df.drop_duplicates(subset='customer_unique_id')
elif 'customer_id' in all_df.columns:
    print("Option 2")
    geolocation_data = all_df.drop_duplicates(subset='customer_id')
else:
    print("Option 3")
    st.error("Neither 'customer_unique_id' nor 'customer_id' exist in the all_df geolocation_data.")
    st.stop()  # This will stop the script execution

# Convert datetime columns to datetime objects
for col in datetime_cols:
    all_df[col] = pd.to_datetime(all_df[col])

# Get the minimum and maximum order approval dates
min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

# Sidebar
with st.sidebar:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.markdown("<h2 style='text-align: center;'>Ady Syamsuri</h2>", unsafe_allow_html=True)  # Centered and enlarged name
        st.image("C:/Users/ASUS/Downloads/Submission_DicodingxBangkit_Data_Analytics/dashboard/logo-garuda.png", width=100)
    with col3:
        st.write(' ')

    # Date Range input for filtering customer data
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Filter main DataFrame based on the selected date range
main_df = all_df[(all_df["order_approved_at"] >= str(start_date)) & 
                 (all_df["order_approved_at"] <= str(end_date))]

if main_df.empty:
    st.warning("No data available for the selected date range.")
else:
    # Initialize DataAnalyzer and BrazilMapPlotter objects
    function = DataAnalyzer(main_df)

    # We only reach this point if the 'data' variable was defined successfully
    customer_map_plot = BrazilMapPlotter(customer_data, plt, mpimg, urllib, st)
    geolocation_map_plot = BrazilMapPlotter(geolocation, plt, mpimg, urllib, st)

    # Process DataFrames
    daily_orders_df = function.create_daily_orders_df()
    daily_spend_df = function.create_daily_spend_df()
    sum_order_items_df = function.create_sum_order_items_df()
    review_score, common_score = function.review_score_df()
    state, most_common_state = function.create_bystate_df()
    order_status, common_status = function.create_order_status()
    customer_spend_df = function.create_customer_spend_df()

    # Define your Streamlit app
    st.title("E-Commerce Public Data Analysis")

    # Add text or descriptions
    st.write("**This is a dashboard for analyzing E-Commerce public data.**")

    # Daily Orders Delivered Section
    st.subheader("Daily Orders Delivered")
    col1, col2 = st.columns(2)

    with col1:
        total_order = daily_orders_df["order_count"].sum()
        st.markdown(f"Total Order: **{total_order}**")

    with col2:
        total_revenue = daily_orders_df["revenue"].sum()
        # Assuming an exchange rate of 1 unit to 15,000 Rupiah (example rate)
        exchange_rate = 15000
        total_revenue_rupiah = total_revenue * exchange_rate
        st.markdown(f"Total Revenue: **{total_revenue_rupiah:,.0f} IDR**")  # Removed "000"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x=daily_orders_df["order_approved_at"],
        y=daily_orders_df["order_count"],
        marker="o",
        linewidth=2,
        color="#90CAF9"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", labelsize=15)
    st.pyplot(fig)

    # Customer Spend Money Section
    st.subheader("Customer Spend Money")
    total_spend = daily_spend_df["total_spend"].sum()
    average_spend = daily_spend_df["total_spend"].mean()
    st.markdown(f"Total Spend by Customers: **{total_spend}**")
    st.markdown(f"Average Spend by Customers: **{average_spend:.2f}**")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x=daily_spend_df["order_approved_at"],
        y=daily_spend_df["total_spend"],
        marker="o",
        linewidth=2,
        color="#FFAB40"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title("Daily Customer Spend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Spend")
    st.pyplot(fig)

    # Order Items by Product Category Section - Most Sold
    st.subheader("Most Items Sold by Product Category")
    total_order_items = sum_order_items_df["product_count"].sum()
    st.markdown(f"Total Order Items: **{total_order_items}**")
    st.markdown(f"Average Order Items per Category: **{total_order_items / len(sum_order_items_df):.2f}**")

    fig, ax = plt.subplots(figsize=(12, 6))
    top_sold_items_df = sum_order_items_df.head(10)
    sns.barplot(data=top_sold_items_df, x="product_category_name_english", y="product_count", palette="viridis")
    ax.set_title("Top 10 Most Sold Items by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Number of Order Items")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    # Order Items by Product Category Section - Least Sold
    st.subheader("Least Items Sold by Product Category")
    least_sold_items_df = sum_order_items_df.tail(10)
    st.markdown(f"Total Order Items: **{least_sold_items_df['product_count'].sum()}**")
    st.markdown(f"Average Order Items per Category: **{least_sold_items_df['product_count'].mean():.2f}**")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=least_sold_items_df, x="product_category_name_english", y="product_count", palette="magma")
    ax.set_title("Top 10 Least Sold Items by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Number of Order Items")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    # Review Scores Distribution Section
    st.subheader("Review Scores Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    review_score.plot(kind='bar', ax=ax, color='orange')
    ax.set_title("Distribution of Review Scores")
    ax.set_xlabel("Review Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Display the most common review score
    st.markdown(f"The most common review score is: **{common_score}**")

    # Geolocation Section - Map
    st.subheader('Customer Geolocation in Brazil')
    geolocation_map_plot.plot()

    # Conclusion Section
    st.subheader("Conclusion")
    st.write(""" 
    The analysis indicates that the "bed_bath_table" category has the highest sales volume, while the "auto" category shows the least. This distinction highlights consumer preferences and trends within the product categories.

    Sales performance exhibited stability from January to May, followed by a slight decline from June to July. A minor uptick was observed in August, which was subsequently followed by a significant drop in September. Notably, there was a sharp rise in sales during October and November, although this was accompanied by another decrease in December.

    Customer spending patterns reflected these fluctuations. Spending remained stable during the first half of the year, experienced a decline from June to September, and saw a significant increase in October and November before tapering off again in December.

    Customer satisfaction levels are remarkably high, as evidenced by the majority of customers rating their experiences as 5 stars, with a substantial number also providing a 4-star rating. This positive feedback underscores the effectiveness of the service provided.

    The demographic profile reveals that São Paulo (SP) has the highest concentration of customers, which suggests that it is a key market. The predominant order status of "delivered" indicates successful fulfillment of orders, contributing to the high customer satisfaction ratings.

    Furthermore, the majority of customers are situated in the southeastern and southern regions of Brazil, particularly in capital cities such as São Paulo, Rio de Janeiro, and Porto Alegre. This geographic insight provides valuable information for targeted marketing and service improvement strategies.
    """)

    # Copyright Section
    st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
    st.markdown("<p style='text-align: center; font-size: 12px;'>Copyright (C) Ady Syamsuri. 2024</p>", unsafe_allow_html=True)