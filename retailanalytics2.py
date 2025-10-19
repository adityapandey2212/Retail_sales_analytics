import pandas as pd 
import numpy as np
import seaborn as sb 
import matplotlib.pyplot as plt
import datetime 
from scipy import stats ## this liibrary is used for using statistical functions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st


st.title("Retail Sales Analytics")

uploaded_file = st.file_uploader("Upload your csv file", type=["csv"])
if not uploaded_file:
    st.info("Please upload the Sample‑Superstore CSV to view charts.")
    st.stop()

def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Forecast', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)


# custom style for submit button - color and width

def style_submit_button():

    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)


# custom style for prediction result text - color and position

def style_prediction():

    st.markdown(
        """
            <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
        unsafe_allow_html=True
    )


# cleaning the data
data = pd.read_csv(r'salesAnalytics/Sample - Superstore.csv', encoding='ISO-8859-1')
print(data.head()) ## this step is done to see whether the data is loaded correrctly or not

print(data.info())
st.write("File loaded. First 5 rows:")
st.dataframe(data.head())

# now we will turn the data into a correlation matrix of heatmap
num_data = data.select_dtypes(include='number') ## this helps in including only numeric data
correlation_matrix = num_data.corr() ## this turns the data into correlation matrix mraniing a square matrix inform of i,j
print(correlation_matrix)
fig = plt.figure(figsize = (7,7)) ## helps to figure the size of matrix
sb.heatmap(correlation_matrix,annot=True,cmap='coolwarm') ##converts matrix into heatmap matrix
plt.title("Correlational Maatrix") ##Gives the title of matrix
plt.show() ## to dispay the matrix
st.pyplot(fig)


# converting data types
data.dropna(inplace=True)
data['Order Date'] = pd.to_datetime(data['Order Date'])## this converts the data type of order data into date time data type
print(data.info())

# putting month and year o order date
data['month'] = data['Order Date'].dt.month
data['year'] = data['Order Date'].dt.year
print(data.head())

# describing the data
print(data.describe())

# performing exploratory data analysis (EDA)
# a) using time series charts like axis graphs
monthly_sales = data.groupby(['year', 'month'])['Sales'].sum().reset_index() ##this line groups the data into year and month and sums thier values and then resets it do that we can plot it properly
print("::m", monthly_sales)
fig2 = plt.figure(figsize=(14,7))
sb.lineplot(data=monthly_sales, x="month", y="Sales", hue="year")
plt.title("Monthly Sales Report")
plt.show()
st.pyplot(fig2)

print(monthly_sales) 
# from this we get a line chart
# b) using bar and pie chart
fig3=plt.figure(figsize=(12,6))
sb.barplot(data=data, x ='Category', y='Sales', hue='Region')
plt.title('Category wise Sales by Region')
plt.show()
st.pyplot(fig3)


region_sales = data.groupby('Region') ['Sales'].sum()
fig4, ax = plt.subplots()
ax.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%')
ax.set_title('Sales Distribution by Region')
# plt.pie(region_sales,labels=region_sales.index,autopct='%1.1f%%')
# plt.title('Sales by Region')
# plt.show()
st.pyplot(fig4)

# c) Scatter plot
fig5=plt.figure(figsize=(8,6))
sb.scatterplot(data=data, x='Sales',y='Profit',hue='Segment')
plt.title("Sales vs Profit by Customer Segment")
plt.show()
st.pyplot(fig5)


# d) performance analysis
data.columns = data.columns.str.strip()
# print(data.columns.tolist())
product_performance = data.pivot_table(values='Sales',index='Category',columns='Sub-Category',aggfunc='sum')
fig6=plt.figure(figsize=(12,8))
sb.heatmap(product_performance,cmap='YlGnBu')
plt.title('Product Performance Heatmap')
plt.show()
st.pyplot(fig6)

# e) Hypothesis testing and statistical analysis
region1 = 'East'
region2 = 'South'

threshold = 0.05
region1_sales = data[data['Region'] == region1] ['Sales']
region2_sales = data[data['Region'] == region2] ['Sales']
t_stat, p_val = stats.ttest_ind(region1_sales, region2_sales)
print(f'p-value = {p_val}')

if(p_val < threshold):
    print("Reject null hypothesis, there is significant difference between two regions")
elif(p_val > threshold):
    print("Did not reject null hypothesis, there is no significant difference between two regions")

print(data)
print(num_data.corr())
print(data.describe())

sb.histplot(data=data, x='Discount', bins=20, kde=True)

## starting the machine learning part
## starting randomforest to forecast  the future sales values
#use montly sales data and create a feature set
monthly_sales['year_month'] = monthly_sales['year']*100 + monthly_sales['month'] ## this line of code will turn the date  like jan 2025 to 202501
# create feature and target
x = monthly_sales[['year','month','year_month']] ## this is feature to use
y = monthly_sales['Sales']
#  use train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# now we train the random forest regressor
model = RandomForestRegressor(n_estimators=100,random_state=42) ## n_estimators = 100 means the sellection will be going through 100 decision trees
model.fit(x_train,y_train)
# predict the model and evaluate
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
st.write(f"Model RMSE: {rmse:.2f}")
# now we predict the future sale values of next 6 months
last_year = monthly_sales['year'].max()
last_month = monthly_sales['month'].max()
future_dates = []
for i in range(1,7):
    month = (last_month + i)%12
    year = last_year + (last_month + i-1)//12
    if month == 0:
        month = 12
    future_dates.append({'year' : year , 'month' : month, 'year_month' : year*100+month})

future_df = pd.DataFrame(future_dates)
future_sales_pred = model.predict(future_df)
# visualization of randomforestregresser
fig_forecast = plt.figure(figsize=(14,7))
sb.lineplot(data=monthly_sales,x='year_month',y='Sales',label = 'Historical Sales')
plt.plot(future_df['year_month'],future_sales_pred,color = "red", marker = "o", label = "Predicted Sales")
plt.title("Retail Sales Forecast")
plt.xlabel("YearMonth")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(fig_forecast)

# we are adding clustering algorith for customer segmentation
customer_data = data.groupby('Customer ID').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

x = customer_data[['Sales', 'Profit']]
# we normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
# apply KMeans algorithm
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)
# visualize the kmeans 
fig_cluster = plt.figure(figsize=(10, 6))
sb.scatterplot(data=customer_data, x='Sales', y='Profit', hue='Cluster', palette='Set2', s=100, alpha=0.7)
plt.title('Customer Segmentation Based on Sales and Profit')
plt.xlabel('Total Sales')
plt.ylabel('Total Profit')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

st.pyplot(fig_cluster)

# Show cluster counts
st.write(customer_data['Cluster'].value_counts())


