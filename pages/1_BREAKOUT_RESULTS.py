#import libraries

import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')

#functions to use in the later part
##check if the given date is business day or not
def is_business_day(date):
  '''
  This function checks if the given date is a business day or not
  '''
  return pd.to_datetime(date).weekday()<5

def get_selling_date_and_close(df,buy_row,holding_period):
  '''
  This function returns the selling date for a given buy date and holding period based on the original dataframe
  '''
  try:
    if debug:st.write('using get selling date and close!')
    if debug:st.write(f'buy_row: {buy_row}')
    buy_index=df[df['row_key']==buy_row].index
    #

    #if len(buy_index) is 0
    if len(buy_index)==0:return None,None
    buy_index=buy_index[0] #first matching

    if debug:st.write(f'buy_index: {buy_index}')
    sell_index=buy_index+holding_period
    if debug:st.write(f'sell_index: {sell_index}')
    sell_index=buy_index+holding_period

    if sell_index>=len(df):return None, None

    sell_date=df.loc[sell_index,'Date']
    sell_price=df.loc[sell_index,'Close']

    if debug:st.write(f'sell_date: {sell_date}')
    if debug:st.write(f'sell_price: {sell_price}')
    return sell_date,sell_price
  except:
    return None,None

# def get_df_between_buy_sell(buy_date,sell_date):
#   '''
#   This function returns the dataframe between the buy date and sell date
#   '''
#   temp_df=df[(df['Date']>=buy_date)&(df['Date']<=sell_date)]
#   return temp_df

#user input ticker, start_date, end_date,volume_threshold%,%change on the end date,holding period
debug=False
#ask the user for the ticker
user_ticker=st.sidebar.text_input("Enter a ticker",value='TSLA',key='ticker').upper()

if debug:st.write(f'ticker: {user_ticker}')

#start and end date
start_date=st.sidebar.date_input('Select start business date',value=pd.to_datetime("2024/01/02"))
if debug:st.write(f'start_date: {start_date}')

#check if the start date is a business day
if not is_business_day(start_date):
   st.warning('Start date needs to be a valid business day !',icon='⚠️')
   st.stop()



#last_business_day
date_today=pd.to_datetime('today').normalize().date()
if debug:st.write(f'today: {date_today}')
bdate=pd.bdate_range(end=date_today,periods=1)[0]
if debug:st.write(f'last_business_day: {bdate}')
end_date=st.sidebar.date_input('Select end business date',value=bdate)
if debug:st.write(f'end_date: {end_date}')

if not is_business_day(end_date):
   st.warning('End date needs to be a valid business day !',icon='⚠️')
   st.stop()

#yfinance is exclusive for end date so making it inclusive
temp_end_date=pd.to_datetime(end_date)+pd.Timedelta(days=1)

if debug:st.write(f'temp_end_date: {temp_end_date}')
#assert that end date is later than the start date
if end_date<start_date:
   st.warning('End date needs to be later than Start date !',icon='⚠️')
   st.stop()

#downloading the stock values between start and end date 
try:
    df_temp=yf.download(user_ticker,start=start_date,end=temp_end_date,group_by='ticker')
except:
   st.warning('Error occured try again with valid ticker !',icon='⚠️')
   st.stop()

if df_temp.empty:
   st.warning('Error occured try again with a valid ticker !',icon='⚠️')
   st.stop()
#reset to keep the date as the column
#might need to change this part for the remote

if debug:st.write(f'before resetting index ...{df_temp.columns}')

if debug:st.write(df_temp)
#if debug:st.write(df_temp.columns)
  
remote=True
if remote:
   df=df_temp[user_ticker]
else:
   df=df_temp.copy()

#df_temp=df_temp.reset_index(drop=False)
df=df.reset_index(drop=False)
#changing Date into Datetime
#df_temp['Date']=pd.to_datetime(df_temp['Date']) #in remote
df['Date']=pd.to_datetime(df['Date']) #in remote
if debug:st.write(f'after resetting index df columns...{df.columns}')

#might need to change this part for the remote
#remote=False
#if remote:
#   df=df_temp[user_ticker]
#else:
#   df=df_temp.copy()

#giving unique name to each row to access the corresponding dates later
df['row_key']=range(1,len(df)+1)
#difference between start and end date
days_diff=int((end_date-start_date).days)

if debug:st.write(f'Differences between end and start date: {days_diff}')

holding_time=st.sidebar.number_input('Enter holding business days',value=10,min_value=0,max_value=days_diff,format='%d')

if debug:st.write(f'holding days {holding_time}')
#st.stop()

#ensuring that the holding time is less than the difference between start and end date
if float(holding_time) > days_diff:
   st.warning('Holding days  higher than the difference of  end_date and start_date',icon='⚠️')
   st.stop()


#volume threshold percentage
volume_threshold=st.sidebar.number_input("Enter volume threshold %",value=20.0,min_value=0.0)
if debug:st.write(f'volume_threshold: {volume_threshold}')
#volume threshold percentage
pct_threshold=st.sidebar.number_input("Enter change threshold %",value=2.0,min_value=0.0)
if debug:st.write(f'pct_threshold: {pct_threshold}')
if debug:st.write(f'before adding columns: {df.columns}')
if debug:st.write(df.head())
#st.stop()

#get daily change % and 20 day average volume
df.loc[:,'daily_change%']=df['Close'].copy().pct_change()*100
df.loc[:,'volume_average_20_days']=df['Volume'].copy().rolling(window=20).mean()
df.loc[:,'volume_condition']=df['Volume'].copy()>df['volume_average_20_days'].copy()*(100+volume_threshold)/100
df.loc[:,'percent_change_condition']=df['daily_change%'].copy()>pct_threshold
df.loc[:,'buy_condition']=(df['volume_condition'].copy() & df['percent_change_condition'].copy())

#filter the dataframe with df_buy
#BUY dataframe here onwards

df_buy=df.copy()[df.copy()['buy_condition']==True]
if df_buy.empty:
   st.warning('NO DATA FOR GIVEN CONDITION',icon='⚠️')
   st.stop()

#NOTE:resetting the columns to get Date column, otherwise it is interpreted as the index by streamlit
df_buy.reset_index(drop=False,inplace=True)
if debug:st.write(f'df_buy_columns: {df_buy.columns}')
if debug:st.write(f'df_buy: {df_buy}')
st.markdown(f"<h4 Style='text-align:center;'>RESULTS FOR GIVEN CONDITIONS FOR {user_ticker.upper()}</h4>",unsafe_allow_html=True)
#getting selling date and price
#Need to fix this part
#find the date original dataframe
#test_df=df_buy['Date']
#st.write(f'test_df: {test_df}')
#df_buy.loc[:,'selling_date']=df_buy['row_key'].copy().map(lambda buy_row:get_selling_date_and_close(df,buy_row,holding_time)[0])
df_buy.loc[:,'selling_date']=df_buy['row_key'].apply(lambda buy_row:get_selling_date_and_close(df,buy_row,holding_time)[0])
df_buy.loc[:,'selling_price']=df_buy['row_key'].copy().map(lambda buy_row:get_selling_date_and_close(df,buy_row,holding_time)[1])
df_buy.loc[:,'return(%)']=(df_buy['selling_price'].copy()/df_buy['Close'].copy()-1)*100
df_buy.loc[:,'mean_return(%)']=df_buy['return(%)'].mean()
df_buy.insert(1,'ticker',user_ticker)

#change the volume into millions
df_buy['Volume']=df_buy['Volume'].div(1e6)
df_buy['volume_average_20_days']=df_buy['volume_average_20_days'].div(1e6)

#getting final df 
if debug:st.write('DF_BUY')
if debug:st.write(df_buy)
if debug:st.write(df_buy.columns)
if debug:st.write(f'index: {df_buy.index}')
#st.stop()

selected_columns=['ticker','Date','Close','selling_date','selling_price','return(%)','mean_return(%)','Volume','volume_average_20_days']
df_final=df_buy[selected_columns].reset_index(drop=True)
df_final=df_final.rename(columns={'Close':'buying_price','Date':'buying_date','Volume':'traded_volume(M)','volume_average_20_days':'avg_20days_vol(M)'}).round(2)

#modify the buying and selling date
#might need to change to datetime if needed
df_final['buying_date']=df_final['buying_date'].dt.date
df_final['selling_date']=df_final['selling_date'].dt.date
df_final['holding_days']=holding_time
df_final['volume_threshold(%)']=volume_threshold
df_final['last_change_threshold(%)']=pct_threshold
df_final.index=range(1,len(df_final)+1)
# df_final.index.name='S.N.'
# df_final.dropna(inplace=True)
#debug=True
#if debug:st.write(df_buy)
#color the values
def color_val(val):
   if val>0:return "color:green;"
   return "color:red;"

# Function to alternate row colors
def alternating_row_colors(row):
    return ['background-color: #F2F2F2' if row.name % 2 == 0 else 'background-color: white'] * len(row)
#df_fin=df_final.style.applymap(color_val,subset=['return(%)'])#.format('{:.2f}')
#df_fin=df_final.style.applymap(color_val,subset=['return(%)'])#.format('{:.2f}')
 # Style the DataFrame
#columns:
df_final.columns.name=''
# st.write(f'column_name:',df_final.columns.name)
# st.write(df_final.columns.to_list())
# st.stop()
#df_final.drop(columns=[('Level1','Price')],inplace=True) #remove Price1 from the Level1
df_final=df_final.reset_index(drop=True)
df_final.index=range(1,len(df_final)+1)
df_final.dropna(inplace=True)
#df_final.index.name='S.N.'
#style_dict={'text-align':'center','font-size':'14px'}
#df_final['holding_days']=df_final['holding_days'].astype(int)
style_dict = {
    'text-align': 'center',
    'font-family': 'Courier,monospace',
    'font-size': '16px',
   #'font-weight': 'bold',
   # 'color': 'green',
   # 'background-color': '#f0f0f0',
    'border': '0.5px solid black',
    #'border-radius': '5px',
    'padding': '5px'
}
df_fin = (
df_final.style
.format("{:.2f}", subset=df_final.select_dtypes(include=["float64"]).columns)
.apply(lambda row:row.apply(color_val), subset=["return(%)", "mean_return(%)"],axis=1)
.apply(alternating_row_colors,axis=1)
.set_properties(**style_dict)
)
#st.dataframe(df_final,use_container_width=True)

#container 1
with st.container():
 st.markdown(df_fin.to_html(),unsafe_allow_html=True)
#include download button
file_name=f'{user_ticker}.csv'
csv=df_final.to_csv()

st.sidebar.download_button(
   label='Download Result',
   data=csv,
   file_name=file_name,
   mime='text/csv'
)

#this part is for the matplotlib plot
x_ticks=df_final['buying_date'].to_list()
y_ticks=df_final['selling_date'].to_list()
custom_ticks=[f'BUY:{x}\nSELL:{y}' for x,y in zip(x_ticks,y_ticks)]
colors=['green' if x>0 else 'red' for x in df_final['return(%)']]
mean_return=df_final['return(%)'].mean()
#custom_ticks
if debug:st.write(custom_ticks)
# Plotting the bar chart
sns.set_theme()
fig, ax = plt.subplots(figsize=(15, 6))
ax = df_final['return(%)'].plot(kind='bar', ax=ax, color=colors)
ax.set_ylabel('return(%)')
ax.set_xlabel('buy/sell dates')
ax.axhline(y=mean_return, color='blue', linestyle='--')
ax.axhline(y=0, color='black', linestyle='-')
ax.bar_label(ax.containers[0], rotation=0, fontsize=8)
ax.set_xticklabels(custom_ticks, rotation=90)
#annotation
text_color='green' if mean_return>0 else 'magenta'
ax.annotate(f'mean_return: {mean_return:0.2f} %',
            xy=(0.5,0.5),
            bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'),  # Highlight with yellow background
            color=text_color,
    xycoords='axes fraction',  # Use axes fraction coordinates for the annotation
    textcoords='axes fraction'  # Use axes fraction coordinates for the text
            )

legend=ax.legend([f'mean_return(%)'])
ax.tick_params(axis='y',which='both',left=True,right=True,direction='in')
ax.tick_params(axis='x',which='both',top=True,bottom=True,direction='in')
title_text=f'{user_ticker} | start_date: {start_date} | end_date: {end_date} | holding_days: {holding_time} | volume_threshold%: {volume_threshold} | price_change_threshold%: {pct_threshold}'
plt.title(title_text)
#plt.grid(True)
# Render the plot in Streamlit
st.pyplot(fig)
