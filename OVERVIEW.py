import streamlit as st


#page configuration
st.set_page_config(layout='wide')

disclaimer_text="""
Disclaimer:
The information provided by this backtesting program is for informational purposes only and should not be considered as financial advice, investment advice, or trading suggestions. All data and analysis are provided "as is" and without any guarantees of accuracy or completeness. Users are solely responsible for their investment decisions. Always conduct your own research and consult with a licensed financial advisor before making any investment decisions.
"""
st.warning(disclaimer_text,icon="⚠️")

overview_text="""

The Backtesting App provides results for holding a ticker over a specified number of days, based on two conditions: when the volume on a given day exceeds a certain percentage above the 20-day average volume (volume threshold %) and when the percentage change on that day surpasses a defined value (change threshold %).

Inputs: <br>

1. Ticker
2. Start business day
3. End business day
4. Holding business days
5. Volume threshold %
6. Change threshold %

Outputs:

1. Downloadable data table with buying date/price, selling date/price, return %, and mean return %
2. Bar graph displaying the corresponding results

"""
st.markdown(f"{overview_text}",unsafe_allow_html=True)