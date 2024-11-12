import streamlit as st
import pandas as pd
import plotly.express as px


def HBO():
    st.title("Page 1")
    st.write("This is the first page.")
    
df=pd.read_csv('data.csv')
df = df.drop_duplicates()

# Display initial data
st.write("### Preview of the Dataset")
st.write(df.head())
st.write("### Dataset Information")
st.write(df.info())

# Grouping data for analysis
grouped = df.groupby(['title', 'type', 'releaseYear', 'imdbId', 'imdbAverageRating', 'imdbNumVotes', 'availableCountries']).agg({'genres': 'sum'}).reset_index()

# User selections for x and y axes
st.markdown(
    """ 
    In this scatter plot, you can analyze different perspectives from the dataset by selecting various variables based on the information you need to examine.
    """
)
x_axis_column = st.selectbox('Select X-axis variable:', grouped.columns)
y_axis_column = st.selectbox('Select Y-axis variable:', grouped.columns)

# Create scatter plot
fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')
fig.update_layout(
    xaxis_title=x_axis_column,
    yaxis_title=y_axis_column
)

# Display plot
st.plotly_chart(fig)

# Optional: Display selected row information based on user input
if st.checkbox("Show Row Information"):
    # Create a sidebar or main section for selections
    option = st.selectbox("Choose an option to view:", ["Select by Title", "Select by Genres", "Select by imdbNumVotes"])
    
    if option == "Select by Title":
        selected_title = st.selectbox("Select a Title:", grouped['title'].unique())
        row_info = grouped[grouped['title'] == selected_title]
        st.write("### Selected Row Information")
        st.write(row_info)
    
    elif option == "Select by Genres":
        selected_genres = st.multiselect("Select genres:", grouped['genres'].unique())
        
        if selected_genres:
            row_info = grouped[grouped['genres'].isin(selected_genres)]
            st.write("### Selected Row Information")
            st.write(row_info)
        else:
            st.write("Please select at least one genre to see the row information.")

    elif option == "Select by imdbNumVotes":
        # Use a slider to select a range of imdbNumVotes
        min_votes, max_votes = int(grouped['imdbNumVotes'].min()), int(grouped['imdbNumVotes'].max())
        selected_range = st.slider("Select range of IMDb Num Votes:", min_value=min_votes, max_value=max_votes, value=(min_votes, max_votes))
        
        # Filter rows based on the selected range
        row_info = grouped[(grouped['imdbNumVotes'] >= selected_range[0]) & (grouped['imdbNumVotes'] <= selected_range[1])]
        
        st.write("### Selected Row Information")
        st.write(row_info)

def main():
    page = st.sidebar.selectbox("Navigation", ["Page 1", "Page 2", "Page 3"])

    if page == "HBO":
        HBO()
    elif page == "Page 2":
        page2()
    elif page == "Page 3":
        page3()

if __name__ == "__main__":
    main()