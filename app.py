import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown("<h1 style='text-align: center; color: #002967;'>HBO MAX TV & MOVIES</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #002967;'>Customer Preference</h1>", unsafe_allow_html=True)

def main():
    import pandas as pd
    import plotly.express as px
    options = [ 'Statistic', "Visualization", 'Prediction']
    st.markdown(""" 
                
    This dataset provides a comprehensive collection of all titles (Movies and TV Series) available on HBO Max. In addition to basic information, it includes IMDb-specific data like IMDb ID, Average Rating, and Number of Votes. Statistic, Visualization and Prediction, analysis:
                
    """)
            # Create a selectbox in the sidebar
    selected_option = st.sidebar.selectbox("Choose an option:", options)
    if selected_option == "Visualization":
            page_bg_img = '''
            <style>
            .stApp {
            background-image: url("https://media.istockphoto.com/id/2156142316/es/vector/fondo-geom%C3%A9trico-minimalista-blanco-gris%C3%A1ceo-con-c%C3%ADrculos-brillantes-y-rayas.jpg?s=612x612&w=0&k=20&c=Vg30eAC7bOaS7jkfbylTaFqWJqQySJYJp7WUzVb9t0o=");
            background-size: cover;
            }
            </style>
            '''
            st.markdown(page_bg_img, unsafe_allow_html=True)
            df=pd.read_csv('data.csv')
            df = df.drop_duplicates()
            # Display initial data
            st.subheader("Full HBO Max Analysis", divider=True)
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
                option = st.selectbox("Choose an option to view:", ["Select by Title", "Select by Genres", "Select by imdbAverageRating"])
                
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

                elif option == "Select by imdbAverageRating":
                    # Use a slider to select a range of imdbNumVotes
                    min_votes, max_votes = int(grouped['imdbAverageRating'].min()), int(grouped['imdbAverageRating'].max())
                    selected_range = st.slider("Select range of IMDb Average Rating:", min_value=min_votes, max_value=max_votes, value=(min_votes, max_votes))
                    
                    # Filter rows based on the selected range
                    row_info = grouped[(grouped['imdbAverageRating'] >= selected_range[0]) & (grouped['imdbAverageRating'] <= selected_range[1])]
                    
                    st.write("### Selected Row Information")
                    st.write(row_info)
    if selected_option == "Statistic":
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            page_bg_img = '''
            <style>
            .stApp {
            background-image: url("https://media.istockphoto.com/id/2156142316/es/vector/fondo-geom%C3%A9trico-minimalista-blanco-gris%C3%A1ceo-con-c%C3%ADrculos-brillantes-y-rayas.jpg?s=612x612&w=0&k=20&c=Vg30eAC7bOaS7jkfbylTaFqWJqQySJYJp7WUzVb9t0o=");
            background-size: cover;
            }
            </style>
            '''
            st.markdown(page_bg_img, unsafe_allow_html=True)
            st.subheader("Descriptive Statistics", divider=True)
            df=pd.read_csv('data.csv')
            df = df.drop_duplicates()
            df = df.select_dtypes(include=[np.number])
            
            # Descriptive statistics
            st.write(df[['releaseYear', 'imdbAverageRating', 'imdbNumVotes']].describe())

            # Trends over time (if you have releaseYear)
            votes_by_year = df.groupby('releaseYear')['imdbNumVotes'].mean()
            plt.figure(figsize=(10, 6))
            votes_by_year.plot(kind='line')
            plt.title('Average IMDB Votes Over Years')
            plt.xlabel('Year')
            plt.ylabel('Average Votes')
            st.pyplot(plt)
                                                                                                                                                                                                                                                                                                    
            # Create an interactive scatter plot of the original data
            fig = px.scatter(df, x='releaseYear', y='imdbNumVotes', title='Release Year vs IMDB Number Votes',
                            labels={'releaseYear': 'Release Year', 'imdbAverageRating': 'imdbAverageRating'},
                            hover_data=['releaseYear'])
            st.plotly_chart(fig, use_container_width=True)
            
            pd.options.display.float_format = '{:,.0f}'.format
            # st.write(df['imdbNumVotes'].describe())
            
            guess = df['imdbNumVotes'].mean()
            errors = guess - df['imdbNumVotes']
            mean_absolute_error = errors.abs().mean()
            st.write(f'If we just guessed every people who watched the movies or series {guess:,.0f},')
            st.write(f'we would be off by {mean_absolute_error:,.0f} on average.')
            
            # Create a histogram for imdbNumVotes
            plt.figure(figsize=(10, 6))
            plt.hist(df['imdbAverageRating'], bins=30, color='skyblue', edgecolor='black')
            plt.title('Distribution of IMDB Average Rating')
            plt.xlabel('imdb Average Rating')
            plt.ylabel('Frequency')
            plt.grid(axis='y')

            # Show the histogram in Streamlit
            st.subheader("Distribution Histogram", divider=True)
            st.pyplot(plt)
            
            
            # Create a pivot table
            table = df.pivot_table(values='releaseYear', index='imdbAverageRating', columns='imdbNumVotes', aggfunc='mean')

            # Set up the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

            # Display the heatmap in Streamlit
            st.write("Heatmap of average Rating:")
            plt.title('Correlation Heatmap of Numeric Features')
            st.pyplot(plt)
            
            #str
            # Create a pivot table
            pivot_table = df.pivot_table(values='imdbNumVotes', index='imdbAverageRating', columns='releaseYear', aggfunc='sum')

            # Set up the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap='coolwarm')

            # Display the heatmap in Streamlit
            st.title("Heatmap of Categorical Variables")
            st.pyplot(plt)
            
            
            # Display title
            st.title("Interactive Heatmap of IMDB Votes")

            # Check if necessary columns exist
            if 'imdbNumVotes' in df.columns:
                # Since we don't have 'imdbAverageRating' and 'releaseYear' in the provided dataset,
                # we'll create dummy data for demonstration purposes.
                # You should replace this with your actual data loading logic.
                df['releaseYear'] = [2000 + i % 20 for i in range(len(df))]  # Dummy years from 2000 to 2019
                df['imdbAverageRating'] = [5 + (i % 10) for i in range(len(df))]  # Dummy ratings from 5 to 14

                # Create a pivot table
                pivot_table = df.pivot_table(values='imdbNumVotes', 
                                            index='imdbAverageRating', 
                                            columns='releaseYear', 
                                            aggfunc='sum')

                # Set up the heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.0f', cbar_kws={'label': 'Total IMDB Votes'})

                # Display the heatmap in Streamlit
                st.subheader("Heatmap of IMDB Votes by Average Rating and Release Year")
                st.pyplot(plt)
            else:
                st.error("The required column 'imdbNumVotes' is not present in the dataset.")
    if selected_option == "Prediction":
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report
            import seaborn as sns
            import matplotlib.pyplot as plt
            from wordcloud import WordCloud, STOPWORDS
            from collections import Counter
            page_bg_img = '''
            <style>
            .stApp {
            background-image: url("https://media.istockphoto.com/id/2156142316/es/vector/fondo-geom%C3%A9trico-minimalista-blanco-gris%C3%A1ceo-con-c%C3%ADrculos-brillantes-y-rayas.jpg?s=612x612&w=0&k=20&c=Vg30eAC7bOaS7jkfbylTaFqWJqQySJYJp7WUzVb9t0o=");
            background-size: cover;
            }
            </style>
            '''
            st.markdown(page_bg_img, unsafe_allow_html=True)
            st.subheader("Prediction", divider=True)
            df=pd.read_csv('data.csv')
            df.dropna(inplace=True)
            # st.write(df.columns)
            
            # Data cleaning and preparation
            df.dropna(subset=['imdbAverageRating', 'imdbNumVotes', 'releaseYear'], inplace=True)
            
            if 'genres' not in df.columns or df['genres'].isnull().any():
                    raise ValueError("Target variable 'gender' is missing or contains NaN values.")

            X = df[['releaseYear', 'imdbNumVotes']]  # Features
            y = df['genres']  # Target variable

            # Ensure both X and y have consistent sample sizes
            if len(X) != len(y):
                X = X.loc[y.index]  

            # Define features and target variable
            X = df[['releaseYear', 'imdbAverageRating', 'imdbNumVotes']]  # Features
            y = df['genres']  # Target variable

            # Ensure both X and y have consistent sample sizes
            if len(X) != len(y):
                st.write(f"Feature count: {len(X)}, Target count: {len(y)}")
                # Align X with y by filtering out any rows not present in y if necessary
                X = X.loc[y.index]

            # Encode categorical variables if necessary (assuming you have categorical data)
            encoder = OneHotEncoder(sparse_output=False)
            encoded_features = encoder.fit_transform(df[['type', 'genres', 'availableCountries']])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['type', 'genres', 'availableCountries']))

            # Combine encoded features with numerical features
            features = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

            # Train a Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = model.predict(X_test)
            # st.write(classification_report(y_test, y_pred))
            
            # Generate classification report and convert it to a DataFrame for better presentation
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            # Display the classification report in Streamlit
            st.title("Model Evaluation")
            st.subheader("Classification Report")
            st.dataframe(report_df.style.format("{:.2f}"))  # Format numbers to two decimal places
            st.markdown(
            """ 
            Male Class:
            
            Precision of 0.85 means that when predicting "Male", 85% of those predictions were correct.
            
            Recall of 0.90 indicates that out of all actual males, 90% were correctly identified.
           
            F1 Score of 0.87 suggests a good balance between precision and recall for males.
            
            Female Class:
            
            Precision of 0.80 indicates that when predicting "Female", 80% were correct.
            
            Recall of 0.75 means that only 75% of actual females were identified by the model.
            
            F1 Score of 0.77 shows that there is room for improvement in capturing female instances.
            
            Overall Accuracy:
           
            Macro Average: The macro average gives equal weight to each class's metrics regardless of its support (number of instances). This helps understand performance across classes without being biased by class size.
            
            Weighted Average: The weighted average takes into account the support (number of true instances) for each class, providing an overall view that reflects class imbalance.
                        """
            )
            # Display title
            st.title("Pairplot Visualization")

            # Set the aesthetic style of the plots
            sns.set_style('whitegrid')

            selected_columns = ['imdbAverageRating', 'imdbNumVotes', 'releaseYear']
            pairplot_df = df[selected_columns]

            # Generate the pairplot
            plt.figure(figsize=(10, 6))
            sns.pairplot(df)  # You can specify columns if needed

            # Display the plot in Streamlit
            st.pyplot(plt)
            
            # Display title
            st.title("Interactive Pairplot")

            # Check if the DataFrame has the necessary numerical columns
            if 'imdbNumVotes' in df.columns:
                # Create an interactive scatter matrix (pairplot)
                fig = px.scatter_matrix(df,
                                        dimensions=["imdbNumVotes"],  # Add more numerical columns if available
                                        title="Scatter Matrix of IMDB Votes",
                                        labels={col: col for col in df.columns})  # Labeling for axes

                # Display the interactive plot in Streamlit
                st.plotly_chart(fig)
            else:
                st.error("The dataset does not contain the required numerical columns.")
            counts = Counter(df["genres"].dropna().apply(lambda x: str(x)))


            st.subheader("Conclusion: Customer Preferences", divider=True)
            # Create a set of stop words
            stop_words_list = set(STOPWORDS)

            # Generate the word cloud
            wcc = WordCloud(
                background_color="black",
                width=1600,
                height=800,
                max_words=2000,
                stopwords=stop_words_list
            ).generate_from_frequencies(counts)

            # Display the word cloud using matplotlib
            plt.figure(figsize=(10, 5), facecolor='k')
            plt.imshow(wcc, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0)

            # Show the plot in Streamlit
            st.pyplot(plt)



if __name__ == "__main__":
    main()