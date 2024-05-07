# Interactive-Machine-Learning-Classifier-Comparison-Web-Application-Using-Streamlit
![ml_photo](https://github.com/Abdalrahmanhassan237/Interactive-Machine-Learning-Classifier-Comparison-Web-Application-Using-Streamlit/assets/158060043/66542ca3-8dc0-4eca-97a9-c73b2610d9c3)

Streamlit application that allows you to explore different machine learning classifiers and compare their performance on different datasets. Let's go through the code step by step:

1. The necessary libraries are imported, including Streamlit for creating the web interface, as well as various machine learning libraries such as NumPy, Matplotlib, pandas, and scikit-learn.

2. An image is displayed at the top using the `st.image()` function. The image is loaded from a file called "ml_photo.jpg" located in the same directory as the script.

3. The title and a brief introduction are displayed using the `st.title()` and `st.write()` functions.

4. The user can select the dataset and classifier from the sidebar using the `st.sidebar.selectbox()` function. The available datasets are breast cancer, diabetes, and iris. The available classifiers are Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest.

5. The `load_dataset()` function is defined to load the selected dataset. Depending on the dataset name, the appropriate dataset is loaded from scikit-learn's `datasets` module. The function returns the input features `x` and the target variable `y`.

6. The selected dataset's name and shape are displayed using the `st.write()`.

7. The `add_parameter()` function is defined to add parameters for the selected classifier. The function checks the selected classifier name and adds the appropriate parameter based on the choice. For example, if the classifier is KNN, a parameter for the number of neighbors (`k`) is added. The function returns a dictionary of parameters.

8. The `add_classifier()` function is defined to create an instance of the selected classifier with the provided parameters. The function checks the classifier name and initializes the classifier accordingly. The function returns the classifier object.

9. The parameters for the selected classifier are added using the `add_parameter()` function.

10. The classifier is created using the `add_classifier()` function.

11. The dataset is split into training and testing sets using the `train_test_split()` function from scikit-learn.

12. The classifier is trained on the training data using the `fit()` method.

13. The classifier makes predictions on the testing data using the `predict()` method.

14. The accuracy of the classifier is calculated using the `accuracy_score()` function from scikit-learn.

15. The accuracy score is displayed using `st.write()`.

16. Principal Component Analysis (PCA) is performed on the dataset to reduce its dimensionality.

17. A scatter plot is created using Matplotlib to visualize the data in the reduced-dimensional space. Each point is colored according to its class label. The plot is displayed using `st.pyplot()`.

Overall, this code provides an interactive interface to explore different classifiers and visualize their performance on different datasets using Streamlit. It allows users to dynamically select datasets, classifiers, and adjust classifier parameters to observe the results in real-time.
