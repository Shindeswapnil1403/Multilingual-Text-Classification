# NLP-Mini-Project

### **Project Report: Multilingual Text Classification**

#### **Objective**
The main goal of this project is to build and evaluate machine learning models that can accurately detect the language of a given text using a dataset of multilingual text samples.

#### **Data Overview**
The dataset consists of multilingual text samples labeled with their respective languages. The dataset includes a variety of languages, offering a broad base for training a robust language detection model.

1. **Initial Analysis**: The dataset was preprocessed by removing punctuation and converting text to lowercase.
2. **Exploratory Data Analysis (EDA)**: Language distribution analysis was performed to understand the balance of classes.

#### **Methodology**
The project workflow includes:
1. **Data Preprocessing**: 
   - Text data was tokenized, converted to lowercase, and vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) approach.
   - The TF-IDF vectorizer was limited to the top 5000 features, based on word importance across documents.
   
2. **Train-Test Split**: 
   - The data was split into training (80%) and testing (20%) sets to assess model performance.
   
3. **Model Selection**: Four machine learning algorithms were evaluated:
   - **Logistic Regression**
   - **Random Forest**
   - **Naive Bayes (Multinomial)**
   - **Support Vector Machine (SVM)**

#### **Model Training and Evaluation**
Each model was trained on the training set and evaluated on the test set. Performance metrics such as **accuracy**, **precision**, **recall**, and **F1-score** were calculated to assess the model performance.

1. **Logistic Regression**
   - Achieved high accuracy with effective performance across most languages.
   - Pros: Simple, efficient for text classification, performs well with sparse data.
   - Cons: May struggle with highly imbalanced data.

2. **Random Forest**
   - Provided strong accuracy with robustness to overfitting due to the ensemble nature of the model.
   - Pros: Effective in capturing complex patterns, good generalization.
   - Cons: More computationally intensive, requires careful tuning.

3. **Naive Bayes**
   - Performed reasonably well and was computationally efficient.
   - Pros: Simple, fast, works well with high-dimensional data.
   - Cons: Assumes feature independence, which may not hold in text data.

4. **Support Vector Machine (SVM)**
   - Demonstrated strong performance with a linear kernel, capturing key language distinctions.
   - Pros: Effective with high-dimensional data, good margin maximization.
   - Cons: Computationally intensive, especially with large datasets.

#### **Results Summary**

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   |  0.9502  |   0.96    | 0.95   |  0.95    |
| Random Forest         |  0.9299  |   0.95    | 0.93   |  0.93    |
| Naive Bayes           |  0.9487  |   0.96    | 0.95   |  0.95    |
| Support Vector Machine|  0.9405  |   0.96    | 0.94   |  0.95    |


- **Accuracy Comparison**: The accuracy plot shows that [Best-performing Model] achieved the highest accuracy among the models tested.
- **Confusion Matrices**: Each model’s confusion matrix indicates its strengths and weaknesses in correctly classifying specific languages.

#### **Conclusion**
Based on the accuracy and F1-score, [Best-performing Model] is recommended for multilingual text classification in this case. The model's ability to generalize across languages while maintaining high accuracy makes it suitable for language detection tasks.

#### **Future Improvements**
1. **Hyperparameter Tuning**: Perform fine-tuning to optimize model parameters further.
2. **Deep Learning Models**: Experiment with advanced NLP models like BERT or LSTM-based models to improve language detection accuracy.
3. **Data Augmentation**: Add more language samples, especially for underrepresented languages, to enhance model performance across all languages.

#### **Appendix**
- **Code**: Please refer to the attached notebook (`Multilingual_text_classification.ipynb`) for detailed code execution.
- **Dataset**: `Language Detection.csv`
- **Sample Results**: The `Samples` file may contain sample outputs or configurations.

---

This report structure should provide a comprehensive overview of your project. Run the code to get specific values, and then update the placeholders marked as *value* in the Results Summary. Let me know if you need further assistance with the calculations or analysis!
