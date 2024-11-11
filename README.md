# 📈 **Hybrid Machine Learning Model for Predicting Apple's Stock Closing Price** 🍏

## 📊 **Project Overview** 
This project demonstrates the use of a **Hybrid Machine Learning Model** designed to predict **Apple's stock closing price** using historical stock data. The model combines two powerful techniques—**LSTM (Long Short-Term Memory)** networks and **Linear Regression**—to leverage both time-series pattern recognition and trend-following capabilities for accurate stock price prediction.

The dataset used in this project includes **Date**, **Open**, **High**, **Low**, **Close**, and **Volume** data for Apple's stock, with the main focus being on predicting the **Close Price** based on historical trends.

## 🧠 **Approach & Techniques**
### Hybrid Model:
The project adopts a hybrid approach by integrating two distinct models:
1. **LSTM (Long Short-Term Memory)**: 
   - LSTM is a specialized type of Recurrent Neural Network (RNN) capable of learning long-term dependencies in time-series data. 
   - **Why LSTM?** LSTM is ideal for sequential data like stock prices, as it remembers previous time steps and captures hidden patterns and trends over time.
   - In this project, LSTM is used to learn from the historical prices and capture complex temporal dependencies.

2. **Linear Regression**:
   - A simpler model compared to LSTM, Linear Regression helps model the relationship between the **Open**, **High**, **Low**, **Volume**, and the **Close Price**.
   - **Why Linear Regression?** While LSTM captures sequential patterns, Linear Regression provides a straightforward understanding of how various features are correlated with the target variable (Closing Price).
   - By combining both, we aim to leverage the strengths of each method to improve prediction accuracy.

### Data Preprocessing:
- **Handling Missing Data**: Missing values in the dataset were handled through imputation, ensuring the integrity of the data before feeding it into the model.
- **Normalization**: Features such as **Open**, **High**, **Low**, and **Volume** were normalized using MinMaxScaler to scale them within the range [0, 1]. This helped the models converge faster and improve prediction accuracy.
- **Feature Engineering**: Additional time-based features like **Day of the Week**, **Month**, and **Year** were extracted to improve the model's ability to recognize trends over time.

### Model Training:
- **LSTM Model**: The LSTM model was trained using a sequence of past stock prices to predict future prices. We used a sliding window approach, where past stock data was used to predict the next closing price.
- **Linear Regression Model**: This was trained on features such as **Open**, **High**, **Low**, and **Volume**, which were selected based on their relevance to predicting the closing price.
- **Hybrid Model**: The outputs of both models were combined, with the final prediction being based on a weighted average or the best-performing model's output.

## 🚀 **Key Features**
- **Hybrid Model Architecture**: A combination of LSTM and Linear Regression to predict stock prices.
- **Time-Series Data Handling**: Handling of sequential data using LSTM, which is critical for stock market predictions.
- **Feature Engineering**: Inclusion of time-based features to enhance the model’s performance.
- **Data Preprocessing**: Techniques like missing data imputation and feature normalization for clean input.

## 📈 **Visualizations**
Visualization is key to understanding how well the model is performing and how it captures trends. Here's a list of the visualizations created in this project:

1. **Stock Price Over Time**: 
   - A line chart was used to visualize the historical Apple stock closing prices over time, helping identify patterns, trends, and outliers in the data.
   - **Tool Used**: Matplotlib, Seaborn.

2. **Model Performance vs. Actual Prices**: 
   - A **line plot** comparing the predicted closing prices from the hybrid model with the actual closing prices.
   - This visualization allows us to see how closely the model’s predictions align with the actual data, indicating the model's performance over time.
   - **Tool Used**: Matplotlib.

3. **Loss Function Curve**:
   - A plot showing the **training and validation loss** curves for the LSTM model. This visualization helps in understanding how well the model is learning over time.
   - It is particularly useful to spot overfitting (if the validation loss diverges from the training loss).
   - **Tool Used**: Matplotlib.

4. **Correlation Heatmap**:
   - A heatmap showing the correlation between different features (such as **Open**, **High**, **Low**, and **Volume**) with the **Close Price**.
   - This helps in understanding how strongly different features relate to the target variable, aiding feature selection.
   - **Tool Used**: Seaborn.

5. **Model Evaluation Metrics**:
   - Visualization of **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-squared** values for the model’s performance. These metrics provide a clear indication of the model’s accuracy.
   - **Tool Used**: Matplotlib, Seaborn.

6. **Prediction vs. Actual Plot**: 
   - A **scatter plot** showing the predicted closing prices vs. the actual closing prices.
   - This plot is used to visually assess how well the model’s predictions align with the real-world data.
   - **Tool Used**: Matplotlib.

## ⚙️ **Technologies Used**
- **Python** 🐍
- **Pandas** 📚
- **Numpy** 🔢
- **Matplotlib** & **Seaborn** 📊 (for data visualization)
- **Keras** & **TensorFlow** 🤖 (for building the LSTM model)
- **Scikit-learn** 📉 (for Linear Regression and model evaluation)
- **Jupyter Notebooks** 📓 (for interactive analysis)

## 🛠️ **Getting Started**
To run this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/PranayaKD/Hybrid-Model-Stock-Price-Prediction.git
cd Hybrid-Model-Stock-Price-Prediction

## 📈 **Results & Evaluation**
The hybrid model successfully combines the benefits of LSTM and Linear Regression, achieving high prediction accuracy for Apple's stock closing price. The evaluation metrics (MAE, RMSE) indicate that the hybrid model performs better than traditional time-series methods, providing more reliable predictions.

The **visualizations** highlight how well the model tracks the actual stock prices and help demonstrate the impact of time-based features and historical trends on the model’s accuracy.

## 💡 **What Makes This Project Stand Out?**
- The **Hybrid Approach** combining LSTM and Linear Regression leverages both deep learning and traditional modeling techniques for more robust stock price predictions.
- The project demonstrates advanced techniques in **time-series analysis**, **data preprocessing**, and **model evaluation**.
- **Clear visualizations** provide valuable insights into the model's performance and data trends, making the predictions more understandable.

## 📝 **Conclusion**
By combining **LSTM's time-series capabilities** with **Linear Regression’s trend analysis**, this project demonstrates an effective approach to forecasting stock prices. It’s a valuable resource for anyone looking to explore **hybrid models**, **time-series analysis**, and **machine learning** for stock market prediction.

## 🔗 **Links**
- [Dataset Source](https://statso.io/building-hybrid-models-case-study/) 📎
- [GitHub Repository](https://github.com/PranayaKD/Hybrid-Model-Stock-Price-Prediction) 💻

## 📢 **Future Improvements**
- Experiment with other machine learning techniques such as **XGBoost**, **Random Forest**, and **Gradient Boosting** to further improve the model’s accuracy.
- Incorporate additional external factors like **financial news sentiment analysis** and **global economic indicators** for more accurate stock forecasting.
- Develop a **real-time prediction system** using **live API data** for dynamic forecasting.

## 🤝 **Contributing**
Feel free to open an issue or submit a pull request if you'd like to contribute. Contributions are welcome! ✨
