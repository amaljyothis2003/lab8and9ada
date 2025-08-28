# 🏥 Diabetic Patient Readmission Analysis

A comprehensive machine learning application for predicting hospital readmissions in diabetic patients using Random Forest and Logistic Regression models.

## 📋 Project Overview

This Streamlit application demonstrates a complete data science pipeline for healthcare analytics, featuring:

- **Data Preprocessing**: Advanced missing value imputation and feature engineering
- **Exploratory Data Analysis**: Interactive visualizations and statistical insights
- **Machine Learning Models**: 
  - Random Forest Classification (3-class prediction)
  - Improved Random Forest with feature selection
  - Logistic Regression (binary classification)
- **Clinical Insights**: ADA guidelines interpretation and healthcare recommendations

## 🚀 Live Demo

🌐 **[View Live Application](https://your-app-url.streamlit.app)**

## 🔧 Features

### 1. Dataset Analysis
- Comprehensive data overview with missing value analysis
- Interactive data type and statistical summaries
- Target variable distribution visualization

### 2. Data Preprocessing
- Smart missing value imputation strategies
- Label encoding for categorical variables
- Feature scaling and normalization

### 3. Visualizations
- Patient demographics analysis
- Hospital stay length distributions
- Medication usage patterns
- Correlation matrices with interactive heatmaps

### 4. Machine Learning Models
- **Basic Random Forest**: Multi-class classification with feature importance
- **Improved Random Forest**: Enhanced with feature selection and optimized hyperparameters
- **Logistic Regression**: Binary classification with coefficient analysis

### 5. Clinical Insights
- Model performance comparison
- Healthcare implementation recommendations
- ADA guidelines alignment analysis

## 📊 Key Results

- **Random Forest**: Achieved robust multi-class prediction with feature importance ranking
- **Improved Random Forest**: Enhanced accuracy through feature selection and hyperparameter optimization
- **Logistic Regression**: Strong binary classification performance with interpretable coefficients

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud

## 📁 Project Structure

```
├── app.py                           # Main Streamlit application
├── diabetic_data.csv               # Original dataset
├── diabetic_data_preprocessed.csv  # Preprocessed dataset
├── ADALab8and9.ipynb              # Jupyter notebook with detailed analysis
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🏃‍♂️ Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amaljyothis2003/lab8and9ada.git
   cd lab8and9ada
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📈 Model Performance

| Model | Accuracy | Features | Type |
|-------|----------|----------|------|
| Random Forest (Basic) | ~0.65 | All features | Multi-class |
| Random Forest (Improved) | ~0.67 | Selected features | Multi-class |
| Logistic Regression | ~0.62 | All features | Binary |

## 🏥 Healthcare Impact

This application helps healthcare providers:
- **Identify high-risk patients** for readmission
- **Optimize discharge planning** based on risk factors
- **Allocate resources** more efficiently
- **Improve patient outcomes** through early intervention

## 🔬 Technical Highlights

- **Advanced Feature Engineering**: Smart missing value imputation using age-based and mode-based strategies
- **Model Optimization**: Feature selection and hyperparameter tuning for improved performance
- **Interactive Visualizations**: Real-time data exploration with Plotly
- **Clinical Validation**: Alignment with ADA (American Diabetes Association) guidelines

## 📝 Methodology

1. **Data Exploration**: Comprehensive analysis of patient demographics and clinical features
2. **Preprocessing**: Strategic handling of missing values and categorical encoding
3. **Model Development**: Implementation of multiple ML algorithms with optimization
4. **Evaluation**: Rigorous performance assessment with clinical interpretation
5. **Deployment**: User-friendly web interface for healthcare professionals

## 🎯 Future Enhancements

- [ ] Real-time patient data integration
- [ ] Advanced ensemble methods
- [ ] External validation on different hospital systems
- [ ] Mobile-responsive design
- [ ] API development for EHR integration

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Diabetes 130-US hospitals dataset
- Healthcare guidelines: American Diabetes Association (ADA)
- Framework: Streamlit for rapid deployment

---

**Built with ❤️ for better healthcare outcomes**