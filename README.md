# 🛒 Supermart Grocery Sales Analysis and Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Results](#analysis-results)
- [Model Performance](#model-performance)
- [Key Insights](#key-insights)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a comprehensive analysis of grocery sales data from a supermarket chain, focusing on understanding sales patterns, customer behavior, and building predictive models for sales forecasting. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model implementation.

### 🎯 Objectives
- Analyze sales patterns across different categories and regions
- Identify key factors affecting sales performance
- Build predictive models for sales forecasting
- Provide actionable insights for business decision-making

## 📊 Dataset

The dataset contains **9,994 records** of grocery sales data with the following features:

| Column | Description | Type |
|--------|-------------|------|
| Order ID | Unique identifier for each order | Object |
| Customer Name | Name of the customer | Object |
| Category | Product category (Oil & Masala, Beverages, etc.) | Object |
| Sub Category | Product subcategory | Object |
| City | City where the sale occurred | Object |
| Order Date | Date of the order | Object |
| Region | Geographic region (North, South, East, West) | Object |
| Sales | Sales amount (target variable) | Integer |
| Discount | Discount percentage applied | Float |
| Profit | Profit amount | Float |
| State | State where the sale occurred | Object |

### 📈 Dataset Statistics
- **Total Records**: 9,994
- **Average Sales**: ₹1,496.60
- **Sales Range**: ₹500 - ₹2,500
- **Average Discount**: 22.68%
- **Average Profit**: ₹374.94

## 🏗️ Project Architecture

```
📦 Supermart-Grocery-Sales-Analysis-and-Prediction/
├── 📊 main.ipynb                                    # Main analysis notebook
├── 📄 README.md                                     # Project documentation
├── 📈 Supermart Grocery Sales - Retail Analytics Dataset.csv  # Dataset
├── 🎯 Supermart-Grocery-Sales-A-Retail-Analytics-Project (1) - Copy.pptx  # Presentation
└── 📋 requirements.txt                              # Dependencies (to be added)
```

## 🛠️ Technologies Used

### Programming Language
- **Python 3.8+**

### Libraries & Frameworks
- **Data Manipulation**: 
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  
- **Data Visualization**:
  - `matplotlib` - Basic plotting
  - `seaborn` - Statistical data visualization
  
- **Machine Learning**:
  - `scikit-learn` - Machine learning algorithms
  - `LinearRegression` - Predictive modeling
  - `StandardScaler` - Feature scaling
  - `LabelEncoder` - Categorical encoding
  - `train_test_split` - Data splitting

### Development Environment
- **Jupyter Notebook** - Interactive development environment

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SHAIKH-AKBAR-ALI/Supermart-Grocery-Sales-Analysis-and-Prediction.git
cd Supermart-Grocery-Sales-Analysis-and-Prediction
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

4. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

5. **Open the main analysis file**: `main.ipynb`

## 💻 Usage

1. **Data Loading**: The notebook automatically loads the dataset from the CSV file
2. **Data Exploration**: Run the EDA cells to understand data distribution and patterns
3. **Preprocessing**: Execute data cleaning and feature engineering steps
4. **Model Training**: Train the Linear Regression model
5. **Evaluation**: Analyze model performance metrics
6. **Visualization**: Generate various plots and charts for insights

### 🔧 Key Code Snippets

```python
# Data Loading
df = pd.read_csv('Supermart Grocery Sales - Retail Analytics Dataset.csv')

# Basic Statistics
print(df.info())
print(df.describe())

# Visualization
df['Category'].value_counts().plot(kind='bar')
plt.title('Sales Distribution by Category')
plt.show()
```

## 📊 Analysis Results

### 🏪 Category Analysis
- **Top Categories by Volume**:
  1. Snacks (1,514 orders)
  2. Eggs, Meat & Fish (1,490 orders)
  3. Fruits & Veggies (1,418 orders)
  4. Bakery (1,413 orders)

### 🌍 Regional Distribution
- Sales are distributed across **4 regions**: North, South, East, West
- All sales data is from **Tamil Nadu** state
- **24 cities** covered in the dataset

### 💰 Financial Insights
- **Sales Range**: ₹500 - ₹2,500
- **Discount Range**: 10% - 35%
- **Profit Margin**: Varies significantly across categories

## 🤖 Model Performance

### Linear Regression Results
- **Mean Squared Error (MSE)**: ~212,757
- **R-Squared Score**: ~0.355
- **Model Interpretation**: The model explains approximately 35.5% of the variance in sales data

### 📈 Performance Metrics
```
Training Accuracy: 35.5%
Test Accuracy: Moderate predictive performance
Feature Importance: Category, Region, and Discount are key predictors
```

## 🔍 Key Insights

### 📊 Business Intelligence
1. **Product Performance**: Snacks and protein-rich categories (Eggs, Meat & Fish) dominate sales volume
2. **Regional Balance**: Sales are relatively well-distributed across regions
3. **Discount Impact**: Higher discounts correlate with increased sales volume
4. **Profit Optimization**: Opportunity exists to optimize profit margins across categories

### 🎯 Recommendations
1. **Inventory Management**: Focus on high-performing categories (Snacks, Meat & Fish)
2. **Regional Strategy**: Develop region-specific marketing campaigns
3. **Pricing Strategy**: Optimize discount strategies to maximize profit
4. **Customer Segmentation**: Analyze customer purchasing patterns for targeted marketing

## 🚀 Future Enhancements

### 🔮 Planned Improvements
- [ ] **Advanced Models**: Implement Random Forest, XGBoost, and Neural Networks
- [ ] **Time Series Analysis**: Add seasonal trend analysis
- [ ] **Customer Segmentation**: Implement RFM analysis
- [ ] **Interactive Dashboard**: Create Streamlit/Dash dashboard
- [ ] **Feature Engineering**: Add more sophisticated features
- [ ] **Cross-Validation**: Implement k-fold cross-validation
- [ ] **Hyperparameter Tuning**: Optimize model parameters

### 📱 Deployment Options
- **Web Application**: Flask/Django web interface
- **API Development**: RESTful API for predictions
- **Cloud Deployment**: AWS/Azure deployment
- **Real-time Analytics**: Streaming data analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### 📝 Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shaikh Akbar Ali**
- GitHub: [@SHAIKH-AKBAR-ALI](https://github.com/SHAIKH-AKBAR-ALI)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Dataset source: Retail Analytics Dataset
- Inspiration: Real-world retail analytics challenges
- Community: Open source data science community

---

## 📞 Contact

For any questions or suggestions, please feel free to reach out:
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

---

*Last Updated: December 2024*