# Retail Demand Forecasting & Sales Analysis

A comprehensive Streamlit application for retail sales analysis, demand forecasting, and inventory optimization.

## Features

### 📊 Overview Dashboard
- Total sales metrics and KPIs
- Interactive sales trend visualization
- Moving average analysis
- Data summary and statistics

### 🔍 Exploratory Data Analysis
- **Time Series Decomposition**: Identify trends and patterns
- **Weekly Patterns**: Analyze sales by day of week
- **Monthly Patterns**: Understand monthly seasonality
- **Yearly Seasonality**: Visualize annual patterns
- **Product/Category Analysis**: Breakdown by product and category
- **Correlation Analysis**: Understand relationships between variables

### 🔮 Sales Forecasting
- **Prophet Model**: Advanced time series forecasting with seasonality
- **Simple Moving Average**: Quick forecasting method
- **Linear Trend**: Trend-based forecasting
- Model performance metrics (MAE, RMSE)
- Confidence intervals
- Forecast export functionality

### 📦 Inventory Recommendations
- **Reorder Point Calculation**: Based on lead time and demand
- **Safety Stock Optimization**: Configurable safety stock multiplier
- **Inventory Simulation**: Simulate stock levels over time
- **Actionable Recommendations**: 
  - Stock level optimization
  - Lead time analysis
  - Demand variability assessment
  - Seasonal adjustments
- Stockout risk analysis

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will open in your browser automatically

3. **Data Options**:
   - **Use Sample Data**: Check the box in the sidebar to generate sample retail data
   - **Upload Your Data**: Upload a CSV or Excel file with the following columns:
     - `Date`: Date column (required)
     - `Sales` or `Revenue`: Numeric sales column (required)
     - `Product`: Product names (optional)
     - `Category`: Product categories (optional)

4. **Navigate through tabs**:
   - **Overview**: Get a quick summary of your sales data
   - **Exploratory Analysis**: Deep dive into patterns and trends
   - **Forecasting**: Generate sales predictions
   - **Inventory Recommendations**: Get actionable inventory insights

## Data Format

Your data file should have the following structure:

| Date | Sales | Product | Category |
|------|-------|---------|----------|
| 2022-01-01 | 1250.50 | Product_1 | Electronics |
| 2022-01-02 | 1320.75 | Product_2 | Clothing |
| ... | ... | ... | ... |

**Required Columns:**
- Date (any date format)
- Sales or Revenue (numeric)

**Optional Columns:**
- Product
- Category
- Quantity
- Any other numeric columns for analysis

## Features in Detail

### Forecasting Methods

1. **Prophet (Advanced)**
   - Handles seasonality automatically
   - Provides confidence intervals
   - Best for data with clear patterns
   - Requires Prophet library

2. **Simple Moving Average**
   - Quick and simple
   - Uses recent average sales
   - Good for stable demand

3. **Linear Trend**
   - Captures overall trend
   - Simple linear regression
   - Good for trending data

### Inventory Recommendations

The application calculates:
- **Daily Demand**: Average units needed per day
- **Reorder Point**: When to place new orders
- **Recommended Stock Level**: Optimal inventory level
- **Inventory Value**: Estimated cost of recommended stock

Recommendations consider:
- Lead time
- Safety stock requirements
- Demand variability
- Seasonal patterns

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Prophet (optional, for advanced forecasting)
- Statsmodels (optional)

## Notes

- The application uses sample data by default for demonstration
- Prophet model provides the most accurate forecasts but requires more data
- All visualizations are interactive (zoom, pan, hover for details)
- Forecasts can be downloaded as CSV files

## Troubleshooting

**Prophet not available?**
- The app will automatically use alternative forecasting methods
- Install Prophet: `pip install prophet`

**Data loading issues?**
- Ensure your date column is in a recognizable format
- Check that sales column contains numeric values
- Verify file format (CSV or Excel)

**Performance issues?**
- For large datasets (>100k rows), consider filtering data
- Use sample data option to test functionality first

## License

This project is open source and available for educational and commercial use.
