import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import logging
import io
from typing import Optional, Tuple, Dict, List
import hashlib
import time
import calendar
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Try to import forecasting libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet not available. Using simpler forecasting methods.")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("❌ scikit-learn is required. Please install it.")

import math
import json
from pathlib import Path

# ────────────────────────────────────────────────
# Configuration & Constants
# ────────────────────────────────────────────────
class Config:
    # Security & Limits
    MAX_FILE_SIZE_MB = 50
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet'}
    MAX_ROWS_DISPLAY = 10000
    
    # Forecast settings
    MIN_DATA_DAYS = 14
    MAX_FORECAST_DAYS = 365
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    
    # Inventory settings
    DEFAULT_LEAD_TIME = 7
    DEFAULT_SAFETY_MULTIPLIER = 1.5
    
    # App settings
    PAGE_TITLE = "Retail Demand Forecasting & Sales Analysis"
    PAGE_ICON = "📊"
    LAYOUT = "wide"

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Clean Black & White CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background-color: #000000;
        border-radius: 8px;
    }
    .metric-card {
        background-color: #1a1a1a;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #cccccc;
    }
    .insight-card {
        background-color: #1a1a1a;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .anomaly-card {
        background-color: #1a1a1a;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
        background-color: #1a1a1a;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000;
        color: white;
        border-bottom: 3px solid white;
    }
    .score-card {
        background-color: #000000;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    /* Streamlit component styling */
    .stButton>button {
        background-color: #000000;
        color: white;
        border: 1px solid #333;
    }
    .stButton>button:hover {
        background-color: #333;
        color: white;
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# NEW FEATURE 1: AI Insights Generator
# ────────────────────────────────────────────────
class InsightGenerator:
    """Generate automated business insights from data"""
    
    @staticmethod
    def generate_sales_insights(daily_sales: pd.DataFrame) -> List[Dict]:
        """Generate actionable insights from sales data"""
        insights = []
        
        if daily_sales.empty:
            return insights
        
        # 1. Trend Analysis
        recent_growth = daily_sales['Sales'].tail(30).pct_change().mean() * 30
        if recent_growth > 0.05:
            insights.append({
                'type': 'positive',
                'title': '📈 Strong Growth Trend',
                'message': f"Monthly growth rate: {recent_growth:.1%}",
                'action': 'Consider increasing inventory to meet growing demand'
            })
        elif recent_growth < -0.05:
            insights.append({
                'type': 'warning',
                'title': '⚠️ Declining Trend',
                'message': f"Monthly decline: {abs(recent_growth):.1%}",
                'action': 'Review marketing strategies and product offerings'
            })
        
        # 2. Weekend vs Weekday Analysis
        daily_sales['is_weekend'] = daily_sales['Date'].dt.dayofweek >= 5
        weekend_sales = daily_sales[daily_sales['is_weekend']]['Sales'].mean()
        weekday_sales = daily_sales[~daily_sales['is_weekend']]['Sales'].mean()
        
        if weekend_sales > weekday_sales * 1.3:
            uplift = ((weekend_sales / weekday_sales) - 1) * 100
            insights.append({
                'type': 'info',
                'title': '📅 Weekend Opportunity',
                'message': f"Weekend sales are {uplift:.0f}% higher than weekdays",
                'action': 'Increase weekend staffing and promotions'
            })
        
        # 3. Seasonality Detection
        if len(daily_sales) > 90:
            daily_sales['month'] = daily_sales['Date'].dt.month
            monthly_avg = daily_sales.groupby('month')['Sales'].mean()
            peak_month = monthly_avg.idxmax()
            
            insights.append({
                'type': 'info',
                'title': '📊 Seasonal Pattern',
                'message': f"Peak sales in {calendar.month_name[peak_month]}",
                'action': 'Plan inventory and promotions for peak season'
            })
        
        # 4. Volatility Analysis
        volatility = daily_sales['Sales'].pct_change().std()
        if volatility > 0.3:
            insights.append({
                'type': 'warning',
                'title': '⚡ High Volatility',
                'message': f"Daily sales volatility: {volatility:.1%}",
                'action': 'Increase safety stock levels'
            })
        
        # 5. Best/Worst Days
        best_day = daily_sales.loc[daily_sales['Sales'].idxmax()]
        worst_day = daily_sales.loc[daily_sales['Sales'].idxmin()]
        
        insights.append({
            'type': 'info',
            'title': '📊 Performance Highlights',
            'message': f"Best day: {best_day['Date'].date()} (${best_day['Sales']:,.0f}) | "
                      f"Worst day: {worst_day['Date'].date()} (${worst_day['Sales']:,.0f})",
            'action': 'Analyze factors driving best day performance'
        })
        
        return insights

# ────────────────────────────────────────────────
# NEW FEATURE 2: Anomaly Detection System
# ────────────────────────────────────────────────
class AnomalyDetector:
    """Detect anomalies and unusual patterns in sales data"""
    
    @staticmethod
    def detect_sales_anomalies(daily_sales: pd.DataFrame, sensitivity: float = 0.95) -> Dict:
        """Detect anomalies using multiple methods"""
        
        anomalies = {
            'statistical': [],
            'seasonal': [],
            'trend': []
        }
        
        if len(daily_sales) < 30:
            return anomalies
        
        # Method 1: Statistical Outliers (Z-score)
        daily_sales['z_score'] = np.abs(
            (daily_sales['Sales'] - daily_sales['Sales'].rolling(30, min_periods=1).mean()) 
            / daily_sales['Sales'].rolling(30, min_periods=1).std()
        )
        
        z_threshold = stats.norm.ppf(sensitivity)
        statistical_anomalies = daily_sales[daily_sales['z_score'] > z_threshold]
        
        for _, row in statistical_anomalies.iterrows():
            anomalies['statistical'].append({
                'date': row['Date'],
                'value': row['Sales'],
                'z_score': row['z_score'],
                'reason': f"Extreme value (Z-score: {row['z_score']:.2f})"
            })
        
        # Method 2: Isolation Forest (ML-based)
        if SKLEARN_AVAILABLE and len(daily_sales) > 100:
            try:
                # Prepare features
                features = daily_sales[['Sales']].copy()
                features['rolling_mean'] = daily_sales['Sales'].rolling(7).mean()
                features['rolling_std'] = daily_sales['Sales'].rolling(7).std()
                features = features.fillna(features.mean())
                
                # Train model
                iso_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                predictions = iso_forest.fit_predict(features)
                
                # Get anomalies
                ml_anomalies = daily_sales[predictions == -1]
                for _, row in ml_anomalies.iterrows():
                    anomalies['statistical'].append({
                        'date': row['Date'],
                        'value': row['Sales'],
                        'reason': "ML-detected anomaly (unusual pattern)"
                    })
            except:
                pass
        
        # Method 3: Missing Data Detection
        expected_dates = pd.date_range(
            start=daily_sales['Date'].min(),
            end=daily_sales['Date'].max(),
            freq='D'
        )
        missing_dates = set(expected_dates) - set(daily_sales['Date'])
        
        for missing_date in missing_dates:
            anomalies['trend'].append({
                'date': missing_date,
                'reason': "Missing sales data"
            })
        
        # Method 4: Sudden Changes
        daily_sales['pct_change'] = daily_sales['Sales'].pct_change()
        sudden_changes = daily_sales[
            (abs(daily_sales['pct_change']) > 0.5) &  # 50% change
            (daily_sales['Sales'] > daily_sales['Sales'].quantile(0.25))  # Not tiny sales
        ]
        
        for _, row in sudden_changes.iterrows():
            change_type = "increase" if row['pct_change'] > 0 else "decrease"
            anomalies['trend'].append({
                'date': row['Date'],
                'value': row['Sales'],
                'change': row['pct_change'],
                'reason': f"Sudden {change_type} of {abs(row['pct_change']):.0%}"
            })
        
        return anomalies
    
    @staticmethod
    def generate_anomaly_report(anomalies: Dict) -> str:
        """Generate human-readable anomaly report"""
        if not any(len(v) > 0 for v in anomalies.values()):
            return "✅ No anomalies detected. Data looks clean!"
        
        report_lines = ["## 🚨 Anomaly Detection Report"]
        
        total_anomalies = sum(len(v) for v in anomalies.values())
        report_lines.append(f"**Total anomalies found:** {total_anomalies}")
        
        for anomaly_type, anomaly_list in anomalies.items():
            if anomaly_list:
                report_lines.append(f"\n### {anomaly_type.title()} Anomalies:")
                for i, anomaly in enumerate(anomaly_list[:5], 1):  # Show top 5
                    date_str = anomaly['date'].strftime('%Y-%m-%d')
                    if 'value' in anomaly:
                        report_lines.append(f"{i}. **{date_str}**: {anomaly['reason']} (Value: ${anomaly['value']:,.0f})")
                    else:
                        report_lines.append(f"{i}. **{date_str}**: {anomaly['reason']}")
        
        if total_anomalies > 5:
            report_lines.append(f"\n*... and {total_anomalies - 5} more anomalies*")
        
        report_lines.append("\n### 🤔 Recommended Actions:")
        report_lines.append("1. Investigate significant anomalies")
        report_lines.append("2. Verify data quality for missing dates")
        report_lines.append("3. Review business events on anomaly dates")
        
        return "\n".join(report_lines)

# ────────────────────────────────────────────────
# NEW FEATURE 3: Promotion Impact Simulator
# ────────────────────────────────────────────────
class PromotionSimulator:
    """Simulate promotional impact on sales"""
    
    @staticmethod
    def simulate_promotion_impact(
        base_sales: pd.DataFrame,
        promotion_dates: List,
        uplift_percentage: float = 0.3,
        halo_days: int = 2
    ) -> Dict:
        """Simulate promotion impact with halo effect"""
        
        simulated = base_sales.copy()
        impact_details = []
        
        for promo_date in promotion_dates:
            # Find the closest date in data
            date_diff = abs(simulated['Date'] - promo_date)
            if date_diff.min() > pd.Timedelta(days=1):
                continue  # Skip if date not in data
            
            promo_idx = date_diff.idxmin()
            original_value = simulated.loc[promo_idx, 'Sales']
            
            # Apply promotion uplift
            uplift_amount = original_value * uplift_percentage
            simulated.loc[promo_idx, 'Sales'] = original_value + uplift_amount
            
            impact_details.append({
                'date': simulated.loc[promo_idx, 'Date'],
                'original': original_value,
                'uplift': uplift_amount,
                'new_total': simulated.loc[promo_idx, 'Sales'],
                'uplift_pct': uplift_percentage
            })
            
            # Apply halo effect
            for i in range(1, halo_days + 1):
                if promo_idx + i < len(simulated):
                    halo_idx = promo_idx + i
                    halo_original = simulated.loc[halo_idx, 'Sales']
                    halo_uplift = uplift_amount * (1 - (i / (halo_days + 1)))
                    simulated.loc[halo_idx, 'Sales'] = halo_original + halo_uplift
        
        # Calculate summary metrics
        total_uplift = simulated['Sales'].sum() - base_sales['Sales'].sum()
        roi = (total_uplift / (len(promotion_dates) * 1000)) * 100  # Assuming $1000 per promotion
        
        return {
            'simulated_sales': simulated,
            'total_uplift': total_uplift,
            'roi_percentage': roi,
            'impact_details': impact_details,
            'promotion_dates': promotion_dates
        }
    
    @staticmethod
    def calculate_optimal_promotion_schedule(
        historical_sales: pd.DataFrame,
        budget: float = 5000,
        max_promotions: int = 4
    ) -> List:
        """Suggest optimal promotion dates based on historical patterns"""
        
        if len(historical_sales) < 90:
            return []
        
        suggestions = []
        
        # Analyze day of week performance
        historical_sales['day_of_week'] = historical_sales['Date'].dt.dayofweek
        dow_performance = historical_sales.groupby('day_of_week')['Sales'].mean()
        
        # Find best performing days
        best_days = dow_performance.nlargest(2).index.tolist()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day_idx in best_days:
            suggestions.append({
                'type': 'Day of Week',
                'recommendation': f"Promote on {day_names[day_idx]}s",
                'reason': f"Historically {((dow_performance[day_idx] / dow_performance.mean()) - 1) * 100:.0f}% above average",
                'confidence': 'High'
            })
        
        # Analyze monthly patterns
        if len(historical_sales) > 365:
            historical_sales['month'] = historical_sales['Date'].dt.month
            monthly_performance = historical_sales.groupby('month')['Sales'].mean()
            best_month = monthly_performance.idxmax()
            
            suggestions.append({
                'type': 'Seasonal',
                'recommendation': f"Focus promotions in {calendar.month_name[best_month]}",
                'reason': f"Peak sales month, {((monthly_performance[best_month] / monthly_performance.mean()) - 1) * 100:.0f}% above average",
                'confidence': 'Medium'
            })
        
        return suggestions

# ────────────────────────────────────────────────
# NEW FEATURE 4: Performance Scorecard
# ────────────────────────────────────────────────
class PerformanceScorecard:
    """Generate performance score and grades"""
    
    @staticmethod
    def calculate_overall_score(
        df: pd.DataFrame,
        forecast_results: Optional[Dict] = None,
        inventory_metrics: Optional[Dict] = None
    ) -> Dict:
        """Calculate 0-100 performance score"""
        
        scores = {}
        
        # 1. Data Quality Score (0-30 points)
        if df is not None:
            # Completeness
            completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            
            # Consistency (date gaps)
            if 'Date' in df.columns:
                date_gaps = df['Date'].diff().dt.days.fillna(1).max()
                consistency = 1 - min(1, (date_gaps - 1) / 30)  # Penalize gaps > 1 day
            
            scores['data_quality'] = (completeness * 0.7 + consistency * 0.3) * 30
        
        # 2. Forecast Quality Score (0-30 points)
        if forecast_results and 'metrics' in forecast_results:
            metrics = forecast_results['metrics']
            if 'mape' in metrics and metrics['mape'] is not None:
                accuracy_score = max(0, 30 - (metrics['mape'] * 0.3))  # MAPE < 10% = good
                scores['forecast_quality'] = accuracy_score
        
        # 3. Inventory Efficiency Score (0-20 points)
        if inventory_metrics:
            turnover = inventory_metrics.get('stock_turnover', 0)
            # Ideal turnover: 8-12
            if 8 <= turnover <= 12:
                turnover_score = 20
            elif 6 <= turnover < 8 or 12 < turnover <= 14:
                turnover_score = 15
            else:
                turnover_score = 10
            
            service_level = inventory_metrics.get('service_level', 0.95)
            service_score = min(20, service_level * 20)
            
            scores['inventory_efficiency'] = (turnover_score + service_score) / 2
        
        # 4. Growth Score (0-20 points)
        if df is not None and 'Sales' in df.columns:
            if len(df) > 60:
                recent_avg = df['Sales'].tail(30).mean()
                previous_avg = df['Sales'].head(30).mean()
                
                if previous_avg > 0:
                    growth_rate = (recent_avg / previous_avg) - 1
                    growth_score = min(20, max(0, (growth_rate + 0.1) * 100))  # +10% growth = full points
                    scores['growth'] = growth_score
        
        # Calculate total
        total_score = sum(scores.values())
        
        # Assign grade
        if total_score >= 90:
            grade = 'A+'
        elif total_score >= 85:
            grade = 'A'
        elif total_score >= 80:
            grade = 'A-'
        elif total_score >= 75:
            grade = 'B+'
        elif total_score >= 70:
            grade = 'B'
        elif total_score >= 65:
            grade = 'B-'
        elif total_score >= 60:
            grade = 'C+'
        else:
            grade = 'C'
        
        return {
            'total_score': round(total_score, 1),
            'grade': grade,
            'breakdown': scores,
            'recommendations': PerformanceScorecard.generate_recommendations(scores)
        }
    
    @staticmethod
    def generate_recommendations(scores: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores.get('data_quality', 0) < 25:
            recommendations.append("📊 Improve data completeness and consistency")
        
        if scores.get('forecast_quality', 0) < 25:
            recommendations.append("🔮 Consider using Prophet model for better accuracy")
        
        if scores.get('inventory_efficiency', 0) < 15:
            recommendations.append("📦 Review inventory levels and safety stock")
        
        if scores.get('growth', 0) < 15:
            recommendations.append("📈 Implement growth strategies and promotions")
        
        return recommendations

# ────────────────────────────────────────────────
# Forecasting Models - SIMPLIFIED VERSION
# ────────────────────────────────────────────────
class ForecastingModel:
    """Forecasting model wrapper"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Training forecasting model...")
    def train_prophet(_df: pd.DataFrame, forecast_days: int, conf_interval: float) -> Dict:
        """Train Prophet model with caching"""
        try:
            forecast_data = _df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
            
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=conf_interval,
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
            
            m.fit(forecast_data)
            
            future = m.make_future_dataframe(periods=forecast_days)
            forecast = m.predict(future)
            
            # Prepare forecast results
            future_forecast = forecast[forecast['ds'] > forecast_data['ds'].max()][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            ].copy()
            future_forecast.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
            
            return {
                'success': True,
                'forecast': future_forecast,
                'historical': forecast_data
            }
            
        except Exception as e:
            logger.error(f"Prophet training error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def simple_moving_average(df: pd.DataFrame, forecast_days: int, window: int) -> Dict:
        """Simple moving average forecast"""
        try:
            daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
            last_avg = daily_sales['Sales'].tail(window).mean()
            std_dev = daily_sales['Sales'].tail(window).std()
            
            future_dates = pd.date_range(
                start=daily_sales['Date'].max() + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            preds = [last_avg] * forecast_days
            upper_bound = [last_avg + 1.96 * std_dev] * forecast_days
            lower_bound = [max(0, last_avg - 1.96 * std_dev)] * forecast_days
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': preds,
                'Upper_Bound': upper_bound,
                'Lower_Bound': lower_bound
            })
            
            return {
                'success': True,
                'forecast': forecast_df,
                'historical': daily_sales
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def linear_trend(df: pd.DataFrame, forecast_days: int) -> Dict:
        """Linear trend forecast"""
        try:
            if len(df) < 8:
                return {'success': False, 'error': 'Insufficient data for linear trend'}
            
            daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
            
            # Prepare data for linear regression
            daily_sales['days'] = (daily_sales['Date'] - daily_sales['Date'].min()).dt.days
            
            X = daily_sales[['days']].values
            y = daily_sales['Sales'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future
            last_date = daily_sales['Date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            future_days = (future_dates - daily_sales['Date'].min()).days.values.reshape(-1, 1)
            future_preds = model.predict(future_days)
            
            # Calculate confidence intervals
            residuals = y - model.predict(X)
            std_residuals = np.std(residuals)
            conf_upper = future_preds + 1.96 * std_residuals
            conf_lower = np.maximum(0, future_preds - 1.96 * std_residuals)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': future_preds,
                'Upper_Bound': conf_upper,
                'Lower_Bound': conf_lower
            })
            
            return {
                'success': True,
                'forecast': forecast_df,
                'historical': daily_sales
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ────────────────────────────────────────────────
# Inventory Optimizer
# ────────────────────────────────────────────────
class InventoryOptimizer:
    """Inventory optimization calculations"""
    
    @staticmethod
    def calculate_inventory_metrics(
        df: pd.DataFrame,
        lead_time: int,
        safety_multiplier: float,
        unit_price: float,
        service_level: float = 0.95
    ) -> Dict:
        """Calculate inventory optimization metrics"""
        try:
            daily_sales = df.groupby('Date')['Sales'].sum()
            
            if len(daily_sales) < 30:
                lookback = len(daily_sales)
            else:
                lookback = 60
            
            recent_sales = daily_sales.tail(lookback)
            
            # Basic calculations
            avg_daily_sales = recent_sales.mean()
            max_daily_sales = recent_sales.max()
            std_daily_sales = recent_sales.std()
            
            # Convert to units
            if unit_price > 0:
                avg_daily_units = avg_daily_sales / unit_price
                std_daily_units = std_daily_sales / unit_price
            else:
                avg_daily_units = std_daily_units = 0
            
            # Safety stock calculation (simplified)
            z_score = 1.65  # for 95% service level
            safety_stock = z_score * std_daily_units * np.sqrt(lead_time)
            
            # Reorder point
            reorder_point = (avg_daily_units * lead_time) + (safety_stock * safety_multiplier)
            
            # Economic Order Quantity (EOQ) - simplified
            # Assuming holding cost = 25% of unit price, ordering cost = $50
            holding_cost_rate = 0.25
            ordering_cost = 50
            annual_demand = avg_daily_units * 365
            
            if unit_price > 0 and annual_demand > 0:
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost_rate * unit_price))
                eoq = max(1, int(eoq))
            else:
                eoq = 0
            
            # Stock turnover
            if avg_daily_units > 0:
                stock_turnover = 365 * avg_daily_units / reorder_point if reorder_point > 0 else 0
            else:
                stock_turnover = 0
            
            return {
                'avg_daily_sales': avg_daily_sales,
                'avg_daily_units': avg_daily_units,
                'std_daily_units': std_daily_units,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq,
                'stock_turnover': stock_turnover,
                'max_daily_units': max_daily_sales / unit_price if unit_price > 0 else 0,
                'inventory_value': reorder_point * unit_price,
                'service_level': service_level
            }
            
        except Exception as e:
            logger.error(f"Inventory calculation error: {str(e)}")
            return {}

# ────────────────────────────────────────────────
# Data Validator
# ────────────────────────────────────────────────
class DataValidator:
    """Validate and clean data"""
    
    @staticmethod
    def validate_file(file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        try:
            # Check file size
            max_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024
            if file.size > max_size:
                return False, f"File size exceeds {Config.MAX_FILE_SIZE_MB}MB limit"
            
            # Check extension
            file_ext = Path(file.name).suffix.lower()
            if file_ext not in Config.ALLOWED_EXTENSIONS:
                return False, f"File type not supported. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            
            return True, "File validation successful"
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Validate and clean dataframe"""
        try:
            # Create a copy to avoid modifying original
            df_clean = df.copy()
            
            # Check if dataframe is empty
            if df_clean.empty:
                return False, "DataFrame is empty", df_clean
            
            # Convert date columns
            date_cols = [col for col in df_clean.columns if 'date' in col.lower()]
            if not date_cols:
                return False, "No date column found. Expected column with 'date' in name", df_clean
            
            # Use first date column found
            date_col = date_cols[0]
            df_clean['Date'] = pd.to_datetime(df_clean[date_col], errors='coerce')
            
            # Check for invalid dates
            if df_clean['Date'].isnull().any():
                invalid_count = df_clean['Date'].isnull().sum()
                return False, f"{invalid_count} rows have invalid dates", df_clean
            
            # Identify sales/revenue column
            sales_cols = []
            for col in ['sales', 'revenue', 'amount', 'value']:
                if col in df_clean.columns.str.lower():
                    matches = df_clean.columns[df_clean.columns.str.lower() == col]
                    if len(matches) > 0:
                        sales_cols.append(matches[0])
            
            if not sales_cols:
                # Try to find numeric columns
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    sales_col = numeric_cols[0]
                    df_clean['Sales'] = df_clean[sales_col]
                else:
                    return False, "No numeric sales/revenue column found", df_clean
            else:
                df_clean['Sales'] = df_clean[sales_cols[0]]
            
            # Sort by date
            df_clean = df_clean.sort_values('Date')
            
            # Remove duplicates
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['Date'] + list(df_clean.columns.difference(['Date'])))
            duplicates_removed = initial_rows - len(df_clean)
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Check for sufficient data
            if len(df_clean) < Config.MIN_DATA_DAYS:
                return False, f"Insufficient data. Minimum {Config.MIN_DATA_DAYS} days required", df_clean
            
            return True, f"Data validation successful. {duplicates_removed} duplicates removed", df_clean
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}", pd.DataFrame()

# ────────────────────────────────────────────────
# Session State Initialization
# ────────────────────────────────────────────────
def init_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'inventory_metrics' not in st.session_state:
        st.session_state.inventory_metrics = None
    if 'data_quality_report' not in st.session_state:
        st.session_state.data_quality_report = {}
    if 'app_start_time' not in st.session_state:
        st.session_state.app_start_time = datetime.now()
    if 'detected_anomalies' not in st.session_state:
        st.session_state.detected_anomalies = {}
    if 'performance_score' not in st.session_state:
        st.session_state.performance_score = None
    if 'generated_insights' not in st.session_state:
        st.session_state.generated_insights = []

init_session_state()

# ────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────
st.markdown(f'<h1 class="main-header">📊 {Config.PAGE_TITLE}</h1>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
# NEW FEATURE: Quick Actions Bar
# ────────────────────────────────────────────────
if st.session_state.data is not None:
    st.markdown("### ⚡ Quick Actions")
    quick_cols = st.columns(5)
    
    with quick_cols[0]:
        if st.button("🔍 Find Anomalies", use_container_width=True):
            daily_sales = st.session_state.data.groupby('Date')['Sales'].sum().reset_index()
            anomalies = AnomalyDetector.detect_sales_anomalies(daily_sales)
            st.session_state.detected_anomalies = anomalies
            st.rerun()
    
    with quick_cols[1]:
        if st.button("💡 Generate Insights", use_container_width=True):
            daily_sales = st.session_state.data.groupby('Date')['Sales'].sum().reset_index()
            insights = InsightGenerator.generate_sales_insights(daily_sales)
            st.session_state.generated_insights = insights
            st.rerun()
    
    with quick_cols[2]:
        if st.button("📊 Performance Score", use_container_width=True):
            score = PerformanceScorecard.calculate_overall_score(
                st.session_state.data,
                st.session_state.forecast_results,
                st.session_state.inventory_metrics
            )
            st.session_state.performance_score = score
            st.rerun()
    
    with quick_cols[3]:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with quick_cols[4]:
        if st.button("📥 Export View", use_container_width=True):
            st.info("Export feature coming soon!")

# ────────────────────────────────────────────────
# Sidebar – Data Upload & Configuration
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Upload your data", "Try with sample data"],
        help="Upload your sales data or explore with sample data"
    )
    
    if data_source == "Upload your data":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=list(Config.ALLOWED_EXTENSIONS),
            help=f"Max size: {Config.MAX_FILE_SIZE_MB}MB. Supported: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
        
        if uploaded_file is not None:
            # Validate file
            is_valid, message = DataValidator.validate_file(uploaded_file)
            
            if not is_valid:
                st.error(f"❌ {message}")
                st.session_state.data = None
            else:
                with st.spinner("Loading and validating data..."):
                    try:
                        # Read file based on extension
                        file_ext = Path(uploaded_file.name).suffix.lower()
                        
                        if file_ext == '.csv':
                            df = pd.read_csv(uploaded_file, encoding='utf-8')
                        elif file_ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(uploaded_file)
                        elif file_ext == '.parquet':
                            df = pd.read_parquet(uploaded_file)
                        
                        # Validate and clean data
                        is_valid, message, df_clean = DataValidator.validate_dataframe(df)
                        
                        if is_valid:
                            # Calculate data hash to detect changes
                            data_hash = hashlib.md5(pd.util.hash_pandas_object(df_clean).values).hexdigest()
                            
                            # Only update if data has changed
                            if data_hash != st.session_state.data_hash:
                                st.session_state.data = df_clean
                                st.session_state.data_hash = data_hash
                                st.session_state.data_quality_report = {
                                    'rows': len(df_clean),
                                    'columns': len(df_clean.columns),
                                    'date_range': f"{df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}",
                                    'missing_values': df_clean.isnull().sum().sum(),
                                    'duplicates_removed': len(df) - len(df_clean)
                                }
                                st.success("✅ Data loaded successfully!")
                                logger.info(f"Data loaded: {len(df_clean)} rows, {len(df_clean.columns)} columns")
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.data = None
                            
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
                        logger.error(f"File loading error: {str(e)}")
                        st.session_state.data = None
    
    elif data_source == "Try with sample data":
        @st.cache_data(ttl=Config.CACHE_TTL)
        def generate_sample_data_cached(n_days=365, n_products=5):
            """Generate sample data with caching"""
            np.random.seed(42)
            dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
            products = [f'Product_{i+1}' for i in range(n_products)]
            categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
            
            data = []
            for date in dates:
                # Base trend with seasonality
                trend = 100 + (date - dates[0]).days * 0.15
                weekly = 25 * np.sin(2 * np.pi * date.dayofweek / 7)
                monthly = 40 * np.sin(2 * np.pi * date.day / 30)
                yearly = 60 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                # Add some randomness
                noise = np.random.normal(0, 10)
                
                for product in products:
                    product_factor = np.random.uniform(0.5, 1.5)
                    sales = max(10, trend + weekly + monthly + yearly + noise) * product_factor
                    
                    # Weekend effect
                    if date.dayofweek >= 5:  # Saturday or Sunday
                        sales *= 1.3
                    
                    # Holiday effect
                    if date.month == 12 and date.day in range(15, 26):  # Christmas period
                        sales *= 1.5
                    
                    data.append({
                        'Date': date,
                        'Sales': round(sales, 2),
                        'Revenue': round(sales * np.random.uniform(10, 100), 2),
                        'Quantity': int(np.random.poisson(sales / 15)),
                        'Product': product,
                        'Category': np.random.choice(categories),
                        'Price': round(np.random.uniform(10, 100), 2),
                        'Region': np.random.choice(['North', 'South', 'East', 'West'])
                    })
            
            df = pd.DataFrame(data)
            return df
        
        if st.button("Generate Sample Data", type="primary", use_container_width=True):
            with st.spinner("Generating realistic retail data..."):
                df = generate_sample_data_cached()
                st.session_state.data = df
                
                # Generate data quality report
                st.session_state.data_quality_report = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'date_range': f"{df['Date'].min().date()} to {df['Date'].max().date()}",
                    'missing_values': 0,
                    'duplicates_removed': 0,
                    'products': df['Product'].nunique(),
                    'categories': df['Category'].nunique()
                }
                
                st.success(f"✅ Sample data loaded! Explore all features.")
                logger.info("Sample data generated successfully")
    
    # Advanced Settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    # Data sampling for large datasets
    if st.session_state.data is not None and len(st.session_state.data) > Config.MAX_ROWS_DISPLAY:
        sample_size = st.slider(
            "Sample Size (for display)",
            min_value=1000,
            max_value=min(Config.MAX_ROWS_DISPLAY, len(st.session_state.data)),
            value=min(5000, len(st.session_state.data)),
            step=1000,
            help="Large datasets are sampled for better performance"
        )
    else:
        sample_size = None
    
    # Reset button
    if st.button("🔄 Reset Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ────────────────────────────────────────────────
# Main Content - Data Overview
# ────────────────────────────────────────────────
if st.session_state.data is None:
    # Show welcome/instruction screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: white;">
            <h2>🚀 Welcome to Retail Forecasting Dashboard</h2>
            <p style="font-size: 1.2rem; color: #cccccc; margin: 2rem 0;">
                Upload your sales data or use sample data to get started with:
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 2rem 0;">
                <div style="text-align: left; padding: 1rem; background: #1a1a1a; border-radius: 8px; color: white;">
                    <strong>📈 Sales Trend Analysis</strong>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #cccccc;">Track performance with AI insights</p>
                </div>
                <div style="text-align: left; padding: 1rem; background: #1a1a1a; border-radius: 8px; color: white;">
                    <strong>🔮 Demand Forecasting</strong>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #cccccc;">Predict future sales accurately</p>
                </div>
                <div style="text-align: left; padding: 1rem; background: #1a1a1a; border-radius: 8px; color: white;">
                    <strong>📦 Inventory Optimization</strong>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #cccccc;">Smart stock recommendations</p>
                </div>
                <div style="text-align: left; padding: 1rem; background: #1a1a1a; border-radius: 8px; color: white;">
                    <strong>🤖 AI-Powered Insights</strong>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #cccccc;">Automated anomaly detection</p>
                </div>
            </div>
            <p style="margin-top: 2rem; color: #cccccc;">
                👈 <strong>Use the sidebar</strong> to upload data or generate sample data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# ────────────────────────────────────────────────
# Data Quality Report
# ────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Data Quality Report")

col1, col2, col3, col4, col5 = st.columns(5)

report = st.session_state.data_quality_report

with col1:
    st.metric("Total Rows", f"{report.get('rows', 0):,}")
with col2:
    st.metric("Total Columns", report.get('columns', 0))
with col3:
    st.metric("Date Range", report.get('date_range', 'N/A'))
with col4:
    st.metric("Missing Values", report.get('missing_values', 0))
with col5:
    if 'duplicates_removed' in report and report['duplicates_removed'] > 0:
        st.info(f"Duplicates: {report['duplicates_removed']}")

# Data filters
st.markdown("---")
st.subheader("🔍 Data Filters")

df = st.session_state.data

# Apply sampling if needed
if sample_size and len(df) > sample_size:
    df_display = df.sample(n=sample_size, random_state=42)
    st.info(f"Showing {sample_size:,} random samples from {len(df):,} total rows")
else:
    df_display = df.copy()

# Dynamic filters
filter_cols = st.columns(4)

with filter_cols[0]:
    if 'Product' in df.columns:
        products = st.multiselect(
            "Products",
            options=df['Product'].unique(),
            default=df['Product'].unique()[:3] if len(df['Product'].unique()) > 3 else df['Product'].unique()
        )
        if products:
            df_display = df_display[df_display['Product'].isin(products)]

with filter_cols[1]:
    if 'Category' in df.columns:
        categories = st.multiselect(
            "Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
        if categories:
            df_display = df_display[df_display['Category'].isin(categories)]

with filter_cols[2]:
    if 'Region' in df.columns:
        regions = st.multiselect(
            "Regions",
            options=df['Region'].unique() if 'Region' in df.columns else [],
            default=df['Region'].unique() if 'Region' in df.columns else []
        )
        if regions:
            df_display = df_display[df_display['Region'].isin(regions)]

with filter_cols[3]:
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_display = df_display[
            (df_display['Date'].dt.date >= start_date) & 
            (df_display['Date'].dt.date <= end_date)
        ]

# Prepare daily sales data
daily_sales = df_display.groupby('Date')['Sales'].sum().reset_index(name='Total_Sales')

if len(daily_sales) < Config.MIN_DATA_DAYS:
    st.error(f"⚠️ Insufficient data after filtering. Need at least {Config.MIN_DATA_DAYS} days.")
    st.info("Try adjusting your filters to include more data.")
    st.stop()

# ────────────────────────────────────────────────
# NEW FEATURE: AI Insights Display
# ────────────────────────────────────────────────
if st.session_state.generated_insights:
    st.markdown("---")
    st.subheader("💡 AI-Generated Insights")
    
    insight_cols = st.columns(2)
    for idx, insight in enumerate(st.session_state.generated_insights):
        with insight_cols[idx % 2]:
            st.markdown(f"""
            <div class="insight-card">
                <strong>{insight['title']}</strong><br>
                {insight['message']}<br>
                <small><em>💡 {insight['action']}</em></small>
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────────────────────────────
# NEW FEATURE: Anomaly Detection Display
# ────────────────────────────────────────────────
if st.session_state.detected_anomalies:
    st.markdown("---")
    st.subheader("🚨 Detected Anomalies")
    
    anomaly_report = AnomalyDetector.generate_anomaly_report(st.session_state.detected_anomalies)
    st.markdown(anomaly_report)
    
    # Show anomaly visualization
    fig_anomaly = go.Figure()
    
    fig_anomaly.add_trace(go.Scatter(
        x=daily_sales['Date'],
        y=daily_sales['Total_Sales'],
        mode='lines',
        name='Sales',
        line=dict(color='white', width=2)
    ))
    
    # Highlight anomalies
    all_anomalies = []
    for anomaly_type in st.session_state.detected_anomalies.values():
        for anomaly in anomaly_type:
            if 'date' in anomaly:
                all_anomalies.append(anomaly['date'])
    
    anomaly_sales = daily_sales[daily_sales['Date'].isin(all_anomalies)]
    
    if not anomaly_sales.empty:
        fig_anomaly.add_trace(go.Scatter(
            x=anomaly_sales['Date'],
            y=anomaly_sales['Total_Sales'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # Set dark theme for plot
    fig_anomaly.update_layout(
        title="Sales with Detected Anomalies",
        height=400,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333')
    )
    
    st.plotly_chart(fig_anomaly, use_container_width=True)

# ────────────────────────────────────────────────
# NEW FEATURE: Performance Scorecard Display
# ────────────────────────────────────────────────
if st.session_state.performance_score:
    st.markdown("---")
    st.subheader("🏆 Performance Scorecard")
    
    score = st.session_state.performance_score
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h1 style="font-size: 4rem; margin: 0; color: white;">{score['total_score']}</h1>
            <h2 style="margin: 0.5rem 0; color: white;">Overall Grade: {score['grade']}</h2>
            <div style="margin-top: 1rem;">
        """, unsafe_allow_html=True)
        
        for category, cat_score in score['breakdown'].items():
            st.progress(cat_score / 30 if 'quality' in category else cat_score / 20, 
                       text=f"{category.replace('_', ' ').title()}: {cat_score:.1f}")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Show recommendations
    if score['recommendations']:
        st.subheader("📋 Improvement Recommendations")
        for rec in score['recommendations']:
            st.info(rec)

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab1, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", 
    "🔮 Forecast", 
    "📦 Inventory",
    "🎯 Promotion Simulator",
    "📁 Data Explorer"
])

# ── Tab 1: Dashboard ───────────────────────────────────────
with tab1:
    st.header("Executive Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = daily_sales['Total_Sales'].sum()
    avg_daily = daily_sales['Total_Sales'].mean()
    max_daily = daily_sales['Total_Sales'].max()
    min_daily = daily_sales['Total_Sales'].min()
    std_daily = daily_sales['Total_Sales'].std()
    cv = (std_daily / avg_daily * 100) if avg_daily > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Sales</div>
            <div class="metric-value">${total_sales:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Daily Sales</div>
            <div class="metric-value">${avg_daily:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Daily Sales</div>
            <div class="metric-value">${max_daily:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Variability (CV)</div>
            <div class="metric-value">{cv:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Sales Trend Chart
    fig = go.Figure()
    
    # Add daily sales
    fig.add_trace(go.Scatter(
        x=daily_sales['Date'],
        y=daily_sales['Total_Sales'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='white', width=2)
    ))
    
    # Add moving average
    window_size = st.slider("Moving Average Window", 7, 90, 30, key="dashboard_ma")
    daily_sales[f'{window_size}D_MA'] = daily_sales['Total_Sales'].rolling(window=window_size, center=True).mean()
    
    fig.add_trace(go.Scatter(
        x=daily_sales['Date'],
        y=daily_sales[f'{window_size}D_MA'],
        mode='lines',
        name=f'{window_size}-Day Moving Avg',
        line=dict(color='#cccccc', width=3, dash='dash')
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title="Sales Trend with Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        height=500,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='#1a1a1a',
            font=dict(color='white')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Sales by Day of Week")
        daily_sales['DayOfWeek'] = daily_sales['Date'].dt.day_name()
        weekly_sales = daily_sales.groupby('DayOfWeek')['Total_Sales'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig_weekly = px.bar(
            x=weekly_sales.index,
            y=weekly_sales.values,
            labels={'x': 'Day', 'y': 'Average Sales ($)'},
            color=weekly_sales.values,
            color_continuous_scale='gray'
        )
        
        # Update layout for dark theme
        fig_weekly.update_layout(
            height=300,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        st.subheader("📈 Cumulative Sales")
        daily_sales['Cumulative_Sales'] = daily_sales['Total_Sales'].cumsum()
        
        fig_cum = px.area(
            x=daily_sales['Date'],
            y=daily_sales['Cumulative_Sales'],
            labels={'x': 'Date', 'y': 'Cumulative Sales ($)'}
        )
        
        # Update layout for dark theme
        fig_cum.update_layout(
            height=300,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig_cum, use_container_width=True)

# ── Tab 3: Forecasting ───────────────────────────────────────
with tab3:
    st.header("Demand Forecasting")
    
    # Forecast configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days = st.number_input(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=Config.MAX_FORECAST_DAYS,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
    
    with col2:
        conf_interval = st.slider(
            "Confidence Level",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Statistical confidence interval for forecasts"
        )
    
    with col3:
        train_test_split = st.slider(
            "Train/Test Split (%)",
            min_value=70,
            max_value=95,
            value=80,
            step=5,
            help="Percentage of data to use for training"
        )
    
    # Model selection
    st.subheader("🤖 Forecasting Model Selection")
    
    if PROPHET_AVAILABLE:
        model_options = ["Prophet (Recommended)", "Linear Regression", "Moving Average"]
    else:
        model_options = ["Linear Regression", "Moving Average"]
    
    selected_model = st.radio(
        "Select Forecasting Model",
        model_options,
        horizontal=True,
        help="Choose the algorithm for forecasting"
    )
    
    # Additional parameters based on model
    if selected_model == "Moving Average":
        ma_window = st.slider(
            "Moving Average Window",
            min_value=7,
            max_value=90,
            value=14,
            step=7,
            help="Number of days to average for baseline"
        )
    
    # Run forecast
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Training model and generating forecast..."):
            try:
                if selected_model == "Prophet (Recommended)":
                    result = ForecastingModel.train_prophet(
                        df_display,
                        forecast_days,
                        conf_interval
                    )
                    
                    if result['success']:
                        forecast_df = result['forecast']
                        historical = result['historical']
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=historical['ds'],
                            y=historical['y'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='white', width=2)
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#4CAF50', width=2, dash='dash')
                        ))
                        
                        # Confidence interval
                        if 'Lower_Bound' in forecast_df.columns and 'Upper_Bound' in forecast_df.columns:
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Upper_Bound'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Lower_Bound'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(76, 175, 80, 0.2)',
                                name=f'{int(conf_interval*100)}% Confidence'
                            ))
                        
                        # Update layout for dark theme
                        fig.update_layout(
                            title="Prophet Forecast",
                            xaxis_title="Date",
                            yaxis_title="Sales ($)",
                            height=600,
                            hovermode='x unified',
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(gridcolor='#333'),
                            legend=dict(
                                bgcolor='#1a1a1a',
                                font=dict(color='white')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast table
                        st.subheader("📅 Forecast Table")
                        st.dataframe(forecast_df.round(2), use_container_width=True)
                        
                        # Store in session state
                        st.session_state.forecast_results = {
                            'model': 'Prophet',
                            'forecast': forecast_df,
                            'metrics': {}
                        }
                    else:
                        st.error(f"Forecast failed: {result.get('error', 'Unknown error')}")
                
                elif selected_model == "Linear Regression":
                    result = ForecastingModel.linear_trend(df_display, forecast_days)
                    
                    if result['success']:
                        forecast_df = result['forecast']
                        
                        # Plot results
                        fig = go.Figure()
                        
                        # Add historical data
                        if 'historical' in result:
                            fig.add_trace(go.Scatter(
                                x=result['historical']['Date'],
                                y=result['historical']['Sales'],
                                mode='lines',
                                name='Historical',
                                line=dict(color='white', width=2)
                            ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#4CAF50', width=2, dash='dash')
                        ))
                        
                        # Confidence interval
                        if 'Lower_Bound' in forecast_df.columns and 'Upper_Bound' in forecast_df.columns:
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Upper_Bound'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Lower_Bound'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(76, 175, 80, 0.2)',
                                name=f'{int(conf_interval*100)}% Confidence'
                            ))
                        
                        # Update layout for dark theme
                        fig.update_layout(
                            title="Linear Regression Forecast",
                            xaxis_title="Date",
                            yaxis_title="Sales ($)",
                            height=600,
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(gridcolor='#333'),
                            legend=dict(
                                bgcolor='#1a1a1a',
                                font=dict(color='white')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(forecast_df.round(2), use_container_width=True)
                        
                        st.session_state.forecast_results = {
                            'model': 'Linear Regression',
                            'forecast': forecast_df,
                            'metrics': {}
                        }
                    else:
                        st.error(f"Linear regression failed: {result.get('error', 'Unknown error')}")
                
                elif selected_model == "Moving Average":
                    result = ForecastingModel.simple_moving_average(df_display, forecast_days, ma_window)
                    
                    if result['success']:
                        forecast_df = result['forecast']
                        
                        fig = go.Figure()
                        
                        # Add historical data
                        if 'historical' in result:
                            fig.add_trace(go.Scatter(
                                x=result['historical']['Date'],
                                y=result['historical']['Sales'],
                                mode='lines',
                                name='Historical',
                                line=dict(color='white', width=2)
                            ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#4CAF50', width=2, dash='dash')
                        ))
                        
                        # Confidence interval
                        if 'Lower_Bound' in forecast_df.columns and 'Upper_Bound' in forecast_df.columns:
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Upper_Bound'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Lower_Bound'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(76, 175, 80, 0.2)',
                                name=f'{int(conf_interval*100)}% Confidence'
                            ))
                        
                        # Update layout for dark theme
                        fig.update_layout(
                            title=f"Moving Average Forecast ({ma_window}-day window)",
                            xaxis_title="Date",
                            yaxis_title="Sales ($)",
                            height=600,
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(gridcolor='#333'),
                            legend=dict(
                                bgcolor='#1a1a1a',
                                font=dict(color='white')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(forecast_df.round(2), use_container_width=True)
                        
                        st.session_state.forecast_results = {
                            'model': 'Moving Average',
                            'forecast': forecast_df,
                            'metrics': {}
                        }
                    else:
                        st.error(f"Moving average failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
                logger.error(f"Forecasting failed: {str(e)}")
    
    # Show previous forecast if available
    if st.session_state.forecast_results:
        st.subheader("📋 Previous Forecast Summary")
        
        forecast_data = st.session_state.forecast_results['forecast']
        model_name = st.session_state.forecast_results['model']
        
        col1, col2, col3 = st.columns(3)
        
        total_forecast = forecast_data['Forecast'].sum()
        avg_daily_forecast = forecast_data['Forecast'].mean()
        
        with col1:
            st.metric("Forecast Model", model_name)
        with col2:
            st.metric("Total Forecast", f"${total_forecast:,.0f}")
        with col3:
            st.metric("Avg Daily Forecast", f"${avg_daily_forecast:,.0f}")

# ── Tab 4: Inventory ───────────────────────────────────────
with tab4:
    st.header("Inventory Optimization")
    
    st.info("""
    📦 This module helps optimize inventory levels based on demand forecasting.
    Adjust the parameters below to match your business requirements.
    """)
    
    # Inventory parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        lead_time = st.number_input(
            "Lead Time (days)",
            min_value=1,
            max_value=60,
            value=Config.DEFAULT_LEAD_TIME,
            help="Time between placing order and receiving inventory"
        )
    
    with col2:
        safety_multiplier = st.slider(
            "Safety Stock Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=Config.DEFAULT_SAFETY_MULTIPLIER,
            step=0.1,
            help="Multiplier for safety stock (higher = more buffer)"
        )
    
    with col3:
        unit_price = st.number_input(
            "Average Unit Price ($)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
            help="Average price per inventory unit"
        )
    
    with col4:
        service_level = st.slider(
            "Service Level Target",
            min_value=0.85,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Target probability of not stocking out"
        )
    
    # Calculate inventory metrics
    if st.button("Calculate Inventory Recommendations", use_container_width=True):
        with st.spinner("Calculating optimal inventory levels..."):
            try:
                metrics = InventoryOptimizer.calculate_inventory_metrics(
                    df_display,
                    lead_time,
                    safety_multiplier,
                    unit_price,
                    service_level
                )
                
                if metrics:
                    st.session_state.inventory_metrics = metrics
                    
                    # Display metrics
                    st.subheader("📊 Inventory Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Avg Daily Units",
                            f"{metrics['avg_daily_units']:.0f}",
                            help="Average units sold per day"
                        )
                    
                    with col2:
                        st.metric(
                            "Safety Stock",
                            f"{metrics['safety_stock']:.0f} units",
                            help="Buffer stock to prevent shortages"
                        )
                    
                    with col3:
                        st.metric(
                            "Reorder Point",
                            f"{metrics['reorder_point']:.0f} units",
                            help="Inventory level to trigger new order"
                        )
                    
                    with col4:
                        st.metric(
                            "EOQ",
                            f"{metrics['economic_order_quantity']:.0f} units",
                            help="Economic Order Quantity"
                        )
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    recommendations = []
                    
                    if metrics['stock_turnover'] < 6:
                        recommendations.append("**Low turnover**: Consider reducing inventory levels or increasing promotions")
                    elif metrics['stock_turnover'] > 12:
                        recommendations.append("**High turnover**: Risk of stockouts - consider increasing safety stock")
                    
                    if metrics['safety_stock'] > metrics['avg_daily_units'] * 10:
                        recommendations.append("**High safety stock**: May be tying up too much capital")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.success("Inventory levels appear optimal for current settings!")
                    
                    # Download report
                    report_data = {
                        'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'parameters': {
                            'lead_time': lead_time,
                            'safety_multiplier': safety_multiplier,
                            'unit_price': unit_price,
                            'service_level': service_level
                        },
                        'metrics': metrics,
                        'recommendations': recommendations
                    }
                    
                    json_report = json.dumps(report_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Inventory Report",
                        data=json_report,
                        file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error("Failed to calculate inventory metrics")
            
            except Exception as e:
                st.error(f"Inventory calculation error: {str(e)}")
                logger.error(f"Inventory calculation failed: {str(e)}")
    
    # Show previous inventory metrics if available
    if st.session_state.inventory_metrics:
        st.subheader("📋 Previous Inventory Analysis")
        
        metrics = st.session_state.inventory_metrics
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Inventory Value", f"${metrics['inventory_value']:,.2f}")
        with col2:
            st.metric("Stock Turnover", f"{metrics['stock_turnover']:.1f}")

# ── Tab 5: Promotion Simulator ───────────────────────────────────────
with tab5:
    st.header("🎯 Promotion Impact Simulator")
    
    st.info("""
    Simulate the impact of promotions on your sales. This tool helps you:
    - Plan optimal promotion timing
    - Estimate sales uplift
    - Calculate ROI for promotional activities
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Select Promotion Dates")
        
        # Date range for promotions
        min_date = daily_sales['Date'].min()
        max_date = daily_sales['Date'].max()
        
        promo_start = st.date_input(
            "Promotion Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        promo_duration = st.slider(
            "Promotion Duration (days)",
            1, 14, 3,
            help="Number of days the promotion runs"
        )
        
        # Generate promotion dates
        promotion_dates = []
        current_date = promo_start
        for i in range(promo_duration):
            promotion_dates.append(current_date + timedelta(days=i))
    
    with col2:
        st.subheader("💰 Promotion Parameters")
        
        uplift_percentage = st.slider(
            "Expected Uplift Percentage",
            0.1, 1.0, 0.3, 0.05,
            help="Expected sales increase during promotion"
        )
        
        halo_days = st.slider(
            "Halo Effect Days",
            0, 7, 2,
            help="Days after promotion with residual uplift"
        )
        
        promotion_cost = st.number_input(
            "Promotion Cost ($)",
            0, 10000, 1000,
            help="Total cost of running the promotion"
        )
    
    # Run simulation
    if st.button("🚀 Simulate Promotion Impact", type="primary", use_container_width=True):
        with st.spinner("Simulating promotion impact..."):
            try:
                # Prepare base data
                base_sales = daily_sales.copy()
                
                # Run simulation
                simulation_result = PromotionSimulator.simulate_promotion_impact(
                    base_sales=base_sales,
                    promotion_dates=promotion_dates,
                    uplift_percentage=uplift_percentage,
                    halo_days=halo_days
                )
                
                # Display results
                st.subheader("📊 Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Sales Uplift",
                        f"${simulation_result['total_uplift']:,.0f}",
                        delta=f"+{uplift_percentage*100:.0f}%"
                    )
                
                with col2:
                    roi = simulation_result['roi_percentage']
                    st.metric(
                        "Estimated ROI",
                        f"{roi:.1f}%",
                        delta_color="inverse" if roi < 100 else "normal"
                    )
                
                with col3:
                    st.metric(
                        "Promotion Dates",
                        f"{len(promotion_dates)} days",
                        f"{promo_start.strftime('%b %d')}"
                    )
                
                # Plot comparison
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=base_sales['Date'],
                    y=base_sales['Total_Sales'],
                    mode='lines',
                    name='Base Sales',
                    line=dict(color='#666666', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=simulation_result['simulated_sales']['Date'],
                    y=simulation_result['simulated_sales']['Total_Sales'],
                    mode='lines',
                    name='With Promotion',
                    line=dict(color='white', width=3)
                ))
                
                # Highlight promotion period
                for promo_date in promotion_dates:
                    fig.add_vrect(
                        x0=promo_date,
                        x1=promo_date + timedelta(days=1),
                        fillcolor="#4CAF50",
                        opacity=0.2,
                        line_width=0,
                        annotation_text="Promo",
                        annotation_position="top"
                    )
                
                # Update layout for dark theme
                fig.update_layout(
                    title="Promotion Impact Simulation",
                    xaxis_title="Date",
                    yaxis_title="Sales ($)",
                    height=500,
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333'),
                    yaxis=dict(gridcolor='#333'),
                    legend=dict(
                        bgcolor='#1a1a1a',
                        font=dict(color='white')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Get optimal promotion suggestions
                st.subheader("💡 Optimal Promotion Suggestions")
                
                suggestions = PromotionSimulator.calculate_optimal_promotion_schedule(base_sales)
                
                if suggestions:
                    for suggestion in suggestions[:3]:  # Show top 3
                        st.info(f"""
                        **{suggestion['type']}**: {suggestion['recommendation']}
                        
                        *Why?* {suggestion['reason']}  
                        *Confidence*: {suggestion['confidence']}
                        """)
                else:
                    st.info("Need more historical data for optimal timing suggestions")
                
                # ROI Analysis
                st.subheader("📈 ROI Analysis")
                
                additional_profit = simulation_result['total_uplift'] * 0.3  # Assuming 30% profit margin
                net_profit = additional_profit - promotion_cost
                
                roi_data = pd.DataFrame({
                    'Metric': ['Additional Sales', 'Additional Profit (30% margin)', 'Promotion Cost', 'Net Profit', 'ROI'],
                    'Value': [
                        f"${simulation_result['total_uplift']:,.0f}",
                        f"${additional_profit:,.0f}",
                        f"${promotion_cost:,.0f}",
                        f"${net_profit:,.0f}",
                        f"{((additional_profit - promotion_cost) / promotion_cost * 100):.1f}%"
                    ]
                })
                
                st.dataframe(roi_data, use_container_width=True)
                
                if net_profit > 0:
                    st.success(f"✅ Promotion is profitable! Net profit: ${net_profit:,.0f}")
                else:
                    st.warning(f"⚠️ Promotion may not be profitable. Consider adjusting parameters.")
            
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                logger.error(f"Promotion simulation failed: {str(e)}")

# ── Tab 6: Data Explorer ───────────────────────────────────────
with tab6:
    st.header("Data Explorer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        
        # Show data with options
        display_option = st.radio(
            "Display Options",
            ["Head", "Tail", "Sample", "Full Data"],
            horizontal=True
        )
        
        rows_to_show = st.slider("Rows to display", 10, 1000, 100, step=10)
        
        if display_option == "Head":
            st.dataframe(df_display.head(rows_to_show), use_container_width=True)
        elif display_option == "Tail":
            st.dataframe(df_display.tail(rows_to_show), use_container_width=True)
        elif display_option == "Sample":
            st.dataframe(df_display.sample(min(rows_to_show, len(df_display))), use_container_width=True)
        else:
            st.dataframe(df_display, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Column Details")
        
        # Column information
        col_info = pd.DataFrame({
            'Column': df_display.columns,
            'Type': df_display.dtypes.astype(str),
            'Non-Null': df_display.notnull().sum(),
            'Null': df_display.isnull().sum(),
            'Unique': [df_display[col].nunique() for col in df_display.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, height=400)
    
    # Data download
    st.subheader("📥 Export Data")
    
    export_format = st.selectbox("Export Format", ["CSV", "Excel"])
    
    if export_format == "CSV":
        csv_data = df_display.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"retail_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Data')
            summary_df = pd.DataFrame({
                'Metric': ['Total Rows', 'Total Columns', 'Date Range Start', 'Date Range End'],
                'Value': [
                    len(df_display),
                    len(df_display.columns),
                    df_display['Date'].min().date(),
                    df_display['Date'].max().date()
                ]
            })
            summary_df.to_excel(writer, index=False, sheet_name='Summary')
        
        st.download_button(
            label="Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"retail_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ────────────────────────────────────────────────
# Clean Business-Focused Footer
# ────────────────────────────────────────────────
st.markdown("---")

if st.session_state.data is not None:
    df = st.session_state.data
    
    # Calculate key business metrics
    if 'Sales' in df.columns:
        total_sales = df['Sales'].sum()
        avg_daily_sales = df.groupby('Date')['Sales'].sum().mean()
        
        # Get date range
        start_date = df['Date'].min().strftime('%b %d, %Y')
        end_date = df['Date'].max().strftime('%b %d, %Y')
        days_analyzed = (df['Date'].max() - df['Date'].min()).days + 1
        
        # Display metrics in a clean format
        st.markdown(f"""
        <div style="
            background-color: #1a1a1a;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #cccccc; font-size: 0.9rem;">📅 Analysis Period</span><br>
                    <span style="font-weight: 600; color: white;">{start_date} to {end_date}</span>
                </div>
                <div style="text-align: center;">
                    <span style="color: #cccccc; font-size: 0.9rem;">📊 Total Sales</span><br>
                    <span style="font-weight: 600; color: white;">${total_sales:,.0f}</span>
                </div>
                <div style="text-align: center;">
                    <span style="color: #cccccc; font-size: 0.9rem;">📈 Avg Daily</span><br>
                    <span style="font-weight: 600; color: white;">${avg_daily_sales:,.0f}</span>
                </div>
                <div style="text-align: right;">
                    <span style="color: #cccccc; font-size: 0.9rem;">📁 Data Scope</span><br>
                    <span style="font-weight: 600; color: white;">{len(df):,} rows | {days_analyzed} days</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Minimal branding and timestamp
st.markdown("---")
st.markdown(f"""
<div style="
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #cccccc;
    font-size: 0.8rem;
    padding: 0.5rem 0;
">
    <div>Retail Demand Forecasting System • v2.0</div>
    <div>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)