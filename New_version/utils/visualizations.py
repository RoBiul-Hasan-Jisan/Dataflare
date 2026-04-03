import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import pandas as pd
import numpy as np

class Visualizer:
    """Handles all visualization creation"""
    
    @staticmethod
    def to_json(fig):
        """Convert plotly figure to JSON"""
        if fig is None:
            return None
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    @staticmethod
    def create_distribution_plot(df, column, chart_type='histogram'):
        """Create distribution plot with proper styling"""
        if column not in df.columns:
            return None
        
        col_data = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numeric:
            if chart_type == 'histogram':
                fig = px.histogram(
                    df, x=column, nbins=40,
                    title=f"Distribution of {column}",
                    color_discrete_sequence=['#2E86AB']
                )
            elif chart_type == 'box':
                fig = px.box(
                    df, y=column,
                    title=f"Box Plot of {column}",
                    color_discrete_sequence=['#A23B72']
                )
            elif chart_type == 'violin':
                fig = px.violin(
                    df, y=column, box=True,
                    title=f"Violin Plot of {column}",
                    color_discrete_sequence=['#F18F01']
                )
            else:
                fig = px.histogram(df, x=column, nbins=40)
        else:
            # Categorical data
            value_counts = col_data.value_counts().head(15)
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Top 15 Values in {column}",
                labels={'x': column, 'y': 'Count'},
                color_discrete_sequence=['#2E86AB']
            )
        
        fig.update_layout(
            height=450,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df, max_cols=20):
        """Create correlation heatmap"""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2:
            return None
        
        # Limit columns for readability
        if len(num_cols) > max_cols:
            num_cols = num_cols[:max_cols]
        
        # Calculate correlation matrix
        corr_matrix = df[num_cols].corr().round(2)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 9},
            hoverongaps=False,
            colorbar=dict(title="Correlation", titleside="right")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            xaxis=dict(tickangle=-45, side="bottom"),
            yaxis=dict(tickangle=0),
            margin=dict(l=100, r=50, t=80, b=100),
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_missing_values_plot(df):
        """Create missing values visualization"""
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) == 0:
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found in the dataset! 🎉",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        else:
            missing_pct = (missing_counts / len(df) * 100).round(2)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=missing_counts.index,
                    y=missing_counts.values,
                    text=missing_counts.values,
                    textposition='outside',
                    marker_color='#F18F01',
                    name='Missing Count'
                )
            ])
            
            fig.add_trace(go.Scatter(
                x=missing_counts.index,
                y=missing_pct.values,
                yaxis='y2',
                mode='lines+markers',
                name='Missing %',
                marker=dict(color='#A23B72', size=8),
                line=dict(color='#A23B72', width=2)
            ))
            
            fig.update_layout(
                title="Missing Values by Column",
                height=450,
                xaxis_title="Columns",
                yaxis_title="Missing Count",
                yaxis2=dict(
                    title="Missing Percentage (%)",
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white',
                showlegend=True,
                legend=dict(x=0.02, y=0.98)
            )
        
        return fig
    
    @staticmethod
    def create_outlier_plot(df, column):
        """Create outlier visualization using box plot and scatter"""
        if column not in df.columns:
            return None
        
        fig = go.Figure()
        
        # Box plot
        fig.add_trace(go.Box(
            y=df[column].dropna(),
            name=column,
            boxmean='sd',
            marker_color='#2E86AB',
            boxpoints='outliers'
        ))
        
        # Add scatter plot for distribution
        fig.add_trace(go.Scatter(
            x=list(range(len(df[column].dropna()))),
            y=df[column].dropna(),
            mode='markers',
            name='Data Points',
            marker=dict(size=5, color='#F18F01', opacity=0.6),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"Outlier Analysis - {column}",
            height=500,
            yaxis=dict(title="Value", domain=[0, 0.7]),
            yaxis2=dict(title="Distribution", domain=[0.75, 1], overlaying='y', side='right'),
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_model_comparison_chart(results_df):
        """Create model comparison bar chart"""
        if results_df is None or len(results_df) == 0:
            return None
        
        # Get top 10 models
        top_models = results_df.head(10)
        
        # Find metric columns
        metric_cols = top_models.select_dtypes(include=[np.number]).columns
        if len(metric_cols) == 0:
            return None
        
        metric_name = metric_cols[0]
        model_col = "Model" if "Model" in top_models.columns else top_models.columns[0]
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_models[model_col],
                y=top_models[metric_name],
                text=top_models[metric_name].round(4),
                textposition='outside',
                marker_color=['#2E86AB' if i == 0 else '#A0C4E2' for i in range(len(top_models))]
            )
        ])
        
        fig.update_layout(
            title=f"Model Performance Comparison ({metric_name})",
            xaxis_title="Model",
            yaxis_title=metric_name,
            height=450,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_plot(feature_importance_dict):
        """Create feature importance plot"""
        if not feature_importance_dict:
            return None
        
        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[imp for _, imp in sorted_features],
                y=[name for name, _ in sorted_features],
                orientation='h',
                marker_color='#2E86AB',
                text=[f"{imp:.3f}" for _, imp in sorted_features],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Top 15 Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        return fig