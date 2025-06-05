# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Stock Prediction Dashboard"  # Set the browser tab title

# Step 1: Function to Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    df = pd.DataFrame(data)
    df.columns = [col[0] for col in df.columns]
    df['Date'] = pd.to_datetime(df.index)
    return df

# Step 2: Function to Train Model
def train_model(df):
    df = df.drop(['Date', 'Volume'], axis=1)
    df.reset_index(drop=True, inplace=True)
    X = df[['Open', 'High', 'Low']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return rf, mse

# Step 3: Layout of the App
app.layout = html.Div(
    style={
        "backgroundColor": "#121212",  # Dark background color
        "fontFamily": "Arial, sans-serif",
        "padding": "20px",
        "color": "#ffffff",  # White text color
    },
    children=[
        # Header Section
        html.Div(
            style={
                "textAlign": "center",
                "padding": "20px",
                "backgroundColor": "#1f1f1f",  # Darker shade for header
                "color": "#ffffff",
                "borderRadius": "8px",
                "marginBottom": "20px",
            },
            children=[
                html.H1(
                    "Stock Price Prediction Dashboard",
                    style={"fontSize": "36px", "marginBottom": "10px"},
                ),
                html.P("Predict the next closing price of your favorite stocks!"),
            ],
        ),

        # Input Section
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "space-around",
                "gap": "20px",
                "marginBottom": "20px",
            },
            children=[
                # Stock Ticker Input
                html.Div(
                    style={
                        "flex": "1 1 300px",
                        "backgroundColor": "#1f1f1f",  # Darker card background
                        "padding": "15px",
                        "borderRadius": "10px",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)",
                        "overflow": "hidden",
                    },
                    children=[
                        html.Label("Enter Stock Ticker:", style={"fontWeight": "bold", "color": "#ffffff"}),
                        dcc.Input(
                            id="ticker-input",
                            value="TSLA",
                            type="text",
                            placeholder="e.g., AAPL",
                            style={
                                "width": "90%",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "border": "1px solid #333",
                                "backgroundColor": "#333",  # Input background color
                                "color": "#ffffff",  # Input text color
                                "marginTop": "10px",
                            },
                        ),
                    ],
                ),

                # Date Range Picker
                html.Div(
                    style={
                        "flex": "1 1 300px",
                        "backgroundColor": "#1f1f1f",
                        "padding": "15px",
                        "borderRadius": "10px",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)",
                    },
                    children=[
                        html.Label("Select Date Range:", style={"fontWeight": "bold", "color": "#ffffff"}),
                        dcc.DatePickerRange(
                            id="date-picker",
                            start_date="2020-01-01",
                            end_date="2025-02-27",
                            display_format="YYYY-MM-DD",
                            style={
                                "marginTop": "10px",
                                "backgroundColor": "#333",
                                "color": "#ffffff",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # Candlestick Chart and Volume Graph
        html.Div(
            style={
                "backgroundColor": "#1f1f1f",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)",
                "marginBottom": "20px",
            },
            children=[
                html.H3("Stock Price Chart and Volume Chart", style={"textAlign": "center", "marginBottom": "20px"}),
                dcc.Graph(
                    id="candlestick-chart",
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"],
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "Stock_Price_Chart",
                            "height": 500,
                            "width": 800,
                            "scale": 2,
                        },
                        "scrollZoom": True,
                    },
                    style={"height": "500px"},
                ),
                dcc.Graph(
                    id="volume-chart",
                    config={"displayModeBar": False},
                    style={"height": "300px", "marginTop": "20px"},
                ),
            ],
        ),

        # Prediction and MSE Section
        html.Div(
            style={
                "backgroundColor": "#1f1f1f",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)",
                "textAlign": "center",
            },
            children=[
                html.H3(id="prediction-output", style={"color": "#4caf50", "marginBottom": "10px"}),  # Green text
                html.H4(id="mse-output", style={"color": "#ff5722"}),  # Orange text
            ],
        ),

        # Footer Section
        html.Div(
            style={
                "marginTop": "30px",
                "textAlign": "center",
                "color": "#888",
                "fontSize": "14px",
            },
            children=[
                html.P("© 2025 Stock Prediction Dashboard. All rights reserved."),
                html.P("Powered by Dash, Plotly, and Scikit-Learn"),
            ],
        ),
    ],
)

# Step 4: Callbacks for Updating Dashboard
@app.callback(
    [Output("candlestick-chart", "figure"),
     Output("volume-chart", "figure"),
     Output("prediction-output", "children"),
     Output("mse-output", "children")],
    [Input("ticker-input", "value"),
     Input("date-picker", "start_date"),
     Input("date-picker", "end_date")]
)
def update_dashboard(ticker, start_date, end_date):
    # Fetch stock data
    df = fetch_stock_data(ticker, start_date, end_date)

    # Train model
    rf, mse = train_model(df)

    # Candlestick chart
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    candlestick_fig.update_layout(
        title=f"Stock Price Chart - {ticker.upper()}",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",  # Use Plotly's dark theme
    )

    # Volume bar graph
    volume_fig = go.Figure(data=[go.Bar(
        x=df['Date'],
        y=df['Volume'],
        marker_color='rgba(166, 64, 13, 0.8)'
    )])
    volume_fig.update_layout(
        title=f"Trading Volume - {ticker.upper()}",
        yaxis_title="Volume",
        xaxis_title="Date",
        template="plotly_dark",  # Use Plotly's dark theme
    )

    # Predict the next price
    last_row = df.iloc[-1]
    new_data = np.array([last_row['Open'], last_row['High'], last_row['Low']]).reshape(1, -1)
    predicted_price = rf.predict(new_data)[0]

    prediction_text = f"Predicted Next Close Price for {ticker.upper()}: ${predicted_price:.2f}"
    mse_text = f"Range : {mse:.2f}"

    return candlestick_fig, volume_fig, prediction_text, mse_text

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
