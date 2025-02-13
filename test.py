# Provide me a technical analysis of AAVE based on indicators

import pandas as pd
import yfinance as yf
import json
window_data = '{"result": [{"close": 295.42, "high": 301.37, "low": 285.26, "open": 289.54, "time": "2025-01-14 01:00:00", "volume_from": 115682.39, "volume_to": 33789582.72}, {"close": 319.91, "high": 324.01, "low": 285.72, "open": 295.42, "time": "2025-01-15 01:00:00", "volume_from": 87529.4, "volume_to": 26823482.1}, {"close": 311.2, "high": 321.37, "low": 305.08, "open": 319.91, "time": "2025-01-16 01:00:00", "volume_from": 59722.5, "volume_to": 18751355.14}, {"close": 335.38, "high": 337.14, "low": 311, "open": 311.2, "time": "2025-01-17 01:00:00", "volume_from": 96187.71, "volume_to": 31372645.65}, {"close": 315.18, "high": 341.97, "low": 306.16, "open": 335.38, "time": "2025-01-18 01:00:00", "volume_from": 94368.21, "volume_to": 30053403.04}, {"close": 308.85, "high": 347.6, "low": 292.2, "open": 315.18, "time": "2025-01-19 01:00:00", "volume_from": 188742.85, "volume_to": 60427078.1}, {"close": 337.04, "high": 364.83, "low": 296.19, "open": 308.85, "time": "2025-01-20 01:00:00", "volume_from": 247826.34, "volume_to": 83715682.43}, {"close": 370.77, "high": 378.47, "low": 327.39, "open": 337.04, "time": "2025-01-21 01:00:00", "volume_from": 239637.28, "volume_to": 84219693.82}, {"close": 346.25, "high": 372.21, "low": 345.24, "open": 370.77, "time": "2025-01-22 01:00:00", "volume_from": 73949.32, "volume_to": 26407867.91}, {"close": 337.47, "high": 350.17, "low": 324.5, "open": 346.25, "time": "2025-01-23 01:00:00", "volume_from": 145290.78, "volume_to": 49007169.48}, {"close": 334.87, "high": 357.68, "low": 328.08, "open": 337.47, "time": "2025-01-24 01:00:00", "volume_from": 86343.03, "volume_to": 29867936.7}, {"close": 326.36, "high": 338.38, "low": 326.12, "open": 334.87, "time": "2025-01-25 01:00:00", "volume_from": 62828.9, "volume_to": 20845174.3}, {"close": 316.45, "high": 336.18, "low": 315.84, "open": 326.36, "time": "2025-01-26 01:00:00", "volume_from": 50852.5, "volume_to": 16659132.82}, {"close": 305.09, "high": 320.35, "low": 288.06, "open": 316.45, "time": "2025-01-27 01:00:00", "volume_from": 144050.94, "volume_to": 43681842.59}, {"close": 284.05, "high": 307.84, "low": 280.89, "open": 305.09, "time": "2025-01-28 01:00:00", "volume_from": 108935.97, "volume_to": 32176498.68}, {"close": 290.87, "high": 301.82, "low": 282.38, "open": 284.05, "time": "2025-01-29 01:00:00", "volume_from": 133850.09, "volume_to": 39301210.56}, {"close": 315.44, "high": 322.21, "low": 287.12, "open": 290.87, "time": "2025-01-30 01:00:00", "volume_from": 113365.61, "volume_to": 35069546.8}, {"close": 332.46, "high": 348.96, "low": 312.61, "open": 315.44, "time": "2025-01-31 01:00:00", "volume_from": 160783.96, "volume_to": 53530686.87}, {"close": 296.12, "high": 333.22, "low": 295.33, "open": 332.46, "time": "2025-02-01 01:00:00", "volume_from": 68335.12, "volume_to": 21422570.3}, {"close": 258.68, "high": 304.68, "low": 245.98, "open": 296.12, "time": "2025-02-02 01:00:00", "volume_from": 148037.49, "volume_to": 40531739.84}, {"close": 276.65, "high": 285.29, "low": 195.82, "open": 258.68, "time": "2025-02-03 01:00:00", "volume_from": 381405.47, "volume_to": 93317615.78}, {"close": 272.34, "high": 278.55, "low": 249.51, "open": 276.65, "time": "2025-02-04 01:00:00", "volume_from": 167281.57, "volume_to": 43971102.74}, {"close": 259.74, "high": 283.39, "low": 256.18, "open": 272.34, "time": "2025-02-05 01:00:00", "volume_from": 86812.18, "volume_to": 23343648.58}, {"close": 241.51, "high": 266.88, "low": 239.37, "open": 259.74, "time": "2025-02-06 01:00:00", "volume_from": 94966.64, "volume_to": 23947313.64}, {"close": 238.1, "high": 262.14, "low": 231.14, "open": 241.51, "time": "2025-02-07 01:00:00", "volume_from": 108593.71, "volume_to": 26672270}, {"close": 238.83, "high": 241.8, "low": 231.24, "open": 238.1, "time": "2025-02-08 01:00:00", "volume_from": 46264.86, "volume_to": 10928133}, {"close": 241.33, "high": 252.54, "low": 229.98, "open": 238.83, "time": "2025-02-09 01:00:00", "volume_from": 64255.47, "volume_to": 15618239.86}, {"close": 252.62, "high": 258.32, "low": 234.67, "open": 241.33, "time": "2025-02-10 01:00:00", "volume_from": 76398.86, "volume_to": 19003583.11}, {"close": 243.79, "high": 262.75, "low": 242.01, "open": 252.62, "time": "2025-02-11 01:00:00", "volume_from": 65946.83, "volume_to": 16795469.59}, {"close": 252.75, "high": 261.01, "low": 231.85, "open": 243.79, "time": "2025-02-12 01:00:00", "volume_from": 96809.9, "volume_to": 23567577.13}, {"close": 251.33, "high": 258.12, "low": 246.62, "open": 252.75, "time": "2025-02-13 01:00:00", "volume_from": 43330.87, "volume_to": 10925680.09}]}'
data = json.loads(window_data)

dow_data = yf.download("AAVE-USD", period="1mo", interval="1d")
print(dow_data.tail(5))

window = pd.DataFrame(data)
if len(window) < 5:
    print("Not enough data to analyze")

# Create a DataFrame from the "result" list
df = pd.DataFrame(data['result'])

# Convert the "time" column to datetime and extract just the date
df['Date'] = pd.to_datetime(df['time']).dt.date

# Rename columns for better presentation
df = df.rename(columns={
    'close': 'Close',
    'high': 'High',
    'low': 'Low',
    'open': 'Open',
    # You can choose which volume column to use. For example, use "volume_to" as "Volume":
    'volume_to': 'Volume'
})

# Select and reorder only the desired columns
df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

# Optionally, sort the DataFrame by Date
df = df.sort_values('Date').reset_index(drop=True)

last_price = df['Close'].iloc[-1]
rolling_avg = df['Close'].rolling(window=7).mean().iloc[-1]  # 7-day rolling average
price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) >= 2 else 0
volume_change = df['Volume'].iloc[-1] - df['Volume'].iloc[-2] if len(df) >= 2 else 0

ema7 = df['Close'].ewm(span=7, adjust=False).mean().iloc[-1]  # Exponential Moving Average
ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]  # Exponential Moving Average
ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]  # Exponential Moving Average
ema100 = df['Close'].ewm(span=100, adjust=False).mean().iloc[-1]  # Exponential Moving Average
ema200 = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]  # Exponential Moving Average
std = df['Close'].rolling(window=5).std().iloc[-1]  # Standard deviation for Bollinger Bands
bollinger_upper = rolling_avg + (2 * std)
bollinger_lower = rolling_avg - (2 * std)

# Calculate Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
rs = avg_gain / avg_loss if avg_loss != 0 else float('nan')
rsi = 100 - (100 / (1 + rs))

# Calculate Ichimoku indicators
# Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
nine_period_high = df['High'].rolling(window= 9).max().iloc[-1]
nine_period_low = df['Low'].rolling(window= 9).min().iloc[-1]
tenkan_sen = (nine_period_high + nine_period_low) /2
# Kijun-sen (Base Line): (26-period high + 26-period low)/2))
period26_high = df['High'].rolling(window=26).max().iloc[-1]
period26_low = df['Low'].rolling(window=26).min().iloc[-1]
kijun_sen = (period26_high + period26_low) / 2

# Calculate Daily High/Low and Momentum
daily_high = df['High'].iloc[-1]
daily_low = df['Low'].iloc[-1]
buying_momentum = last_price - daily_low    # Distance of the last price from the day's low
selling_momentum = daily_high - last_price   # Distance of the day's high from the last price

# Print the calculated insights
print (
    f"Last price: {last_price:.2f}\n"
    f"7-day Rolling Average: {rolling_avg:.2f}\n"
    f"EMA7: {ema7:.2f}\n"
    f"EMA20: {ema20:.2f}\n"
    f"EMA50: {ema50:.2f}\n"
    f"EMA100: {ema100:.2f}\n"
    f"EMA200: {ema200:.2f}\n"
    f"RSI: {rsi:.2f}\n"
    f"Bollinger Upper Band: {bollinger_upper:.2f}\n"
    f"Bollinger Lower Band: {bollinger_lower:.2f}\n"
    f"Price Change: {price_change:.2f}\n"
    f"Volume Change: {volume_change}\n"
    f"Daily High: {daily_high:.2f}\n"
    f"Daily Low: {daily_low:.2f}\n"
    f"Buying Momentum: {buying_momentum:.2f}\n"
    f"Selling Momentum: {selling_momentum:.2f}\n"
    f"Tenkan Sen: {tenkan_sen:.2f}\n"
    f"Kijun Sen: {kijun_sen:.2f}"
)
