from __future__ import annotations as _annotations

from datetime import datetime
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import json
import pandas as pd

import httpx
import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from utils import ensure_closing_brace

load_dotenv()

llm = os.getenv('LLM_MODEL', 'deepseek/deepseek-chat')

print(llm)
model = OpenAIModel(
    llm,
    base_url = 'https://openrouter.ai/api/v1',
    api_key = os.getenv('OPEN_ROUTER_API_KEY')
) if os.getenv('OPEN_ROUTER_API_KEY', None) else OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class TradingDeps:
    client: httpx.AsyncClient
    ccompare_api_key: str | None = None


system_prompt = """
You are a Crypto Trading Analyst Agent responsible for delivering comprehensive market analysis for any given cryptocurrency token. 

Your objective is to analyze both historical market data and global asset information to generate actionable insights.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the data information with the provided tools before answering the user's question unless you have already.

When answering a question about the cryptocurrency token, always start your answer with the token in brackets and then give your answer on a newline. Like:

[Analysis [token from the user]]

Your answer here...
"""

trading_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=TradingDeps,
    retries=2
)

@trading_agent.tool
async def fetch_global_asset_information(ctx: RunContext[TradingDeps], token: str) -> str:
    """Fetch global asset information for the given cryptocurrency token.
    
    This function retrieves detailed asset data from the CryptoCompare API using the provided token symbol.
    It extracts key information such as the asset's creation date, description snippet, decimal points, various supply
    metrics (maximum, issued, total, circulating, future, locked, burnt, staked), and market capitalization details
    (total and circulating in USD). The function processes the API response to filter only the top-level primitive values
    and returns the results as a formatted string.
    
    Args:
        ctx: The execution context containing necessary dependencies.
        token: The cryptocurrency asset symbol (e.g., "BTC", "ETH") for which to fetch data.
    
    Returns:
        str: A formatted string containing the global asset information, or an error message if the data retrieval fails.
    """

    base_url = f'https://data-api.cryptocompare.com/asset/v1/data/by/symbol'
    params = {
            "asset_symbol": token,
            "api_key": ctx.deps.ccompare_api_key,
        }
    response = await ctx.deps.client.get(base_url, params=params)
    
    print(f"fetch global information for {token}")

    if response.status_code != 200:
        return f"Failed to retrieve asset data: {response.text}"
    
    data = response.json()
    main_info = data["Data"]
    # Build a new dictionary that only keeps keys with primitive values
    first_level_data = {
        key: value 
        for key, value in main_info.items() 
        if not isinstance(value, (dict, list))
    }

    return (
        f"created_on:{datetime.fromtimestamp(int(first_level_data['CREATED_ON']))}\n"
        f"asset_description_snippet:{first_level_data['ASSET_DESCRIPTION_SNIPPET']}\n"
        f"asset_decimal_points:{first_level_data['ASSET_DECIMAL_POINTS']}\n"
        f"name:{first_level_data['NAME']}\n"
        f"supply_max:{first_level_data['SUPPLY_MAX']}\n"
        f"supply_issued:{first_level_data['SUPPLY_ISSUED']}\n"
        f"supply_total:{first_level_data['SUPPLY_TOTAL']}\n"
        f"supply_circulating:{first_level_data['SUPPLY_CIRCULATING']}\n"
        f"supply_future:{first_level_data['SUPPLY_FUTURE']}\n"
        f"supply_locked:{first_level_data['SUPPLY_LOCKED']}\n"
        f"supply_burnt:{first_level_data['SUPPLY_BURNT']}\n"
        f"supply_staked:{first_level_data['SUPPLY_STAKED']}\n"
        f"total_market_cap_usd:{first_level_data['TOTAL_MKT_CAP_USD']}\n"
        f"circulating_market_cap_usd:{first_level_data['CIRCULATING_MKT_CAP_USD']}"
    )


@trading_agent.tool
async def fetch_historical_data(ctx: RunContext[TradingDeps], token: str) -> str:
    """Fetch historical daily price data for a given cryptocurrency token.
    
    This function retrieves historical price data from the CryptoCompare API for the specified token,
    using USD as the quote currency. It fetches up to 30 days of data, including details such as the 
    high, low, open, close, and volume for each day. The UNIX timestamps in the data are converted to 
    human-readable datetime format. The resulting records are then returned as a JSON formatted string.
    
    Args:
        ctx: The context containing necessary dependencies.
        token: The symbol of the cryptocurrency (e.g., 'BTC', 'ETH') to retrieve data for.
    
    Returns:
        str: A JSON formatted string containing an array of historical price data records, or an error message 
             (also in JSON) if data retrieval fails.
    """

    base_url = f'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
            "fsym": token,
            "tsym": "USD",
            "limit": 30,
            "api_key": ctx.deps.ccompare_api_key,
        }
    response = await ctx.deps.client.get(base_url, params=params)
    print(f"fetch historical data for {token}")
    
    if response.status_code != 200:
        return f"Failed to retrieve historical data: {response.text}"
    
    data = response.json()
    # print(json.dumps(data['Data']['Data'], indent=2))
    
    history = data['Data']['Data']
    records=[]

    # Iterate over each data point
     # Iterate over each data point and build a list of dictionaries
    for record in history:
        records.append({
            "time": datetime.fromtimestamp(int(record['time'])).strftime('%Y-%m-%d %H:%M:%S'),
            "high": record['high'],
            "low": record['low'],
            "open": record['open'],
            "close": record['close'],
            "volume_from": record['volumefrom'],
            "volume_to": record['volumeto']
        })

    # print(json.dumps(records, indent=2))    
    return json.dumps(records)

@trading_agent.tool
async def compute_technical_indicators(ctx: RunContext[TradingDeps], df_data: str) -> str:
    """Compute various technical indicators based on historical asset data.

    This function calculates several technical analysis metrics including moving averages,
    exponential moving averages, Bollinger Bands, RSI, and Ichimoku indicators. In addition,
    it computes the daily high/low values and calculates buying and selling momentum based on
    the last price relative to these daily extremes.

    Args:
        ctx: The context.
        df_data: A JSON formatted string containing the historical price data.

    Returns:
        str: A formatted string presenting the computed technical indicators.
    """
    # Convert the JSON string into a DataFrame
    df_data = ensure_closing_brace(df_data)
    data = json.loads(df_data)
    # Create a DataFrame from the "result" list
    df = pd.DataFrame(data['result'])

    if len(df) < 5:
        return "Not enough data to analyze"  

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

    print(df.tail(5))

    # Print the calculated insights
    return (
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
