from __future__ import annotations as _annotations

import asyncio
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from typing import Dict, Union
from dotenv import load_dotenv
import pandas as pd

import httpx
import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from prompt import light_prompt

load_dotenv()

base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')
is_ollama = "localhost" in base_url.lower()

primary_llm_model = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
model = OpenAIModel(primary_llm_model, base_url=base_url, api_key=api_key)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class TradingDeps:
    client: httpx.AsyncClient
    ccompare_api_key: str | None

trading_agent = Agent(
    model=model,
    system_prompt=light_prompt,
    deps_type=TradingDeps,
    retries=2
)


@trading_agent.tool
async def get_market_metrics(ctx: RunContext[TradingDeps], token: str) -> Union[Dict, str]:
    """Fetch key market metrics, such as market cap, circulating supply, trading volume, and volatility index for a given token.
    
    Args:
        ctx: The context.
        token: The token we would like to search for market metrics.
    """

    base_url = f'https://data-api.cryptocompare.com/asset/v1/data/by/symbol'
    params = {
            "asset_symbol": token,
            "api_key": ctx.deps.ccompare_api_key,
        }
    
    try:
        
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

        return {
            'created_on': datetime.fromtimestamp(int(first_level_data['CREATED_ON'])),
            'asset_description_snippet': first_level_data['ASSET_DESCRIPTION_SNIPPET'],
            'asset_decimal_points': first_level_data['ASSET_DECIMAL_POINTS'],
            'name': first_level_data['NAME'],
            'supply_max': first_level_data['SUPPLY_MAX'],
            'supply_issued': first_level_data['SUPPLY_ISSUED'],
            'supply_total': first_level_data['SUPPLY_TOTAL'],
            'supply_circulating': first_level_data['SUPPLY_CIRCULATING'],
            'supply_future': first_level_data['SUPPLY_FUTURE'],
            'supply_locked': first_level_data['SUPPLY_LOCKED'],
            'supply_burnt': first_level_data['SUPPLY_BURNT'],
            'supply_staked': first_level_data['SUPPLY_STAKED'],
            'total_market_cap_usd': first_level_data['TOTAL_MKT_CAP_USD'],
            'circulating_market_cap_usd': first_level_data['CIRCULATING_MKT_CAP_USD']
        }
            
    except Exception as e:
        return f"Error fetching asset data: {str(e)}"

@trading_agent.tool
async def get_crypto_prices(ctx: RunContext[TradingDeps], token: str) -> Union[Dict, str]:
    """Fetches historical crypto price data and technical indicator for a given tocken.

    Args:
        ctx: The context.
        token: The token we would like to search for prices and technical indicator.
    """

    base_url = f'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
            "fsym": token,
            "tsym": "USD",
            "limit": timedelta(weeks=24*1).days,
            "api_key": ctx.deps.ccompare_api_key,
        }
    
    try:
        response = await ctx.deps.client.get(base_url, params=params)
        print(f"fetch historical data for {token} until {timedelta(weeks=24*1).days} days")
        
        if response.status_code != 200:
            return f"Failed to retrieve historical data: {response.text}"
        
        data = response.json()
        # print(json.dumps(data['Data']['Data'], indent=2))
        # print(f"Data found {data}")

        history = data['Data']['Data']
        # print(f"History found {history}")

        records=[]

        # Iterate over each data point and build a list of dictionaries
        for record in history:
            records.append({
                "time": datetime.fromtimestamp(int(record['time'])).strftime('%Y-%m-%d %H:%M:%S'),
                "High": record['high'],
                "Low": record['low'],
                "Open": record['open'],
                "Close": record['close'],
                # You can choose which volume column to use. For example, use "volume_to" as "Volume":
                # "volume_from": record['volumefrom'],
                "Volume": record['volumeto']
            })

        # Convert the JSON string into a DataFrame
        # records = ensure_closing_brace(records)
        # data = json.loads(records)
        # Create a DataFrame from the "result" list
        df = pd.DataFrame(records)

        if len(df) < 5:
            return "Not enough data to analyze" 

        # Convert the "time" column to datetime and extract just the date
        df['Date'] = pd.to_datetime(df['time']).dt.date

        # Select and reorder only the desired columns
        df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

        # Optionally, sort the DataFrame by Date
        df = df.sort_values('Date').reset_index(drop=True)

        indicators = {}
        
        last_price = df['Close'].iloc[-1]
        rolling_avg = df['Close'].rolling(window=7).mean().iloc[-1]  # 7-day rolling average
        indicators["Price Change"] = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) >= 2 else 0
        indicators["Volume Change"] = df['Volume'].iloc[-1] - df['Volume'].iloc[-2] if len(df) >= 2 else 0

        indicators["EMA7"] = df['Close'].ewm(span=7, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        indicators["EMA20"] = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        indicators["EMA50"] = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        indicators["EMA100"] = df['Close'].ewm(span=100, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        indicators["EMA200"] = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        std = df['Close'].rolling(window=5).std().iloc[-1]  # Standard deviation for Bollinger Bands
        indicators["Bollinger Upper Band"] = rolling_avg + (2 * std)
        indicators["Bollinger Lower Band"] = rolling_avg - (2 * std)
        indicators["7-Day Rolling Average"] = rolling_avg

        # Calculate Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else float('nan')
        indicators["RSI"] = 100 - (100 / (1 + rs))

        # Calculate Ichimoku indicators
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
        nine_period_high = df['High'].rolling(window= 9).max().iloc[-1]
        nine_period_low = df['Low'].rolling(window= 9).min().iloc[-1]
        indicators["Tenkan Sen"] = (nine_period_high + nine_period_low) /2
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = df['High'].rolling(window=26).max().iloc[-1]
        period26_low = df['Low'].rolling(window=26).min().iloc[-1]
        indicators["Kijun Sen"] = (period26_high + period26_low) / 2
    
        # Calculate Daily High/Low and Momentum
        daily_high = df['High'].iloc[-1]
        daily_low = df['Low'].iloc[-1]
        indicators["Buying Momentum"] = last_price - daily_low    # Distance of the last price from the day's low
        indicators["Selling Momentum"] = daily_high - last_price   # Distance of the day's high from the last price
        indicators["Daily High"] = daily_high
        indicators["Daily Low"] = daily_low

        return {'stock_price': df.to_dict(orient='records'), 'indicators': indicators}
    except Exception as e:
        print(f"Error fetching price data: {str(e)}")
        return f"Error fetching price data: {str(e)}"

async def main():
    async with httpx.AsyncClient() as client:
        deps = TradingDeps(
            client=client,
            ccompare_api_key=os.getenv('CCOMPARE_API_KEY')
        )
        result = await trading_agent.run(
            'What is your analysis on Bitcoin ?', deps=deps
        )
        # debug(result)
        print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())