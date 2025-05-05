"""
AI and Crypto Integration Demo

This script demonstrates how to use AI models and crypto tools together
for Lucky Train, showing practical applications for crypto trading and analysis.
"""

import os
import logging
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import our integrations
from ai_model_integrations import create_model_interface
from crypto_tools_integration import CryptoToolFactory
from image_model_integrations import create_image_model_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_crypto_market_data(asset="BTC", days=7):
    """Get cryptocurrency market data using Glassnode.
    
    Args:
        asset: Asset symbol
        days: Number of days of data to retrieve
        
    Returns:
        Market data in pandas DataFrame
    """
    logger.info(f"Getting {asset} market data for the last {days} days...")
    
    # Initialize Glassnode API
    glassnode_config = {}
    glassnode_api = CryptoToolFactory.create_tool("glassnode", glassnode_config)
    
    if glassnode_api.get_status().get("status") != "connected":
        logger.error("Failed to connect to Glassnode API")
        return None
    
    # Get price data
    from_date = int((datetime.now() - timedelta(days=days)).timestamp())
    to_date = int(datetime.now().timestamp())
    
    price_data = glassnode_api.get_metric("market/price_usd_close", asset, from_date, to_date)
    
    if "error" in price_data:
        logger.error(f"Error getting price data: {price_data['error']}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(price_data.get("data", []))
    if df.empty:
        logger.warning("No price data returned")
        return None
    
    # Convert timestamps to datetime
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df.rename(columns={'t': 'date', 'v': 'price'}, inplace=True)
    
    return df

def analyze_market_data_with_ai(df):
    """Analyze market data using AI models.
    
    Args:
        df: DataFrame with market data
        
    Returns:
        Analysis results
    """
    if df is None or df.empty:
        logger.error("No data to analyze")
        return "Insufficient data for analysis"
    
    # Calculate price change
    price_change = ((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]) * 100
    price_change_str = f"Price change: {price_change:.2f}%"
    
    # Format data for AI analysis
    data_summary = f"""
Bitcoin price data for the last {len(df)} days:
- Start price: ${df['price'].iloc[0]:.2f}
- End price: ${df['price'].iloc[-1]:.2f}
- {price_change_str}
- Highest price: ${df['price'].max():.2f}
- Lowest price: ${df['price'].min():.2f}
"""
    
    # Initialize AI model for analysis
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Analyzing market data with GPT...")
        gpt_config = {"model": "gpt-3.5-turbo"}
        gpt_model = create_model_interface("openai", gpt_config)
        
        analysis_prompt = f"""
Based on the following Bitcoin price data, provide a brief market analysis and outlook:
{data_summary}

Provide your analysis in a concise format with these sections:
1. Market sentiment (bullish/bearish/neutral)
2. Key observations
3. Short-term outlook (1-2 weeks)
"""
        
        response = gpt_model.generate(
            analysis_prompt,
            system_message="You are a crypto market analyst with expertise in technical analysis. Provide concise, data-driven insights."
        )
        
        analysis = response.get('text', 'Analysis failed')
    else:
        analysis = "OpenAI API key not set - Cannot perform AI analysis"
    
    return analysis

def generate_trading_strategy(market_analysis, asset="BTC"):
    """Generate a trading strategy based on market analysis.
    
    Args:
        market_analysis: Text analysis of the market
        asset: Asset to trade
        
    Returns:
        Trading strategy code
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Generating trading strategy with Claude...")
        claude_config = {"model": "claude-3-sonnet-20240229"}
        claude_model = create_model_interface("anthropic", claude_config)
        
        strategy_prompt = f"""
Based on the following market analysis for {asset}, create a Python trading strategy for Trality:

{market_analysis}

The strategy should:
1. Use the Trality API format
2. Include clear buy/sell signals
3. Include risk management with stop-loss
4. Be implemented as a complete Python function

Please provide only the Python code without explanations.
"""
        
        response = claude_model.generate(
            strategy_prompt,
            system_message="You are an expert crypto trading bot developer. Generate clean, effective trading strategies based on market analysis."
        )
        
        strategy_code = response.get('text', 'Strategy generation failed')
    else:
        strategy_code = "Anthropic API key not set - Cannot generate trading strategy"
    
    return strategy_code

def visualize_market_data(df):
    """Create visualization of market data.
    
    Args:
        df: DataFrame with market data
        
    Returns:
        Path to saved visualization
    """
    if df is None or df.empty:
        logger.error("No data to visualize")
        return None
    
    # Create directory for output
    os.makedirs("output", exist_ok=True)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], 'b-')
    plt.title('Bitcoin Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    
    # Add annotations
    plt.annotate(f"${df['price'].iloc[-1]:.2f}", 
                xy=(df['date'].iloc[-1], df['price'].iloc[-1]),
                xytext=(10, 0),
                textcoords='offset points')
    
    # Save the plot
    output_file = f"output/btc_price_{int(time.time())}.png"
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Price chart saved to {output_file}")
    return output_file

def analyze_chart_with_vision(chart_path):
    """Analyze price chart using GPT-4 Vision.
    
    Args:
        chart_path: Path to the chart image
        
    Returns:
        Vision analysis
    """
    if not chart_path or not os.path.exists(chart_path):
        logger.error("Chart image not found")
        return "Chart image not available for analysis"
    
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Analyzing chart with GPT-4 Vision...")
        gpt4v_config = {"model": "gpt-4-vision-preview"}
        gpt4v = create_image_model_interface("gpt4-vision", gpt4v_config)
        
        prompt = """
Analyze this Bitcoin price chart and provide your observations:
1. Identify key trends or patterns
2. Identify potential support and resistance levels
3. What trading opportunities might exist based on this chart?
4. Any warning signals or concerns?
"""
        
        response = gpt4v.generate(prompt, image_path=chart_path)
        
        if "error" in response:
            logger.error(f"Vision analysis failed: {response['error']}")
            return "Vision analysis failed"
        
        return response.get('analysis', 'No analysis provided')
    else:
        return "OpenAI API key not set - Cannot perform vision analysis"

def generate_market_visualization(market_analysis):
    """Generate an AI-created visualization of the market scenario.
    
    Args:
        market_analysis: Text analysis of the market
        
    Returns:
        Path to saved visualization
    """
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Generating market visualization with DALL-E...")
        dalle_config = {"model": "dall-e-3", "size": "1024x1024"}
        dalle = create_image_model_interface("dalle", dalle_config)
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        save_path = f"output/market_scenario_{int(time.time())}.png"
        
        # Extract sentiment from analysis
        sentiment = "neutral"
        if "bullish" in market_analysis.lower():
            sentiment = "bullish"
        elif "bearish" in market_analysis.lower():
            sentiment = "bearish"
        
        # Generate image
        prompt = f"""
Create a visual representation of a {sentiment} Bitcoin market scenario. 
The image should be a professional, financial visualization that might appear in a crypto trading platform or financial news site.
Include visual elements that represent the {sentiment} trend, market volatility, and trading activity.
Use colors that reflect the market sentiment: green for bullish, red for bearish, or yellow/blue for neutral.
"""
        
        response = dalle.generate(
            prompt=prompt, 
            save_to=save_path
        )
        
        if "error" in response:
            logger.error(f"DALL-E generation failed: {response['error']}")
            return None
        
        return response.get('saved_to')
    else:
        logger.warning("OpenAI API key not set - Cannot generate visualization")
        return None

def create_trading_bot(strategy_code, bot_name="LuckyTrain_AI_Bot"):
    """Create a trading bot on Trality using the generated strategy.
    
    Args:
        strategy_code: Python code for the trading strategy
        bot_name: Name for the bot
        
    Returns:
        Bot creation result
    """
    if os.getenv("TRALITY_API_KEY"):
        logger.info("Creating trading bot on Trality...")
        trality_api = CryptoToolFactory.create_tool("trality")
        
        if trality_api.get_status().get("status") != "connected":
            logger.error("Failed to connect to Trality API")
            return "Failed to connect to Trality"
        
        # Create the bot
        result = trality_api.create_bot(
            name=bot_name, 
            strategy_code=strategy_code,
            exchange="binance"
        )
        
        if result.get("status") == "success":
            return f"Bot created successfully with ID: {result.get('bot_id')}"
        else:
            return f"Bot creation failed: {result.get('error')}"
    else:
        return "Trality API key not set - Cannot create bot"

def main():
    """Main function that demonstrates the AI and crypto integration workflow."""
    
    logger.info("=" * 70)
    logger.info("Starting AI and Crypto Integration Demo")
    logger.info("=" * 70)
    
    # Step 1: Get cryptocurrency market data
    market_data = get_crypto_market_data(asset="BTC", days=30)
    if market_data is not None:
        logger.info(f"Retrieved {len(market_data)} days of Bitcoin data")
    else:
        logger.error("Failed to retrieve market data")
        return
    
    # Step 2: Visualize the data
    chart_path = visualize_market_data(market_data)
    
    # Step 3: Analyze the market data with AI
    market_analysis = analyze_market_data_with_ai(market_data)
    logger.info("\nMarket Analysis:")
    logger.info("-" * 70)
    logger.info(market_analysis)
    logger.info("-" * 70)
    
    # Step 4: Analyze the chart with vision AI
    if chart_path:
        vision_analysis = analyze_chart_with_vision(chart_path)
        logger.info("\nChart Vision Analysis:")
        logger.info("-" * 70)
        logger.info(vision_analysis)
        logger.info("-" * 70)
    
    # Step 5: Generate trading strategy based on analysis
    strategy_code = generate_trading_strategy(market_analysis)
    logger.info("\nGenerated Trading Strategy:")
    logger.info("-" * 70)
    logger.info(strategy_code[:500] + "..." if len(strategy_code) > 500 else strategy_code)
    logger.info("-" * 70)
    
    # Step 6: Create a visual representation of the market scenario
    scenario_image = generate_market_visualization(market_analysis)
    if scenario_image:
        logger.info(f"Market scenario visualization saved to: {scenario_image}")
    
    # Step 7: Create a trading bot (commented out to avoid actual bot creation)
    # bot_result = create_trading_bot(strategy_code)
    # logger.info(f"Bot creation result: {bot_result}")
    
    logger.info("=" * 70)
    logger.info("AI and Crypto Integration Demo Completed")
    logger.info("=" * 70)

if __name__ == "__main__":
    main() 