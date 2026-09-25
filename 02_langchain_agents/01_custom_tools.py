"""
==============================================================================
01_custom_tools.py - Defining and Using Custom Tools in LangChain
==============================================================================

This script demonstrates:
1. Creating tools using the `@tool` decorator.
2. Generating rich tool descriptions and schemas for LLMs.
3. Defining strict Pydantic argument schemas with `args_schema`.
4. Inspecting tool parameters and executing tools directly.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Basic Tool Definition via @tool Decorator
# ---------------------------------------------------------------------------
@tool
def calculate_rectangle_area(width: float, height: float) -> float:
    """Calculates the area of a rectangle given its width and height."""
    return width * height


# ---------------------------------------------------------------------------
# 2. Advanced Tool with Strict Pydantic Schema
# ---------------------------------------------------------------------------
class StockLookupInput(BaseModel):
    ticker: str = Field(description="The 1-4 uppercase stock ticker symbol, e.g. AAPL, GOOGL, NVDA")
    include_currency: bool = Field(default=True, description="Whether to include currency symbol in output")


# Mock database for stock prices
MOCK_PRICES = {
    "AAPL": 225.50,
    "GOOGL": 165.20,
    "NVDA": 118.75,
    "MSFT": 428.10,
}


@tool("stock_price_lookup", args_schema=StockLookupInput)
def get_stock_price(ticker: str, include_currency: bool = True) -> str:
    """Retrieves current market price for a specified stock ticker symbol."""
    clean_ticker = ticker.strip().upper()
    price = MOCK_PRICES.get(clean_ticker)
    if price is None:
        return f"Error: Ticker '{clean_ticker}' not found."
    
    currency_prefix = "$" if include_currency else ""
    return f"{clean_ticker}: {currency_prefix}{price:.2f}"


def main():
    print("--- 1. Inspecting Rectangle Area Tool ---")
    print(f"Tool Name:        {calculate_rectangle_area.name}")
    print(f"Tool Description: {calculate_rectangle_area.description}")
    print(f"Tool Arguments:   {calculate_rectangle_area.args}")

    # Direct invocation
    area_result = calculate_rectangle_area.invoke({"width": 12.5, "height": 4.0})
    print(f"Direct Execution Result: {area_result}")

    print("\n--- 2. Inspecting Stock Price Lookup Tool ---")
    print(f"Tool Name:        {get_stock_price.name}")
    print(f"Tool Description: {get_stock_price.description}")
    print(f"Tool JSON Schema:\n{get_stock_price.args_schema.model_json_schema()}")

    # Direct invocation
    stock_result = get_stock_price.invoke({"ticker": "nvda", "include_currency": True})
    print(f"Direct Execution Result: {stock_result}")


if __name__ == "__main__":
    main()
