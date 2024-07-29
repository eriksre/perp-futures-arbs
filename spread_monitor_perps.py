import pandas as pd
import requests
import numpy as np
import time
import threading
import queue 
from datetime import datetime
import certifi
import re
import urllib3
from spread_manager_class import SpreadManager

#uncomment this if you are having issues with your SSL certificate
#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ca_bundle_path = certifi.where()



def fetch_binance_data(q):
    # Endpoint URLs
    url_futures_ticker = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    url_futures_funding_rate = "https://fapi.binance.com/fapi/v1/premiumIndex"
    url_futures_24hr = "https://fapi.binance.com/fapi/v1/ticker/24hr"  # New URL for futures 24hr ticker data


    #futures_ticker_response = requests.get(url_futures_ticker).json()
    futures_ticker_response = requests.get(url_futures_ticker, verify=True).json()
    futures_funding_rate_response = requests.get(url_futures_funding_rate, verify=True).json()
    futures_24hr_response = requests.get(url_futures_24hr, verify=True).json()  # Fetch futures 24hr data

    # Filter and prepare futures ticker data
    futures_data = [
        {'symbol': item['symbol'], 'binance_futures_bid_price': item['bidPrice'], 'binance_futures_ask_price': item['askPrice']}
        for item in futures_ticker_response if item['symbol'].endswith('USDT')
    ]

    # Prepare futures funding rate data
    funding_rate_data = [
        {'symbol': item['symbol'], 'binance_funding_rate': item['lastFundingRate']}
        for item in futures_funding_rate_response if item['symbol'].endswith('USDT')
    ]

    # Prepare futures volume data from the 24hr ticker response
    futures_volume_data = [
        {'symbol': item['symbol'], 'binance_futures_volume_usdt': item['quoteVolume']}
        for item in futures_24hr_response if item['symbol'].endswith('USDT')
    ]

    # Convert lists to DataFrames
    df_futures = pd.DataFrame(futures_data)
    df_funding_rate = pd.DataFrame(funding_rate_data)
    df_futures_volume = pd.DataFrame(futures_volume_data)

    # Merge DataFrames using 'outer' to ensure no symbols are dropped
    df_merged = pd.merge(df_futures, df_funding_rate, on='symbol', how='outer')
    df_merged = pd.merge(df_merged, df_futures_volume, on='symbol', how='outer')

    #return df_merged
    q.put(df_merged)



def fetch_bitget_data(q):
    # URLs for Bitget's spot and futures markets
    futures_url = "https://api.bitget.com/api/v2/mix/market/tickers"
    
    # Fetching futures market data with the specific parameter for USDT futures
    params = {'productType': 'USDT-FUTURES'}
    futures_response = requests.get(futures_url, params=params, verify=True).json()
    futures_data = futures_response.get('data', [])
    
    # Filtering and preparing futures market data
    futures_contracts = [{'symbol': contract['symbol'],
                          'bitget_funding_rate': float(contract['fundingRate']),
                          'bitget_futures_bid_price': float(contract['bidPr']) if contract['bidPr'] is not None else None,
                          'bitget_futures_ask_price': float(contract['askPr']) if contract['askPr'] is not None else None,
                          'bitget_futures_volume_usdt': float(contract['usdtVolume'])}  # Assuming 'usdtVolume' is correct for futures volume
                         for contract in futures_data if contract['symbol'].endswith('USDT')]
    
    # Creating DataFrames
    futures_df = pd.DataFrame(futures_contracts)
    
    # Merging the DataFrames on 'symbol', merging all columns including the volumes
    merged_df = futures_df
    
    #return merged_df
    q.put(merged_df)


def fetch_blofin_data(q):
    url = 'https://openapi.blofin.com/api/v1/market/tickers'


    # Use certifi to get the path of the CA bundle file
    #False = certifi.where()

    response = requests.get(url, verify=True).json()
    data = response['data']

    # Filter for symbols ending with '-USDT' and prepare the data for the DataFrame
    processed_data = []
    for item in data:
        if item['instId'].endswith('-USDT'):
            symbol = item['instId'].replace('-', '')  # Remove '-' from the symbol
            try:
                bid_price = float(item.get('bidPrice', np.nan))
                ask_price = float(item.get('askPrice', np.nan))
                average_price = (bid_price + ask_price) / 2 if bid_price and ask_price else np.nan
                volume = float(item.get('volCurrency24h', np.nan))
                volume_value = volume * average_price if volume and not np.isnan(average_price) else np.nan
                processed_data.append({
                    'symbol': symbol, 
                    'blofin_futures_bid_price': bid_price,
                    'blofin_futures_ask_price': ask_price,  
                    'blofin_futures_volume_usdt': volume_value
                })
            except ValueError:
                # In case of any conversion error, append NaN values
                processed_data.append({'symbol': symbol, 'blofin_futures_price': np.nan, 'blofin_futures_volume_usdt': np.nan})

    # Creating a DataFrame from the processed data
    df = pd.DataFrame(processed_data)
    
    #return df
    q.put(df)


def fetch_bybit_data(q):
    # Define the base URL for the Bybit API
    base_url = "https://api.bybit.com/v5/market/tickers"
    
    # Initialize a dictionary to store combined data
    combined_data = {}
    
    # Fetch futures data (linear category)
    futures_response = requests.get(base_url, params={'category': 'linear'}, verify=True)
    futures_data = futures_response.json().get('result', {}).get('list', [])
    
    # Process futures data
    for item in futures_data:
        symbol = item.get('symbol')
        if symbol.endswith('USDT'):  # Check if symbol ends with 'USDT'
            funding_rate = item.get('fundingRate')
            bid_price = float(item.get('bid1Price'))
            ask_price = float(item.get('ask1Price'))
            avg_price = (bid_price + ask_price) / 2
            futures_volume = float(item.get('volume24h', 0)) * avg_price  # Multiply volume by avg_price
            
            combined_data[symbol] = {
                'bybit_futures_bid_price': bid_price,
                'bybit_futures_ask_price': ask_price,
                'bybit_funding_rate': funding_rate,
                'bybit_futures_volume_usdt': futures_volume  # Add futures volume
            }

    # Convert the combined data into a DataFrame
    df_combined = pd.DataFrame.from_dict(combined_data, orient='index').reset_index().rename(columns={'index': 'symbol'})
    
    #return df_combined
    q.put(df_combined)


def fetch_gateio_data(q):
    # Define URLs for the Gate.io API endpoints
    futures_url = "https://api.gateio.ws/api/v4/futures/usdt/tickers"
    
    # Make the API requests
    futures_response = requests.get(futures_url, verify=True)
    futures_data = futures_response.json()
    
    # Initialize a dictionary to store the data, filtering for USDT-denominated pairs
    data_dict = {}
    
    # Process futures data
    for item in futures_data:
        # Normalize the symbol format and ensure it's USDT-denominated
        symbol = item['contract'].replace('_', '')
        if 'USDT' in symbol:
            data_dict[symbol] = {
                'gateio_futures_bid_price': float(item.get('highest_bid')),
                'gateio_futures_ask_price': float(item.get('lowest_ask')),
                'gateio_funding_rate': item.get('funding_rate'),
                'gateio_futures_volume_usdt': float(item.get('volume_24h_quote', 0))  # Default to 0 if not available
            }


        # Function to safely convert string to float, defaulting to 0 if conversion fails
    def safe_float_conversion(value, default=0.0):
        try:
            return float(value) if value else default
        except ValueError:
            return default


    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index().rename(columns={'index': 'symbol'})
    
    #return df
    q.put(df)


def fetch_mexc_data(q):
    # URLs for both APIs
    futures_url = "https://contract.mexc.com/api/v1/contract/ticker"
    
    data_dict = {}  # Initialize a dictionary to store combined data


    try:
        # Fetch futures data
        futures_response = requests.get(futures_url, verify=True).json()
        for item in futures_response['data']:
            if item['symbol'].endswith("USDT"):
                symbol = item['symbol'].replace('_', '')  # Standardize symbol format
                funding_rate = float(item['fundingRate']) if item.get('fundingRate', '') else None
                bid = float(item['bid1']) if item.get('bid1', '') else None
                ask = float(item['ask1']) if item.get('ask1', '') else None
                average_price = (bid + ask) / 2 if bid and ask else None
                futures_volume = float(item['amount24']) if item.get('amount24', '') else None
                
                # Store futures data in the dictionary
                data_dict[symbol] = {
                    'mexc_futures_bid_price': bid,
                    'mexc_futures_ask_price': ask,
                    'mexc_funding_rate': funding_rate,
                    'mexc_futures_volume_usdt': futures_volume
                }

        data_df = pd.DataFrame.from_dict(data_dict, orient='index')
        data_df.index.name = 'symbol'

        q.put(data_df)

        #return data_df
    except Exception as e:
        print(f"Error fetching MEXC data: {e}")



def fetch_okx_data(q):
    # URLs for API requests
    tickers_url = 'https://www.okx.com/api/v5/market/tickers'

    # Fetch Swap (Futures) Contracts
    response_swap = requests.get(tickers_url, params={'instType': 'SWAP'}, verify=True)
    swap_data = response_swap.json() if response_swap.status_code == 200 else None
    
    # Process Swap Contracts
    swap_contracts = []
    if swap_data and 'data' in swap_data:
        for contract in swap_data['data']:
            if '-USDT' in contract['instId']:
                symbol_processed = contract['instId'].replace('-', '').replace('SWAP', '')
                average_price = (float(contract['askPx']) + float(contract['bidPx'])) / 2
                volume = float(contract['volCcy24h']) * average_price  # Calculate volume
                
                swap_contracts.append({
                    'symbol': symbol_processed,
                    'okx_futures_ask_price': float(contract['askPx']),
                    'okx_futures_bid_price': float(contract['bidPx']),
                    'okx_futures_volume_usdt': volume,  # Add volume to the dictionary
                })

    # Merge DataFrames
    df_swap = pd.DataFrame(swap_contracts)
    df_final = df_swap
    
    #return df_final
    q.put(df_final)



def fetch_xtcom_data(q):
    # Fetch Futures Symbol List
    futures_url = 'https://fapi.xt.com/future/market/v3/public/symbol/list'
    futures_response = requests.get(futures_url).json()
    futures_symbols = [
        symbol['symbol']
        for symbol in futures_response['result']['symbols']
        if symbol['quoteCoin'] == 'usdt' and symbol['contractType'] == 'PERPETUAL'
    ]

    # Get Bid and Ask for Futures
    futures_tickers_url = 'https://fapi.xt.com/future/market/v1/public/q/agg-tickers'
    futures_tickers_response = requests.get(futures_tickers_url).json()
    futures_bid_ask_data = {item['s']: item for item in futures_tickers_response['result']}

    # Prepare data for futures DataFrame
    futures_data = []
    for symbol in futures_symbols:
        if symbol in futures_bid_ask_data:
            item = futures_bid_ask_data[symbol]
            bp = item.get('bp')
            ap = item.get('ap')
            if bp is not None and ap is not None:
                bid_price = float(bp)
                ask_price = float(ap)
                average_price = (bid_price + ask_price) / 2
                volume = float(item['v']) * average_price  # Adjusted volume calculation
                futures_data.append({
                    "symbol": symbol.replace('_', '').upper(),
                    "xtcom_futures_bid_price": bid_price,
                    "xtcom_futures_ask_price": ask_price,
                    "xtcom_futures_volume_usdt": volume  # Adjusted volume key
                })

    df_futures = pd.DataFrame(futures_data)

    # Merging the DataFrames
    merged_df = df_futures

    q.put(merged_df)


def merge_exchange_dataframes():
    # Create a queue
    df_queue = queue.Queue()


    # Fetch data from each source
    binance_thread = threading.Thread(target=fetch_binance_data, args=(df_queue,))
    bitget_thread = threading.Thread(target=fetch_bitget_data, args=(df_queue,)) 
    blofin_thread = threading.Thread(target=fetch_blofin_data, args=(df_queue,))
    bybit_thread = threading.Thread(target=fetch_bybit_data, args=(df_queue,))
    gateio_thread = threading.Thread(target=fetch_gateio_data, args=(df_queue,))
    mexc_thread = threading.Thread(target=fetch_mexc_data, args=(df_queue,))
    okx_thread = threading.Thread(target=fetch_okx_data, args=(df_queue,))
    xtcom_thread = threading.Thread(target=fetch_xtcom_data, args=(df_queue,))

    # Start threads
    binance_thread.start()
    bitget_thread.start()
    blofin_thread.start()
    bybit_thread.start()
    gateio_thread.start()
    mexc_thread.start()
    okx_thread.start()
    xtcom_thread.start()

    # Join threads
    binance_thread.join()
    bitget_thread.join()
    blofin_thread.join()
    bybit_thread.join()
    gateio_thread.join()
    mexc_thread.join()
    okx_thread.join()
    xtcom_thread.join()


    # Retrieve the DataFrames from the queue and merge them
    merged_data = None
    while not df_queue.empty():
        df = df_queue.get()
        if merged_data is None:
            merged_data = df
        else:
            merged_data = pd.merge(merged_data, df, on='symbol', how='outer', suffixes=('', '_drop'))

    # Clean up columns (if you have overlapping column names, this part is crucial)
    # This might not be necessary if your DataFrames are structured to merge cleanly
    if merged_data is not None:
        merged_data = merged_data[[c for c in merged_data.columns if not c.endswith('_drop')]]

    return merged_data

def remove_multiple_of_10(df):

    # Extract the leading number if it's a multiple of 10
    def extract_if_multiple_of_10(s):
        match = re.match(r'^(\d+)', s)
        if match:
            number = int(match.group(1))
            return number if number % 10 == 0 else np.nan  # Use np.nan for non-multiples of 10
        return np.nan

    # Apply the function to extract the numbers
    df['extracted_number'] = df['symbol'].apply(extract_if_multiple_of_10)

    # Define the price and volume columns
    price_columns = [col for col in df.columns if 'price' in col]
    volume_columns = [col for col in df.columns if 'volume' in col]

    # Adjust the price and volume columns
    for index, row in df.iterrows():
        multiplier = row['extracted_number']
        # Check if multiplier is valid
        if pd.notnull(multiplier) and multiplier != 0:
            for col in price_columns:
                # Only adjust if the price is not null or zero
                if pd.notnull(row[col]) and row[col] != 0:
                    df.at[index, col] = row[col] / multiplier
            for col in volume_columns:
                # Only adjust if the volume is not null or zero
                if pd.notnull(row[col]) and row[col] != 0:
                    df.at[index, col] = row[col] * multiplier

    # Remove the leading numbers from the symbols and create 'adjusted_symbol'
    df['adjusted_symbol'] = df['symbol'].str.replace(r'^\d+', '', regex=True)

    # Make sure the 'symbol' column is retained until this point
    # Now you can drop or modify it if needed

    # Aggregate the data based on 'adjusted_symbol'
    aggregation_functions = {col: 'mean' if col in price_columns else 'sum' for col in df.columns if col not in ['symbol', 'extracted_number', 'adjusted_symbol']}
    # Adding 'first' to keep the first occurrence of non-numeric values
    aggregation_functions['symbol'] = 'first'
    aggregated_df = df.groupby('adjusted_symbol').aggregate(aggregation_functions)

    return aggregated_df

def calculate_adjusted_prices(df):
    # Convert all columns except 'symbol' to numeric to avoid type errors
    def convert_columns_to_numeric_except_symbol(df):
        cols_to_convert = [col for col in df.columns if col != 'symbol']  # Exclude the 'symbol' column
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' converts errors to NaN
        return df

    # Apply the conversion function to the DataFrame
    df = convert_columns_to_numeric_except_symbol(df)

    # Helper function to calculate adjusted prices for a given type (mid, bid, ask)
    def calculate_adjusted_column(df, price_type, adjusted_suffix):
        price_columns = [col for col in df.columns if price_type in col]
        funding_rate_columns = [col for col in df.columns if 'funding_rate' in col]

        for price_col in price_columns:
            exchange_name = price_col.split('_' + price_type)[0]  # Extract exchange name from column
            funding_rate_col = exchange_name + '_funding_rate'  # Construct funding rate column name

            if funding_rate_col in funding_rate_columns:
                adjusted_price_col = exchange_name + adjusted_suffix
                if adjusted_price_col not in df.columns:
                    df[adjusted_price_col] = df[price_col] * (1 + df[funding_rate_col])

    # Calculate adjusted prices for mid, bid, and ask
    calculate_adjusted_column(df, 'futures_bid_price', '_adjusted_futures_bid_price')
    calculate_adjusted_column(df, 'futures_ask_price', '_adjusted_futures_ask_price')

    return df





def calculate_spread(df):
    # Convert all columns except 'symbol' to numeric to avoid type errors
    cols_to_convert = [col for col in df.columns if col != 'symbol']  # Exclude the 'symbol' column
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')  # 'coerce' converts errors to NaN

    volume_threshold = 200000

    # Instrument columns
    #futures_columns = [col for col in df.columns if 'futures_price' in col and 'adjusted' not in col]
    #spot_columns = [col for col in df.columns if 'spot_price' in col]
    #funding_rate_columns = [col for col in df.columns if 'funding_rate' in col]

    # Automatically identify exchange columns by extracting unique prefixes, excluding 'symbol'
    #exchange_columns = {col.split('_')[0]: [col] for col in df.columns if col != 'symbol'}



    def apply_price_difference_filter(prices):
        valid_prices = prices.dropna()
        if not valid_prices.empty:
            min_price = valid_prices.min()
            return prices.where(valid_prices <= 10 * min_price, np.nan)
        return prices

    def calculate_adjusted_futures_spread(df):
        bid_price_columns = [col for col in df.columns if 'adjusted_futures_bid_price' in col]
        ask_price_columns = [col for col in df.columns if 'adjusted_futures_ask_price' in col]

        price_to_volume = {price: price.replace('adjusted_futures_bid_price', 'futures_volume_usdt').replace('adjusted_futures_ask_price', 'futures_volume_usdt') for price in bid_price_columns + ask_price_columns}

        for col in bid_price_columns + ask_price_columns:
            volume_col = price_to_volume[col]
            df.loc[(df[volume_col] < volume_threshold) | (df[col] == 0), col] = np.nan

        filtered_bid_prices = df[bid_price_columns].apply(apply_price_difference_filter, axis=1)
        filtered_ask_prices = df[ask_price_columns].apply(apply_price_difference_filter, axis=1)

        min_ask_prices = filtered_ask_prices.min(axis=1)
        max_bid_prices = filtered_bid_prices.max(axis=1)


        min_ask_indices = pd.Series(index=df.index, dtype=float)
        max_bid_indices = pd.Series(index=df.index, dtype=float)

        for i in df.index:
            if not filtered_ask_prices.loc[i].isnull().all():
                min_ask_indices[i] = filtered_ask_prices.loc[i].idxmin()
            if not filtered_bid_prices.loc[i].isnull().all():
                max_bid_indices[i] = filtered_bid_prices.loc[i].idxmax()

        mean_bid_ask_price = (min_ask_prices + max_bid_prices) / 2
        mean_bid_ask_price.loc[(min_ask_indices != max_bid_indices) & (min_ask_prices.isnull() | max_bid_prices.isnull())] = np.nan

        df['adjusted_futures_spread_value'] = (max_bid_prices - min_ask_prices) / mean_bid_ask_price
        df['adjusted_futures_spread_value'] = df['adjusted_futures_spread_value'].replace([np.inf, -np.inf], np.nan)
        df['adjusted_futures_spread_instruments'] = max_bid_indices.astype(str) + ", " + min_ask_indices.astype(str)

        return df

    df = calculate_adjusted_futures_spread(df)

    def calculate_futures_spread(df):
        futures_bid_columns = [col for col in df.columns if 'futures_bid_price' in col and 'adjusted' not in col]
        futures_ask_columns = [col for col in df.columns if 'futures_ask_price' in col and 'adjusted' not in col]

        price_to_volume = {price: price.replace('bid_price', 'volume_usdt').replace('ask_price', 'volume_usdt') for price in futures_bid_columns + futures_ask_columns}

        filtered_bid_prices = pd.DataFrame(index=df.index, columns=futures_bid_columns)
        filtered_ask_prices = pd.DataFrame(index=df.index, columns=futures_ask_columns)

        for col in futures_bid_columns:
            volume_col = price_to_volume[col]
            filtered_bid_prices.loc[(df[volume_col] >= volume_threshold) & (df[col] != 0), col] = df[col]

        for col in futures_ask_columns:
            volume_col = price_to_volume[col]
            filtered_ask_prices.loc[(df[volume_col] >= volume_threshold) & (df[col] != 0), col] = df[col]

        filtered_bid_prices = filtered_bid_prices.apply(apply_price_difference_filter, axis=1)
        filtered_ask_prices = filtered_ask_prices.apply(apply_price_difference_filter, axis=1)

        max_bid_price = filtered_bid_prices.max(axis=1)
        min_ask_price = filtered_ask_prices.min(axis=1)

        max_bid_instrument = pd.Series(index=df.index, dtype=float)
        min_ask_instrument = pd.Series(index=df.index, dtype=float)

        for i in df.index:
            if not filtered_bid_prices.loc[i].isnull().all():
                max_bid_instrument[i] = filtered_bid_prices.loc[i].idxmax()
            if not filtered_ask_prices.loc[i].isnull().all():
                min_ask_instrument[i] = filtered_ask_prices.loc[i].idxmin()

        mean_bid_ask_price = (max_bid_price + min_ask_price) / 2
        df['futures_spread_value'] = (max_bid_price - min_ask_price) / mean_bid_ask_price
        df['futures_spread_value'] = df['futures_spread_value'].replace([np.inf, -np.inf], np.nan)

        df['futures_spread_instruments'] = max_bid_instrument.astype(str) + ", " + min_ask_instrument.astype(str)

        return df

    df = calculate_futures_spread(df)


    return df

def print_top_spreads(df):
    # Ensure the input is a pandas DataFrame

    # Assuming there's a symbol column for each spread type in your DataFrame
    categories = {
        'Futures': ('futures_spread_value', 'futures_spread_instruments', 'symbol'),
        'Adjusted Futures': ('adjusted_futures_spread_value', 'adjusted_futures_spread_instruments', 'symbol')
    }

    # Iterate through each category to sort and print the top 5
    for name, (value_col, instrument_col, symbol_col) in categories.items():
        print(f"Top 5 {name} Spreads:")
        # Sort the DataFrame based on the value column in descending order
        sorted_df = df.sort_values(by=value_col, ascending=False)
        # Select the top 5 and keep only the relevant columns
        top_5 = sorted_df[[value_col, instrument_col, symbol_col]].head(5)
        print(top_5)
        print("\n")  # Add a newline for better readability between categories

def filter_dataframe_columns(df, excluded_symbols=None):
    """
    Keeps only the specified columns, drops all rows where every column is either 0 and/or NaN,
    and filters out specified symbols.
    """

    columns_to_keep = [
        "symbol", 
        "adjusted_futures_spread_value", 
        "adjusted_futures_spread_instruments", 
        "futures_spread_value", 
        "futures_spread_instruments"
    ]

    # Define the columns to check for 0 or NaN values (excluding 'symbol')
    check_columns = [
        "adjusted_futures_spread_value", 
        "adjusted_futures_spread_instruments", 
        "futures_spread_value", 
        "futures_spread_instruments"
    ]

    df = df.dropna(subset=check_columns, how='all')  # Drops rows where all check_columns are NaN
    df = df[(df[check_columns] != 0).any(axis=1)]  # Keep rows where not all check_columns are 0
    
    # Filter out specified symbols
    if excluded_symbols is None:
        excluded_symbols = ['GPTUSDT']  # Default to filtering out 'GPTUSDT'
    df = df[~df['symbol'].isin(excluded_symbols)]
    
    return df


def drop_na_rows_dataframe(df):
    # Step 1: Convert 'nan, nan' strings to actual NaN values in all columns except 'symbol'
    for col in df.columns:
        if col != 'symbol':
            df[col] = df[col].apply(lambda x: pd.NA if str(x).strip().lower() in ['nan', 'nan, nan'] else x)
    
    # Step 2: Drop columns where all values are NaN, excluding the 'symbol' column
    columns_to_drop = [col for col in df.columns if col != 'symbol' and df[col].isnull().all()]
    df = df.drop(columns=columns_to_drop)
    
    # Step 3: Filter out rows where all values, except for the 'symbol' column, are NaN
    useful_rows = df.loc[:, df.columns != 'symbol'].notna().any(axis=1)
    df = df[useful_rows]
    
    return df

def drop_useless_cols(df):
    # Drop 'time' column
    df = df[['symbol', 'adjusted_futures_spread_value', 'adjusted_futures_spread_instruments', 'futures_spread_value', 'futures_spread_instruments']]
    df = df.dropna(subset=['futures_spread_instruments', 'adjusted_futures_spread_instruments', 'adjusted_futures_spread_value', 'adjusted_futures_spread_instruments'], how='all')
    return df


def print_top_spreads(df):
    # Ensure the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Assuming there's a symbol column for each spread type in your DataFrame
    categories = {
        'Futures': ('futures_spread_value', 'futures_spread_instruments', 'symbol'),
        'Adjusted Futures': ('adjusted_futures_spread_value', 'adjusted_futures_spread_instruments', 'symbol')
    }

    # Iterate through each category to sort and print the top 5
    for name, (value_col, instrument_col, symbol_col) in categories.items():
        print(f"Top 5 {name} Spreads:")
        # Sort the DataFrame based on the value column in descending order
        sorted_df = df.sort_values(by=value_col, ascending=False)
        # Select the top 5 and keep only the relevant columns
        top_5 = sorted_df[[value_col, instrument_col, symbol_col]].head(5)
        print(top_5)
        print("\n")  # Add a newline for better readability between categories




if __name__ == "__main__":
    spread_manager = SpreadManager()
    while True:
        merged_data = merge_exchange_dataframes()
        # merged_data = merge_exchange_dataframes(webhook_type='mainnet') # uncomment for mainnet 
        merged_data = calculate_adjusted_prices(merged_data)
        merged_data = remove_multiple_of_10(merged_data)
        merged_data = calculate_spread(merged_data)
        merged_data = drop_na_rows_dataframe(merged_data)
        merged_data = drop_useless_cols(merged_data)
        spread_manager.add_dataframe(merged_data)
        # print(merged_data) #uncomment to print the dataframe







