from common_imports import *

def download_stock_data(output_dir, stock_tickers, start_day, end_day,):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for stock_code in tqdm(stock_tickers, desc="Downloading stock data"):
        
        try:
            stock_data = pd.DataFrame(fdr.DataReader(stock_code, start_day, end_day))
            stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
            stock_data = stock_data.reset_index()
            stock_data.to_csv(os.path.join(output_dir, f"{stock_code}.csv"), encoding='utf-8', index=False)
            
        except Exception as e:
            print(f"Failed to download data for {stock_code}: {e}")