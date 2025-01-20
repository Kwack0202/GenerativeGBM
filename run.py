from common_imports import *
from data.data_download import download_stock_data
from data.data_loader import StockDataLoader
from models.GAN import GAN
from scripts.train_gan import train
from scripts.test_gan import test

fix_seed = 42
random.seed(fix_seed)
np.random.seed(fix_seed)

## ==================================================
parser = argparse.ArgumentParser(description="Stock generative GBM")

## ==================================================
## basic config
## ==================================================
parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    default="pretrain",
    help="task name [options : data_download]"
)

## ==================================================
## data download
## ==================================================
parser.add_argument(
    '--output_dir',
    type=str,
    default='./stock_data/Nasdaq30/',
    help='origin data directory'
    )
parser.add_argument(
    '--tickers',
    type=str,
    nargs='+',
    default=[
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
        "WMT", "UNH", "V", "XOM", "MA", 
        "PG", "COST", "JNJ", "ORCL", "HD", 
        "BAC", "KO", "NFLX", "MRK",  "CVX", 
        "CRM", "ADBE", "AMD", "PEP", "TMO"
    ],
    help='List of stock tickers'
    )
parser.add_argument(
    '--start_day',
    type=str,
    default='2010-06-01',
    help='Start date (format : YYYY-MM-DD)'
    )
parser.add_argument(
    '--end_day',
    type=str,
    default='2025-01-10',
    help='End date (format : YYYY-MM-DD)'
    )

## ==================================================
'''
Parsing the arguments here
'''
args = parser.parse_args()

if args.task_name == "data_download":
    print("Start downloading the original data")
    
    download_stock_data(
        args.output_dir, 
        args.tickers, 
        args.start_day, 
        args.end_day
        )