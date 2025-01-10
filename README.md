## GenerativeGBM
A project to combine generative AI and GBM to generate high-quality virtual stock price data and develop stock price prediction models.


## Usage
### 1. Download Origin Stock Data 📊 
**shell file**
```
sh ./scripts/data_download.sh
```
**shell code details (example)**
```
python run.py \
   --task_name data_download \
   --output_dir ./datasets/Nasdaq/ \
   --tickers AAPL NVDA MSFT GOOG AMZN \
   --start_day 2010-06-01 \
   --end_day 2025-01-01 \
```