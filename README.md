## GenerativeGBM
This project combines generative models and GBM to generate high-quality virtual stock price data and develop stock price prediction models.

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

### 2. Train 📑
**shell file**
```
sh ./scripts/train.sh
```
![Framework](./assets/result_sample(MSFT).png)

<div style="display: flex; gap: 20px;">
  <img src="./assets/result_sample(MSFT).png" style="width: 50%;">
  <img src="./assets/result_sample_noise_distribution(MSFT).png" style="width: 50%;">
</div>
