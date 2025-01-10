#!/bin/bash
python run.py \
    --task_name data_download \
    --output_dir ./datasets/Nasdaq/ \
    --tickers AAPL NVDA MSFT GOOG AMZN \
    --start_day 2010-06-01 \
    --end_day 2025-01-01 \