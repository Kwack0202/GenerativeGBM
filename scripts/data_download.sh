#!/bin/bash
python run.py \
    --task_name data_download \
    --output_dir ./datasets/Nasdaq/ \
    --tickers AAPL MSFT NVDA GOOG AMZN BRK-B LLY AVGO TSLA JPM WMT UNH V XOM MA PG COST JNJ ORCL HD BAC KO NFLX MRK CVX CRM ADBE AMD PEP TMO \
    --start_day 2019-01-01 \
    --end_day 2025-01-01 \