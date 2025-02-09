from common_imports import *

def get_sliding_window_data(
    df, 
    date_col: str, 
    test_start_year: int, 
    total_test_months: int, 
    sliding_test_months: int, 
    train_months: int
    ):
    """
    df: Raw DataFrame (origin csv file)
    date_col: (ex: 'index' or 'Date' or ...)
    test_start_year: Start year of the first test period (ex: 2022)
    total_test_months: full test period (ex: 3 year -> 36)
    sliding_test_months: test period per slide (ex: 1 year -> 1.0 / harf year -> 0.5)
    train_months: Period to use as train data per slide (ex: 3 year -> 3.0)
    
    return : A list with a (train_df, test_df, window_info) tuple for each sliding window as an element
    
    *window_info is a dictionary, containing information such as 
    the start/end date and number of rows in each window of train/test.
    """
    # index(date column)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    # 1st test start date: 01.01
    initial_test_start_date = pd.Timestamp(f"{test_start_year}-01-01")
    
    # train date ~ test date
    overall_start_date = initial_test_start_date - pd.DateOffset(months=train_months)
    overall_end_date = initial_test_start_date + pd.DateOffset(months=total_test_months) - pd.Timedelta(days=1)
    
    # data filtering
    df_period = df[(df[date_col] >= overall_start_date) & (df[date_col] <= overall_end_date)].copy()
    
    # counting the number of window
    n_windows = total_test_months // sliding_test_months
    
    windows = []
    
    for i in range(n_windows):
        # test date of current window 
        current_test_start_date = initial_test_start_date + pd.DateOffset(months=i * sliding_test_months)
        # test end date
        current_test_end_date = current_test_start_date + pd.DateOffset(months=sliding_test_months) - pd.Timedelta(days=1)
        
        # train date of current window 
        current_train_start_date = current_test_start_date - pd.DateOffset(months=train_months)
        # train end date
        current_train_end_date = current_test_start_date - pd.Timedelta(days=1)
        
        train_data = df_period[(df_period[date_col] >= current_train_start_date) & 
                               (df_period[date_col] <= current_train_end_date)].copy()
        test_data = df_period[(df_period[date_col] >= current_test_start_date) & 
                              (df_period[date_col] <= current_test_end_date)].copy()
        
        # window infr (모델 실행 시 출력하기 위함)
        window_info = {
            "train_start": current_train_start_date.date(),
            "train_end": current_train_end_date.date(),
            "test_start": current_test_start_date.date(),
            "test_end": current_test_end_date.date(),
            "train_rows": len(train_data), 
            "test_rows": len(test_data)
        }
        
        windows.append((train_data, test_data, window_info))
        
    return windows
