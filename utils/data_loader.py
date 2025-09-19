import pandas as pd


def rolling_time(file_path, train_period, test_period):
    df = pd.read_csv(file_path)
    df_year = pd.to_datetime(df['日期']).dt.year
    total_years = df_year.nunique()
    print (f"Unique years in data: {df_year.unique()}")
    print (f"Total unique years in data: {total_years}")
    return total_years - train_period - test_period + 1

def load_data(file_path, train_period, test_period, rolling_index):

    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by='日期')
    
    start_year = df['日期'].dt.year.min() + rolling_index
    train_start = start_year
    train_end = start_year + train_period - 1
    test_start = train_end + 1
    test_end = test_start + test_period - 1

    train_data = df[(df['日期'].dt.year >= train_start) & (df['日期'].dt.year <= train_end)]
    test_data = df[(df['日期'].dt.year >= test_start) & (df['日期'].dt.year <= test_end)]

    if test_data.empty:
        test_start += 1
        test_end += 1
        test_data = df[(df['日期'].dt.year >= test_start) & (df['日期'].dt.year <= test_end)]
    
    print(f"Rolling Index: {rolling_index}")
    print(f"Train Period: {train_start} to {train_end}, Test Period: {test_start} to {test_end}")
    print(f"Train Data Shape: {train_data.shape}, Test Data Shape: {test_data.shape}")
    
    return train_data, test_data

if __name__ == "__main__":
    years = rolling_time('data/esg_tfidf_with_return_cleaned.csv', 5, 1)
    print(f"Rolling time periods available: {years}")
    for i in range(years):
        train_data, test_data = load_data('data/esg_tfidf_with_return_cleaned.csv', 5, 1, i)