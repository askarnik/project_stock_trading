def extract_technicals(df, tech_funcs_dict, tech_funcs_params):
    '''
    Serially applies a dictionary of technical functions to a dataframe
    '''
    output = df
    for name, func in tech_funcs_dict.items():
        arg_list = tech_funcs_params[name]
        for arg_tuple in arg_list:
            output = func(output, *arg_tuple)

    return output

def update_technicals(df, close_price, tech_funcs_dict, tech_funcs_params):
    '''
    Takes technicals df adds a new close for the next day, then updates the next day's
    technicals based on that close price
    '''
    # exclude prior TI columns
    seed_columns = ['Open', 'High', 'Low', 'Close', 'Volume',]
    seed_columns_df = df[seed_columns]

    last_row = seed_columns_df.iloc[-1, :].copy()
    last_row["Close"] = close_price
    seed_columns_df = seed_columns_df.append(last_row, ignore_index = True)

    return extract_technicals(seed_columns_df, tech_funcs_dict, tech_funcs_params)
