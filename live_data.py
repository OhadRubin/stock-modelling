import os

import numpy as np
import pandas as pd
import json
import exchange_calendars as tc
import config
from DataProcessor import DataProcessor
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from datetime import date
from datetime import timedelta
import yfinance as yf


def get_portfolio_state(price_array, tech_array, turbulence_array, split_array, model_name,cwd):
    file_exists = os.path.exists('mydata.json')
    if not file_exists:
        status = {}

        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "split_val_array": split_array,
            "if_train": False,
        }
        env_instance = StockTradingEnv(config_params=env_config)
        config.current_model_name = model_name
        cwd = cwd
        episode_total_assets, step_status = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd, status=status, status_bool=True
        )

    else:
        f = open('mydata.json')
        status = json.load(f)

        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "split_val_array": split_array,
            "if_train": False,
        }
        env_instance = StockTradingEnv(config_params=env_config)
        config.current_model_name = model_name
        cwd = cwd
        episode_total_assets, step_status = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd, status=status, live=True, status_bool=True
        )

    if 'stocks' not in status:
        status['stocks'] = None
    status['stocks'] = [int(k) for k in env_instance.stocks]
    if 'stocks_cool_down' not in status:
        status['stocks_cool_down'] = None
    status['stocks_cool_down'] = [int(p) for p in env_instance.stocks_cool_down]

    if 'amount' not in status:
        status['amount'] = None
    status['amount'] = env_instance.amount


    current_date = date.today()
    current_date = current_date.strftime("%Y-%m-%d")
    if current_date not in status:
        status[current_date] = []
    status[current_date].append(step_status)
    # status.update(step_status)

    print(status)
    with open('mydata.json', 'w') as f:
        json.dump(status, f)



config.model_alive = True
bse_calendar = tc.get_calendar("XBOM")
today = date.today()

# dd/mm/YY
end = today.strftime("%Y-%m-%d")
if bse_calendar.is_session(end):
    last_working_date = bse_calendar.previous_session(end)
    start = last_working_date.strftime("%Y-%m-%d")
    print(end)
    print(start)

    DP = DataProcessor(data_source='yahoofinance')
    data = DP.download_data(start_date=start,
                            end_date=end,
                            ticker_list=config.TICKERS_LIST,
                            time_interval='1d')

    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, config.TECHNICAL_INDICATORS_LIST)
    data = DP.add_vix(data)
    data = DP.add_stock_split(data)
    price_array, tech_array, turbulence_array, split_array, txn_dates = DP.df_to_array(data, True)
    model_name = 'sac'
    cwd="./Nifty_50_stock_model_sac_full_data.zip"
    get_portfolio_state(price_array, tech_array, turbulence_array, split_array, model_name, cwd )

    # curr_state = self.get_state(price_array, tech_array, turbulence_array)
    # print(curr_state)

else:
    print("today is not a working day for the exchange")
