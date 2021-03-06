import yfinance as yf

instrument_to_name_map = [("^DJI", "DJI")]

interval_period = ("1d", "10y")


def file_name(inst):
    return "data/" + inst + "_" + interval_period[0] + "_" + interval_period[1] + ".csv"


def download():
    for inst in instrument_to_name_map:
        data = yf.download(
            tickers=inst[0],
            period=interval_period[1],
            interval=interval_period[0],
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None
        )
        file = file_name(inst[1])
        data.to_csv(file)


download()
