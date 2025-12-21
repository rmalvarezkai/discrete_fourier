"""
Discrete Fourier - Example script
=================================

Author: Ricardo Marcelo Alvarez
Date: 2025-12-19
"""

import sys
import time
import pprint # pylint: disable=unused-import
import numpy
import pandas
import talib
from ehdtd import EhdtdRO, Ehdtd # pylint: disable=unused-import

import discrete_fourier.df_common_functions as dcf # pylint: disable=unused-import
from discrete_fourier import DiscreteFourier # pylint: disable=unused-import


def main(argv): # pylint: disable=unused-argument
    """
    main function
    =============
    """
    # pylint: disable=no-member
    result = 0

    __display_rows = 50
    pandas.set_option('display.min_rows', __display_rows)
    pandas.set_option('display.max_rows', __display_rows)

    __db_data = {
        'db_type': 'postgresql',  # postgresql, mysql
        'db_name': 'ehdtd',
        'db_user': 'ehdtd',
        'db_pass': 'MPBGghtEgQJi',
        'db_host': '127.0.0.1',
        'db_port': '5432'
    }

    __exchanges = ['binance', 'bybit', 'okx', 'bingx', 'kucoin']
    __exchange = __exchanges[0]
    __symbol = 'BTC/FDUSD'
    __interval = '1h'
    __start_from = 0
    __until_to = None
    __return_type = 'pandas' # pandas', 'list', 'list_consistent_streams'
                             # or 'list_consistent_streams_pandas'

    __delta_seconds = Ehdtd.get_delta_seconds_for_interval(interval=__interval)

    __n_values_p = 12400
    __n_values = 10800
    __n_futures = 12

    __low_coef = 0.15
    __open_price_coef = 0.20
    __close_price_coef = 0.50
    __high_coef = 0.15
    __ema_timeperiod_s = 5
    __ema_timeperiod_d = 3

    __columns_to_del = [
        'close_time',
        'open_price',
        'low',
        'high',
        'volume',
        'status'
    ]

    __start_from = int(time.time()) - (__n_values_p * __delta_seconds)

    __ehdtd_inst = EhdtdRO(__exchange,
                           __db_data)

    __data = __ehdtd_inst.get_data_from_db(__symbol,
                                           __interval,
                                           __start_from,
                                           __until_to,
                                           __return_type)

    __data['X__estimated_price'] = (
        __low_coef * talib.EMA(__data['low'], timeperiod=__ema_timeperiod_s) +
        __open_price_coef * talib.EMA(__data['open_price'], timeperiod=__ema_timeperiod_s) +
        __close_price_coef * talib.EMA(__data['close_price'], timeperiod=__ema_timeperiod_s) +
        __high_coef * talib.EMA(__data['high'], timeperiod=__ema_timeperiod_s)
    )

    __data['X__estimated_price'] = __data['X__estimated_price'].round(2)
    # __data = __data.filter(regex=r"^(X__|Y__|I__)", axis=1).dropna().reset_index(drop=True)
    __data = __data.dropna().reset_index(drop=True)
    __data = __data.tail(__n_values).reset_index(drop=True)

    __data_min = __data['X__estimated_price'].min()
    __data_max = __data['X__estimated_price'].max()
    __data_zero = round(((__data_min + __data_max) / 2), 2)
    __data['X__estimated_price_b'] = (__data['X__estimated_price'] - __data_zero).round(2)

    __data_min_b = __data['X__estimated_price_b'].min()
    __data_max_b = __data['X__estimated_price_b'].max()
    __data_max_var_value = max(abs(__data_min_b), abs(__data_max_b))
    __data['X__estimated_price_b'] = (
        (__data['X__estimated_price_b'] / __data_max_var_value).round(8)
    )

    # Calcular polyfit con los últimos valores para extender suavemente
    __polyfit_window = 256  # Número de valores para calcular tendencia
    __polyfit_degree = 5    # Grado del polinomio (2 = cuadrático)

    __last_values = __data['X__estimated_price_b'].tail(__polyfit_window).values
    __trend_coefs = numpy.polyfit(range(__polyfit_window), __last_values, __polyfit_degree)

    # Predecir valores futuros con polyfit
    __future_trend = numpy.polyval(
        __trend_coefs,
        range(__polyfit_window, __polyfit_window + __n_futures)
    )

    # Calcular Fourier CON datos originales (sin extensión) para comparar
    __original_len = len(__data)
    __fourier_coefs_original = DiscreteFourier.calculate_fourier_coefs(
        __data['X__estimated_price_b'].tolist()
    )

    # Extender dataframe
    __data = __data.reindex(range(len(__data) + __n_futures))

    # Guardar extensión de polyfit en columna separada para verificación
    __data['X__polyfit_extension'] = __data['X__estimated_price_b'].copy()
    __data.loc[__data.index[-__n_futures:], 'X__polyfit_extension'] = __future_trend

    # print('=' * 80)
    # print(__data)
    # print('=' * 80)

    # Calcular Fourier CON extensión de polyfit
    __fourier_coefs = DiscreteFourier.calculate_fourier_coefs(
        __data['X__polyfit_extension'].tolist()
    )

    __fourier_calculated_values = [
        DiscreteFourier.calculate_fourier_value(__fourier_coefs, n + 1)
        for n in range(len(__data))
    ]

    # Calcular Fourier SIN extensión (solo datos originales) en posiciones futuras
    __fourier_original_values = [
        DiscreteFourier.calculate_fourier_value(__fourier_coefs_original, n + 1)
        for n in range(__original_len, __original_len + __n_futures)
    ]

    __data['X__fourier_value'] = __fourier_calculated_values[:len(__data)]
    __data['X__fourier_no_extension'] = numpy.nan
    __data.loc[__data.index[-__n_futures:], 'X__fourier_no_extension'] = __fourier_original_values
    __data['X__fourier_value'] = __data['X__fourier_value'].round(8)
    __data['X__fourier_value_abs'] = (
        (__data['X__fourier_value'] * __data_max_var_value) + __data_zero
    ).round(2)

    __data.drop(columns=__columns_to_del, inplace=True)

    # Mostrar las últimas 20 filas con columnas clave
    print(__data[['X__estimated_price_b', 'X__polyfit_extension',
                   'X__fourier_value', 'X__fourier_no_extension', 
                   'X__fourier_value_abs']].tail(20))

    print('=' * 80)
    print(f'LEN: {len(__data)}')
    print(f'DATA MIN: {__data_min}')
    print(f'DATA MAX: {__data_max}')
    print(f'DATA ZERO: {__data_zero}')
    print(f'DATA MAX VAR VALUE: {__data_max_var_value}')
    print('=' * 80)

    dominant_period = DiscreteFourier.find_dominant_period(
        fourier_coefs=__fourier_coefs
    )
    print('Dominant Period:')
    pprint.pprint(dominant_period)
    print('=' * 80)

    top_periods = DiscreteFourier.find_top_periods(
        fourier_coefs=__fourier_coefs,
        n_periods=5
    )
    print('Top Periods:')
    pprint.pprint(top_periods)
    print('=' * 80)

    return result

if __name__ == "__main__":
    main(sys.argv[1:])
