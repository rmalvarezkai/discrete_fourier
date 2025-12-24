"""
Discrete Fourier - Example script
=================================

Author: Ricardo Marcelo Alvarez
Date: 2025-12-19
"""

import sys
import os
import datetime
import pwd
import grp
import time
import pprint # pylint: disable=unused-import
import numpy
import pandas
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
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

    # Load database credentials from .database.env
    load_dotenv('.database.env')

    __db_data = {
        'db_type': 'postgresql',  # postgresql, mysql
        'db_name': os.getenv('DB_NAME', ''),
        'db_user': os.getenv('DB_USER', ''),
        'db_pass': os.getenv('DB_PASS', ''),
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
    __filter_top_n = 4

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
        DiscreteFourier.calculate_fourier_value(__fourier_coefs, n+1)
        for n in range(len(__data))
    ]

    __fourier_calculated_filter_values = [
        DiscreteFourier.calculate_fourier_value(__fourier_coefs, n+1, __filter_top_n)
        for n in range(len(__data))
    ]

    # Calcular Fourier SIN extensión (solo datos originales) en posiciones futuras
    __fourier_original_values = [
        DiscreteFourier.calculate_fourier_value(__fourier_coefs_original, n+1)
        for n in range(__original_len, __original_len + __n_futures)
    ]

    __fourier_derivative_values = [
        DiscreteFourier.calculate_fourier_derivative_value(__fourier_coefs, n + 1)
        for n in range(len(__data))
    ]

    __fourier_derivative_filter_values = [
        DiscreteFourier.calculate_fourier_derivative_value(__fourier_coefs, n + 1, __filter_top_n)
        for n in range(len(__data))
    ]

    __fourier_double_derivative_values = [
        DiscreteFourier.calculate_fourier_double_derivative_value(__fourier_coefs, n + 1)
        for n in range(len(__data))
    ]

    __fourier_double_derivative_filter_values = [
        DiscreteFourier.calculate_fourier_double_derivative_value(__fourier_coefs,
                                                                   n + 1,
                                                                   __filter_top_n)
        for n in range(len(__data))
    ]

    __n_data_d = 2  # Para eliminar valores inestables en derivadas

    __data['X__fourier_value'] = __fourier_calculated_values[:len(__data)]
    __data['X__fourier_value_filtered'] = __fourier_calculated_filter_values[:len(__data)]

    __data['X__fourier_derivative_value'] = __fourier_derivative_values[:len(__data)]
    __data['X__fourier_derivative_value'] = 0
    __data['X__fourier_derivative_value_filtered'] = (
        __fourier_derivative_filter_values[:len(__data)]
    )
    __data['X__fourier_double_derivative_value'] = (
        __fourier_double_derivative_values[:len(__data)]
    )
    __data['X__fourier_double_derivative_value'] = 0

    __data['X__fourier_double_derivative_value_filtered'] = (
        __fourier_double_derivative_filter_values[:len(__data)]
    )

    __data.loc[__data.index[-__n_data_d:], 'X__fourier_derivative_value'] = numpy.nan
    __data.loc[__data.index[-__n_data_d:], 'X__fourier_derivative_value_filtered'] = numpy.nan
    __data.loc[__data.index[-__n_data_d:], 'X__fourier_double_derivative_value'] = numpy.nan
    __data.loc[__data.index[-__n_data_d:], 'X__fourier_double_derivative_value_filtered'] = (
        numpy.nan
    )

    __data.loc[__data.index[0:__n_data_d], 'X__fourier_derivative_value'] = numpy.nan
    __data.loc[__data.index[0:__n_data_d], 'X__fourier_derivative_value_filtered'] = numpy.nan
    __data.loc[__data.index[0:__n_data_d], 'X__fourier_double_derivative_value'] = numpy.nan
    __data.loc[__data.index[0:__n_data_d], 'X__fourier_double_derivative_value_filtered'] = (
        numpy.nan
    )

    __data['X__fourier_no_extension'] = numpy.nan
    __data.loc[__data.index[-__n_futures:], 'X__fourier_no_extension'] = __fourier_original_values
    __data['X__fourier_value'] = __data['X__fourier_value'].round(8)
    __data['X__fourier_value_abs'] = (
        (__data['X__fourier_value'] * __data_max_var_value) + __data_zero
    ).round(2)

    __data.drop(columns=__columns_to_del, inplace=True)

    # Mostrar las últimas 20 filas con columnas clave
    print(__data[['X__estimated_price_b', 'X__polyfit_extension',
                  'X__fourier_value', 'X__fourier_value_filtered',
                  'X__fourier_no_extension', 'X__fourier_value_abs']].tail(20))

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
    pprint.pprint(dominant_period, sort_dicts=False)
    print('=' * 80)

    # Validar el período dominante
    validation = DiscreteFourier.validate_period(
        data_in=__data['X__estimated_price_b'].dropna().tolist(),
        period=dominant_period['period']
    )
    print('Period Validation:')
    pprint.pprint(validation, sort_dicts=False)
    print('=' * 80)

    top_periods = DiscreteFourier.find_top_periods(
        fourier_coefs=__fourier_coefs,
        n_periods=__filter_top_n
    )
    print('Top Periods:')
    pprint.pprint(top_periods, sort_dicts=False)
    print('=' * 80)

    # return result
    # Validar los top períodos
    __windows_size = 256
    __data_in = __data['X__estimated_price_b'].dropna().tolist()
    __period_init = 1024
    __period_end = len(__data_in) - __windows_size
    __correlation_limit = 0.85
    __confidence_limit = 0.85

    print('Find cycles:')
    for p in range(__period_init, __period_end + 1):
        if len(__data_in) >= (__windows_size + p):
            val = DiscreteFourier.validate_period(
                data_in=__data_in,
                period=p,
                window_size=__windows_size
            )

            if val is not None and isinstance(val, dict) and 'valid' in val and val['valid']:
                if 'correlation' in val\
                    and val['correlation'] is not None\
                    and isinstance(val['correlation'], (float, int))\
                    and val['correlation'] >= __correlation_limit:
                    if 'confidence' in val\
                        and val['confidence'] is not None\
                        and isinstance(val['confidence'], (float, int))\
                        and val['confidence'] >= __confidence_limit:

                        print(f"Period {p}: "
                              f"valid={val['valid']}, "
                              f"confidence={val['confidence']}, "
                              f"correlation={val['correlation']}")
    print('=' * 80)

    # Create plotly visualization

    # Create .html directory if it doesn't exist
    os.makedirs('.html', exist_ok=True)

    # Import subplots for creating multiple charts

    # Create figure with 3 subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Fourier Analysis - Value',
            'Fourier Analysis - First Derivative (Slope)',
            'Fourier Analysis - Second Derivative (Curvature)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.34, 0.33, 0.33],
        shared_yaxes=False
    )

    # First subplot: Fourier values
    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__estimated_price_b'],
        mode='lines',
        name='Estimated Price (Normalized)',
        line=dict(color='blue', width=2),
        legendgroup='group1',
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__fourier_value'],
        mode='lines',
        name='Fourier Value',
        line=dict(color='red', width=2, dash='dash'),
        legendgroup='group1',
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__fourier_value_filtered'],
        mode='lines',
        name=f'Fourier Value Filtered (top {__filter_top_n})',
        line=dict(color='green', width=2, dash='dot'),
        legendgroup='group1',
        showlegend=True
    ), row=1, col=1)

    # Second subplot: First derivative
    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__fourier_derivative_value'],
        mode='lines',
        name='First Derivative',
        line=dict(color='orange', width=2),
        legendgroup='group2',
        showlegend=True
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__fourier_derivative_value_filtered'],
        mode='lines',
        name=f'First Derivative Filtered (top {__filter_top_n})',
        line=dict(color='cyan', width=2, dash='dash'),
        legendgroup='group2',
        showlegend=True
    ), row=2, col=1)

    # Third subplot: Second derivative
    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__fourier_double_derivative_value'],
        mode='lines',
        name='Second Derivative',
        line=dict(color='purple', width=2),
        legendgroup='group3',
        showlegend=True
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=__data.index,
        y=__data['X__fourier_double_derivative_value_filtered'],
        mode='lines',
        name=f'Second Derivative Filtered (top {__filter_top_n})',
        line=dict(color='yellow', width=2, dash='dash'),
        legendgroup='group3',
        showlegend=True
    ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title_text=f'Fourier Analysis - {__symbol} ({__interval})',
        hovermode='x unified',
        template='plotly_dark',
        height=1400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Update x-axes
    fig.update_xaxes(title_text="Index", row=1, col=1)
    fig.update_xaxes(title_text="Index", row=2, col=1)
    fig.update_xaxes(title_text="Index", row=3, col=1)

    # Update y-axes with independent scales (matches=None ensures independence)
    fig.update_yaxes(title_text="Value", row=1, col=1, matches=None)
    fig.update_yaxes(title_text="Derivative", row=2, col=1, matches=None)
    fig.update_yaxes(title_text="2nd Derivative", row=3, col=1, matches=None)

    # Save to HTML file with dark theme
    html_filename = f'.html/fourier_analysis_{__symbol.replace("/", "_")}_{__interval}.html'

    fig.write_html(
        html_filename,
        config={'displayModeBar': True, 'displaylogo': False},
        include_plotlyjs='cdn',
        full_html=True,
        default_width='100%',
        default_height='100%'
    )

    # Read the generated HTML and wrap it with dark theme
    with open(html_filename, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Insert dark background style into the HTML
    html_content = html_content.replace(
        '<head>',
        '<head><style>body { background-color: #1e1e1e; margin: 0; padding: 20px; }</style>'
    )

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Set file permissions
    os.chown(html_filename, -1, os.getgid())  # Keep user, set group
    os.chmod(html_filename, 0o664)

    # Try to set www-data as owner and repository as group
    try:
        www_data_uid = pwd.getpwnam('www-data').pw_uid
        repository_gid = grp.getgrnam('repository').gr_gid
        os.chown(html_filename, www_data_uid, repository_gid)
    except (KeyError, PermissionError) as e:
        print(f'Warning: Could not set user/group ownership: {e}')

    print(f'Plot saved to: {html_filename}')

    # Create or update index.html
    index_filename = '.html/index.html'

    # Get all HTML files in the directory (excluding index.html)
    html_files = [f for f in os.listdir('.html') if f.endswith('.html') and f != 'index.html']
    html_files.sort()

    # Generate index.html content
    index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fourier Analysis - Index</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        h1 {
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .file-list {
            background-color: #2d2d2d;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        }
        .file-item {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #4CAF50;
            background-color: #3a3a3a;
        }
        .file-item a {
            color: #64B5F6;
            text-decoration: none;
            font-size: 16px;
        }
        .file-item a:hover {
            text-decoration: underline;
            color: #90CAF9;
        }
        .timestamp {
            color: #999;
            font-size: 12px;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <h1>Fourier Analysis Reports</h1>
    <div class="file-list">
"""

    if html_files:
        for html_file in html_files:
            file_path = os.path.join('.html', html_file)
            file_time = os.path.getmtime(file_path)
            timestamp = datetime.datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
            index_content += f"""        <div class="file-item">
            <a href="{html_file}">{html_file}</a>
            <span class="timestamp">Last modified: {timestamp}</span>
        </div>
"""
    else:
        index_content += """        <p>No analysis reports available yet.</p>
"""

    index_content += """    </div>
</body>
</html>
"""

    # Write index.html
    with open(index_filename, 'w', encoding='utf-8') as f:
        f.write(index_content)

    # Set permissions for index.html
    try:
        os.chmod(index_filename, 0o664)
        www_data_uid = pwd.getpwnam('www-data').pw_uid
        repository_gid = grp.getgrnam('repository').gr_gid
        os.chown(index_filename, www_data_uid, repository_gid)
    except (KeyError, PermissionError) as e:
        print(f'Warning: Could not set user/group ownership for index.html: {e}')

    print(f'Index updated: {index_filename}')
    print('=' * 80)
    print('http://192.168.252.23/discrete_fourier')
    print('=' * 80)
    return result

if __name__ == "__main__":
    main(sys.argv[1:])
