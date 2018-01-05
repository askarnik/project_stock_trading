import numpy as np
from sklearn.linear_model import Lasso
from functools import partial

def connect_line(arr, span_tuple):
    '''
    Return function for a line that connects two values from arr.
    Inputs:
    - arr: numpy 1d-array of time-series values
    - span_tuple: start and enpdpoints for line
    '''

    # if not all(map(lambda x: isinstance(x, int), span_tuple)):
    #     raise ValueError('span_tuple must contain all ints')

    first, last = span_tuple
    # if not (last > first > 0):
    #     raise ValueError('last should be greater than first should be greater than 0')

    run = last - first
    rise = arr[last] - arr[first]
    slope = rise / run
    intercept = arr[first] - first * slope

    return lambda x: intercept + slope * x


def line_point_distance(arr, x1, x2, x0):
    '''
    Calculate the minimal orthogonal distance from a line connecting
    (x1, arr[x1]) and (x2, arr[x2]) to a third point (x0, arr[x0])
    Distance formula from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line.
    Inputs:
    - arr: numpy 1d-array of time-series values
    - x1: int, start x-value of line
    - x2: int, end x-value of line
    - x3: int, third point for distance to be calculated
    '''
    y1 = arr[x1]
    y2 = arr[x2]
    y0 = arr[x0]

    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return numerator / denominator


def farthest_point(arr, x1, x2):
    '''
    Find the farthest point between two points that define a line on a time-series.
    '''
    xx = np.arange(len(arr))

    distances_partial = partial(line_point_distance, arr, x1, x2)
    distances = distances_partial(xx)
    mask = (xx > x1) & (xx < x2)
    farthest_point_indx = np.argmax(distances[mask]) + x1
    max_distance = distances[farthest_point_indx]

    return farthest_point_indx, max_distance


def flatten(nested):
    '''
    Utility to flatten nested lists.
    '''
    flattened = []
    for item in nested:
        if type(item) != list:
            flattened.append(item)
        else:
            flattened += flatten(item)
    return flattened


def plr_recursive(arr, span_tuple, epsilon):
    '''
    Compute piece-wise linear regression on a time series.
    Inputs:
    - arr: numpy 1d-array of time-series values
    - span_tuple: start and endpoints
    - epsilon: threshold for line-breaking
    Outputs:
    - lines: list of lambda functions
    - points: list of points that dicate when lambdas should be used
    '''
    first, last = span_tuple
    farthest_indx, distance = farthest_point(arr, first, last)

    lines = []
    points = []

    if distance < epsilon: #base case
        line = connect_line(arr, (int(first), int(last)))
        point = first

        return line, point

    else: #recursive case

        line_1, point_1 = plr_recursive(arr, (first, farthest_indx), epsilon)
        line_2, point_2 = plr_recursive(arr, (farthest_indx, last), epsilon)

        lines.extend([line_1, line_2])
        points.extend([point_1, point_2])

        #flatten out recursive nests
        return flatten(lines), flatten(points)


def PLR(arr, epsilon):
    '''
    Compute full PLR for a time series
    '''
    xx = np.arange(len(arr), dtype = float)
    lines, points = plr_recursive(arr, (0, len(arr) -1), epsilon)

    if type(lines) != list:
        plr_values = lines(xx)

    else:
        plr_values = np.piecewise(xx, [xx >= point for point in points], lines)

    return plr_values, lines, points


def up_down_trend(arr, epsilon):
    '''
    Classify point in a time series as either in an up trend or a down trend
    as determined by a PLR.
    
    Inputs
    - arr: numpy 1d-array of time-series values
    - epsilon: tolerance for PLR procedure
    
    Outputs
    - list of list of bools, where True indicates membership of an up trend
    '''
    pieces, lines, points = PLR(arr, epsilon)
    points.append(len(arr) - 1)
    diffs = np.diff(points)
    
    #I think this compensates for undercounting the first trend by one
    diffs[0] += 1 
    
    trend_labels = []
    
    for point, diff in zip(points[1:], diffs):
        up = [pieces[point] - pieces[point - 5] > 0]
        trend_labels.append(up * diff)
    
    return trend_labels


def trading_signal(arr, epsilon):
    '''
    Convert a time-series into a trading signal ranging from 0 to 1.
    For more information refer to "A dynamic threshold decision system ..."
    by Chang et al.
    
    Inputs
    - arr: numpy 1d-array of time-series values
    - epsilon: tolerance for PLR procedure
    
    Outputs
    - numpy 1d-array with the same length as arr
    '''
    labels = up_down_trend(arr, epsilon)
    
    final_signal_list = []
    for trend_list in labels:
        up = any(trend_list)
        length = len(trend_list)
        half_length, remainder = divmod(length, 2)
        
        if remainder:
            half_length += 1
            
        if up:
            first_half = 0.5 - np.arange(half_length) / length
        else:
            first_half = 0.5 + np.arange(half_length) / length
        
        if remainder:
            trend_signal = np.concatenate((first_half, first_half[-2::-1]))
        else:
            trend_signal = np.concatenate((first_half, first_half[::-1]))
        
        final_signal_list.append(trend_signal)
        
    return np.concatenate(final_signal_list)


def calc_profit(arr, epsilon):
    trend_labels = up_down_trend(arr, epsilon)
    
    arr_indexes = list(np.array([len(x) for x in trend_labels]).cumsum() - 1)
    trends = [trend_labels[x][0] for x in range(len(trend_labels))]
    
    buys = []
    profit = 0
    
    arr_indexes.insert(0, 0)
    trends.append(not trends[len(trends) - 1])
    
    for curr_index, curr_trend in zip(arr_indexes, trends):
        if curr_trend:
            buys.append(arr[curr_index])
        else:
            profit += len(buys) * arr[curr_index] - np.sum(buys)
            buys = []
            
    return profit


def best_epsilon(close, profit_flag=False):
    epsilon_values = list(np.logspace(-3, 0, 4))
    epsilon_values.extend([.5*x for x in range(3,41)])

    profits = []
    epsilons = []
    
    for val in epsilon_values:
        try:
            curr_profit = calc_profit(close, val)
            profits.append(curr_profit)
            epsilons.append(val)
        except:
            continue

    if profit_flag:
        return epsilons, profits
    else:
        return epsilons[np.argmax(profits)]