from datetime import datetime
import time
import pandas as pd


def get_weekday(start_day, current_day):
    start_day = datetime.strptime(start_day, "%d.%m.%Y %H:%M:%S")
    end_day = datetime.strptime(current_day, "%d.%m.%Y %H:%M:%S")

    return int((end_day - start_day).days % 7) + 1


def get_weekday_train(row):
    date_time = row['Date Time']
    return get_weekday("01.01.2009 00:10:00", date_time)


def get_weekday_test(row):
    date_time = row['Date Time']
    return get_weekday("01.01.2015 00:10:00", date_time)


def get_week_number(start_day, current_day):
    start_day = datetime.strptime(start_day, "%d.%m.%Y %H:%M:%S")
    end_day = datetime.strptime(current_day, "%d.%m.%Y %H:%M:%S")

    return int((end_day - start_day).days / 7)


def get_week_number_train(row):
    date_time = row['Date Time']
    return get_week_number("01.01.2009 00:10:00", date_time)


def get_week_number_test(row):
    date_time = row['Date Time']
    return get_week_number("01.01.2015 00:10:00", date_time)


def get_year(row):
    date_time = row['Date Time']
    year = time.strptime(date_time, "%d.%m.%Y %H:%M:%S").tm_year

    return year


def weather_split(data_path, output_paths):
    data = pd.read_csv(data_path)

    data['Year'] = data.apply(get_year, axis=1)

    train_data_row_index = data[(data['Year'] <= 2014)].index.tolist()
    test_data_row_index = data[(data['Year'] >= 2015)].index.tolist()

    train_data_row_index.append(train_data_row_index[-1] + 1)
    test_data_row_index = test_data_row_index[1:]

    train_data = data.iloc[train_data_row_index]
    test_data = data.iloc[test_data_row_index]

    # 第几周
    train_data['Week Number'] = train_data.apply(get_week_number_train, axis=1)
    test_data['Week Number'] = test_data.apply(get_week_number_test, axis=1)

    # 星期几
    train_data['WeekDay'] = train_data.apply(get_weekday_train, axis=1)
    test_data['WeekDay'] = test_data.apply(get_weekday_test, axis=1)

    train_data.to_csv(output_paths[0], index=False)
    test_data.to_csv(output_paths[1], index=False)
