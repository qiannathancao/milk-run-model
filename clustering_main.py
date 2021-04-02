import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import glob
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import defaultdict

pd.options.mode.chained_assignment = None
sys.path.insert(0, '../main_generalized_run_rev_1')
import config


class ETL_data:
    def __init__(self):
        pass

    def distance_on_sphere_numpy(self, coordinate_df):
        """
        Compute a distance matrix of the coordinates using a spherical metric.
        :param coordinate_array: numpy.ndarray with shape (n,2); latitude is in 1st col, longitude in 2nd.
        :returns distance_mat: numpy.ndarray with shape (n, n) containing distance in km between coords.
        """
        # Radius of the earth in km (GRS 80-Ellipsoid)
        EARTH_RADIUS = 6371.007176
        km2mile_ratio = 0.62137

        # Unpacking coordinates
        latitudes = coordinate_df.loc[:, 'latitude']
        longitudes = coordinate_df.loc[:, 'longitude']
        # Convert latitude and longitude to spherical coordinates in radians.
        degrees_to_radians = np.pi / 180.0
        phi_values = (90.0 - latitudes) * degrees_to_radians
        theta_values = longitudes * degrees_to_radians
        # Expand phi_values and theta_values into grids
        theta_1, theta_2 = np.meshgrid(theta_values, theta_values)
        theta_diff_mat = theta_1 - theta_2
        phi_1, phi_2 = np.meshgrid(phi_values, phi_values)
        # Compute spherical distance from spherical coordinates
        angle = (np.sin(phi_1) * np.sin(phi_2) * np.cos(theta_diff_mat) + np.cos(phi_1) * np.cos(phi_2))
        arc = np.arccos(angle)
        # Multiply by earth's radius to obtain distance in km
        return np.nan_to_num(arc * EARTH_RADIUS * km2mile_ratio)

    def col_name(self, df):
        """
        this is to trim the data_frame column names to a unique format:
        all case, replace space to underscore, remove parentheses
        param df:
            raw from share drive for
        return:
            polished data set with new column names
        """
        df.columns = df.columns.str.strip().str.lower().str.replace('-', '').str.replace(' ', '_').str.replace('(', ''). \
            str.replace(')', '').str.replace('"', '')
        return df

    # covert time format from str to timestamp
    def str2time(self, x):
        """
        paramter
            x: string formated time
        return:
            timeStamp with mm/dd/yyyy format
        """
        try:
            return datetime.strptime(x, '%b %d, %Y')
        except:
            try:
                return datetime.strptime(x, '%d-%b-%y')
            except:
                return '0000-00-00'

    def logic_and_3_condition(self, x1, x2, x3):
        return np.logical_and(np.logical_and(x1, x2), x3)

    def clean_zip(self, zipcode_input_file):
        """
        parameter:
            df: original zipcode msater file
            file_path: zipcode file directory
        """

        # change zipcodes which contain alphabix letter to 0 (outside of USA)
        def to_string(x):
            try:
                return str(x)
            except:
                return 0

        zipcode = self.col_name(zipcode_input_file)
        zipcode.loc[:, 'zip_code'] = zipcode['zip_code'].apply(lambda x: to_string(x))
        zipcode.loc[:, 'state_abbreviation'] = zipcode.state_abbreviation.str.strip()
        return zipcode

    def clean_cass(self, source_file_path, source_state=None, source_country=None, dest_zip='54942',
                   shipping_date_start='2019-01-01', shipping_window=7, truck_mode='LT',
                   inbound_indicator='INBOUND'):
        """
        parameter:
            df: dataFrame, original dataset downloaded from cass
            source_state: str, comma needed, default 'WI' as the largest shipping from
            dest_zip: str, default 54942 as greenville
            shipping_date_start: str, starting shipping schedule cut-off date
            shipping_window: int, days out as shipping period window
            truck_mode: str, no comma needed, default LT for less than truck and full truck both inclusive

        return:
            df: cleaned dataFrame
        """
        shipping_date_start = datetime.strptime(shipping_date_start, '%Y-%m-%d')

        # read in source files from share drive cognos reports
        extension = 'csv'
        files = [f for f in glob.glob(os.path.join(source_file_path, config.FY_SELECTION.format(extension)))]
        df = self.col_name(pd.concat([pd.read_csv(f, low_memory=False) for f in files]))

        # cleanse above df from CASS_cognos report
        df = df[['segment_description', 'shipment_date', 'mode', 'inbound_outbound_indicator', 'shipper_name',
                 'shipper_address', 'shipper_city', 'shipper_zip', 'shipper_state', 'shipper_country',
                 'destination_city', 'destination_zip', 'destination_state',
                 'bill_of_lading_number', 'ship_weight', 'miles', 'billed_amount']]
        df = df[df.inbound_outbound_indicator == inbound_indicator]
        df = df[df.destination_zip == dest_zip]
        df = df[df.shipper_country == source_country]
        df.loc[:, 'ship_weight'] = df.ship_weight.apply(lambda x: x.replace(',', '')).astype('int')
        df.drop(columns=['inbound_outbound_indicator'], inplace=True)
        df.drop(columns=['shipper_country'], inplace=True)

        if source_state and source_state is not 'all':
            states = source_state.split(',')
            df = df[np.logical_and(df.destination_zip == dest_zip, df.shipper_state.isin(states))]

        df.loc[:, 'shipping_date'] = df.shipment_date.apply(lambda x: self.str2time(x))

        df = df[self.logic_and_3_condition((df.shipping_date >= shipping_date_start),
                                           (df.shipping_date <= shipping_date_start + dt.timedelta(shipping_window)),
                                           (df['mode'].isin(list(truck_mode))))]

        df.loc[:, 'miles'] = df.miles.apply(lambda x: x.replace(',', '')).astype('int')
        df.loc[:, 'billed_amount'] = df.billed_amount.apply(lambda x: x.replace(',', '')).astype('float')

        # change zipcodes which contain alphabix letter to 0 (outside of USA)
        def to_string(x):
            try:
                return str(x)
            except:
                return 0

        df.loc[:, 'shipper_zip'] = df['shipper_zip'].apply(lambda x: to_string(x))

        df = df.reset_index(drop=True)
        return df

    def cass_merge_zip(self, df_cass, df_zip):
        """
        parameter:
            df1: cleaned cass dataset
            df2: original zipcode matrix
        return:
            merged dataFrame contains longitude and latitude
        """
        df = pd.merge(df_cass, df_zip[['zip_code', 'longitude', 'latitude', 'state_abbreviation']], how='left',
                      left_on='shipper_zip', right_on='zip_code')
        return df

    def name_convention(self, df, level=config.FUZZY_LEVEL):
        """
        :param df: Dataframe, a result from function clean_cass
        :param level: float, a threshold for fuzzy process
        :return df: Dataframe: renaming the df with normalized and generalized supplier name
        """
        df.loc[:, 'std_name'] = df.shipper_name.str.strip().str.lower().str.replace('-', ''). \
            str.replace(' ', '').str.replace('(', '').str.replace(')', ''). \
            str.replace('"', '').str.replace('inc', '').str.replace('.', '').str.replace(',', '')
        df.loc[:, 'shipper_zip'] = df.shipper_zip.apply(lambda x: x.lstrip('0'))

        name_dict = defaultdict(list)
        name_list = list(df.std_name.unique())
        df_extended_name = pd.DataFrame(columns=['std_name', 'unique_name'])

        while name_list:
            df_temp = pd.DataFrame(columns=['std_name', 'unique_name'])
            key = name_list[0]
            check_list = name_list[1:]
            fuzzy_res = process.extract(key, check_list, scorer=fuzz.partial_token_set_ratio)
            selected_names = [n for n, m in fuzzy_res if m >= level]
            name_dict[key] = selected_names
            selected_names.append(key)
            df_temp.loc[:, 'std_name'] = selected_names
            df_temp.loc[:, 'unique_name'] = key
            df_extended_name = pd.concat([df_extended_name, df_temp]).reset_index(drop=True)
            name_list = [x for x in name_list if x not in selected_names]
        df = df.merge(df_extended_name, how='left', on='std_name', copy=False)

        return df

    def period_freq(self, df, freq_list=config.RESAMPLE_LIST, period_level=config.PERIOD_FREQ_LEVEL):
        """
        df: dataFrame like dfcass_df
        freq_list: list like ['W', '2W', 'M']
        target_freq_level: float, threshold for slicing limit of shipment freq / max possible ship count in that group
        """
        ratio_master = pd.DataFrame()
        df_ts = df.copy()
        df_ts.index = df_ts.shipping_date

        for f in freq_list:
            # validate consecutive shipment happens, lagging one period for bool check
            df_freq = df_ts.groupby(['unique_name', 'shipper_zip'])['ship_weight'].resample(f).agg(['count']). \
                reset_index()

            df_freq.loc[:, 'count_0_1'] = df_freq['count'].where(df_freq['count'] == 0, 1)

            df_freq.loc[:, 'shift_1p'] = df_freq.groupby(['unique_name'])['count_0_1'].shift().fillna(0)

            df_freq.loc[:, 'continuous_shipment'] = np.logical_and(df_freq.count_0_1 == 1, df_freq.shift_1p == 1)

            # count occurrence from each in freq_list
            max_count = max(
                df_freq.groupby(['unique_name', 'shipper_zip'])['shipper_zip'].agg('count').reset_index(drop=True))

            # find freq shipment ratio per each in freq_list
            ratio_freq = (df_freq.groupby(['unique_name', 'shipper_zip'])['continuous_shipment'].sum() / max_count) \
                .reset_index()
            ratio_freq.loc[:, 'freq'] = f

            ratio_master = pd.concat([ratio_master, ratio_freq], ignore_index=True, axis=0)

        # create the list of supplier associate with target_freq_level
        # weekly
        s_name_wk = ratio_master[(ratio_master.freq == 'W') &
                                 (ratio_master.continuous_shipment >= period_level)]['unique_name']

        s_zip_wk = ratio_master[(ratio_master.freq == 'W') &
                                (ratio_master.continuous_shipment >= period_level)]['shipper_zip']

        weekly_suppliers = list(zip(s_name_wk, s_zip_wk))
        df_weekly = df[pd.Series(zip(df.unique_name, df.shipper_zip)).isin(weekly_suppliers)]
        df_weekly.loc[:, 'freq'] = 'weekly'

        # biweekly
        s_name_2wk = ratio_master[(ratio_master.freq == '2W') &
                                  (ratio_master.continuous_shipment >= period_level)]['unique_name']

        s_zip_2wk = ratio_master[(ratio_master.freq == '2W') &
                                 (ratio_master.continuous_shipment >= period_level)]['shipper_zip']

        bi_weekly_suppliers = [x for x in list(zip(s_name_2wk, s_zip_2wk)) if x not in weekly_suppliers]

        df_bi_weekly = df[pd.Series(zip(df.unique_name, df.shipper_zip)).isin(bi_weekly_suppliers)]
        df_bi_weekly.loc[:, 'freq'] = 'bi_weekly'

        # monthly
        monthly_suppliers = [x for x in list(zip(ratio_master.unique_name, ratio_master.shipper_zip)) if
                             x not in weekly_suppliers and x not in bi_weekly_suppliers]
        df_monthly = df[pd.Series(zip(df.unique_name, df.shipper_zip)).isin(monthly_suppliers)]
        df_monthly.loc[:, 'freq'] = 'monthly'

        return df_weekly, df_bi_weekly, df_monthly

    def agg_measurement(self, df, freq_list=config.RESAMPLE_LIST):
        """
        para:
            df: df_result from function period_freq
            freq_list: ['W', '2W', 'M']

        return:
            aggregated df with mean of measurement: weight, bill_amount, miles
        """
        df_freq = pd.DataFrame()
        measurement_list = ['ship_weight', 'miles', 'billed_amount']
        for i, f in enumerate(freq_list):
            df_measurement = pd.DataFrame()
            for m in measurement_list:
                # total shipments within each epoch of frequency of weekly-biweekly-monthly
                df[i].index = df[i]['shipping_date']
                df_temp = df[i].groupby(['shipper_zip', 'unique_name', 'freq'])[m].resample(f).agg(['sum'])
                df_measurement = pd.concat([df_measurement, df_temp], axis=1)

            df_freq = pd.concat([df_freq, df_measurement], axis=0)
        df_freq.columns = measurement_list
        df_freq[measurement_list] = df_freq[measurement_list].replace(0, np.nan)
        df_freq.reset_index(inplace=True)

        def f(x):
            d = defaultdict()
            d['ship_weight_freq_median'] = x['ship_weight'].median()

            d['ship_weight_annum'] = x['ship_weight'].sum()
            d['shipment_count_annum'] = x['miles'].count()
            d['billed_amount_annum'] = x['billed_amount'].sum()
            return pd.Series(d, index=['ship_weight_freq_median', 'ship_weight_annum', 'shipment_count_annum',
                                       'billed_amount_annum'])

        df_freq = df_freq.groupby(['shipper_zip', 'unique_name', 'freq']).apply(f).reset_index()
        df_freq.loc[:, 'shipper_zip'] = df_freq['shipper_zip'].apply(lambda x: x[1:] if x[0] == '0' else x)
        return df_freq


class ClusterModel:
    """ to complete this, complete ipynb to py """

    def __init__(self, output_path):
        self.output_path = output_path

    def kcluster(self, cass_zip_file, k_range=20):
        """
        :param cass_zip_file: cass_merge_zip generated file
        :param k_range: initialize number of max clustering upper bound
        :return:
        """

        geo_all = cass_zip_file.loc[:, ['longitude', 'latitude', 'freq']]
        unique_freq = geo_all.freq.unique()
        df_distortion_all = pd.DataFrame()
        K = range(1, k_range)
        for f in unique_freq:
            distortions = []
            for k in K:
                geo = geo_all.loc[:, ['longitude', 'latitude']]
                kmeans = KMeans(n_clusters=k, random_state=0).fit(geo)
                distortions.append(sum(np.min(cdist(geo, kmeans.cluster_centers_, 'euclidean'), axis=1)) / geo.shape[0])
                df_distortion = pd.DataFrame(distortions, columns=['sum_square_distances_to_center'])
                df_distortion.loc[:, 'freq'] = f
            df_distortion_all = pd.concat([df_distortion_all, df_distortion])
        df_distortion_all.loc[:, 'distortion_moving_diff'] = df_distortion_all.groupby(['freq'])[
            'sum_square_distances_to_center'].diff(-1)
        df_distortion_all.to_csv(os.path.join(self.output_path, config.OUTPUT_K_SELECTION), index='No_cluster')
        cass_zip['label'] = kmeans.labels_
        cass_zip.to_csv(os.path.join(self.output_path, config.OUTPUT_CASSZIP_FILE_KLABELS), index=False)

    def dbscan(self, cass_zip_file):

        cass_zip_cluster = pd.DataFrame()
        geo_focus = cass_zip_file.copy()
        unique_freq = geo_focus.freq.unique()
        etl = ETL_data()
        for f in unique_freq:
            g_df = geo_focus[geo_focus.freq == f]
            distance_matrix = etl.distance_on_sphere_numpy(g_df.loc[:, ['longitude', 'latitude']])
            db = DBSCAN(eps=config.EPS, min_samples=config.MIN_SAMPLES, metric=config.METRIC,
                        leaf_size=config.LEAF_SIZE)
            db.fit(distance_matrix)
            g_df.loc[:, 'label'] = db.labels_
            cass_zip_cluster = pd.concat([cass_zip_cluster, g_df])
        cass_zip_cluster.to_csv(os.path.join(self.output_path, config.OUTPUT_SUPPLIER_CLUSTER_FILE), index=False)
        cass_zip_cluster.to_pickle(os.path.join(self.output_path, config.OUTPUT_SUPPLIER_CLUSTER_PKL))
        return None


def main_preprocess():
    etld = ETL_data()
    zipcode_input_file = pd.read_csv(os.path.join(config.INPUT_PATH, config.INPUT_ZIPCODE_FILE))
    zipcode = etld.clean_zip(zipcode_input_file)
    dfcass = etld.clean_cass(source_file_path=config.SHARE_DRIVE_PATH,
                             dest_zip=config.DESTINATION_DEPORT_ZIP,
                             source_state=config.SOURCE_STATE,
                             source_country=config.SOURCE_COUNTRY,
                             shipping_date_start=config.SHIPPING_WINDOW_START,
                             shipping_window=config.SHIPPING_WINDOW_SPAN,
                             truck_mode=config.TRUCK_MODE,
                             inbound_indicator=config.SHIPPING_INDICATOR)

    df_name_convention = etld.name_convention(df=dfcass, level=config.FUZZY_LEVEL)

    df_result = etld.period_freq(df_name_convention, freq_list=config.RESAMPLE_LIST,
                                 period_level=config.PERIOD_FREQ_LEVEL)

    df_freq_agg = etld.agg_measurement(df_result, freq_list=config.RESAMPLE_LIST)

    test_unique = df_freq_agg[df_freq_agg.duplicated(['shipper_zip', 'unique_name'], keep=False)]
    if test_unique.shape[0] != 0:
        print('go back to check df_freq_agg has duplication')
        return None
    cass_zip = etld.cass_merge_zip(df_freq_agg, zipcode)
    cass_zip.to_csv(os.path.join(config.OUTPUT_PATH, config.OUTPUT_CASSZIP_CSV), index=False)
    return cass_zip


def main_cluster(cass_zip_input):
    KMM = ClusterModel(output_path=config.OUTPUT_PATH)
    KMM.kcluster(cass_zip_file=cass_zip_input, k_range=config.K_MAX)
    KMM.dbscan(cass_zip_file=cass_zip_input)


if __name__ == "__main__":
    _list = list(np.random.randint(1, 9, 4))
    print(f'###### program is running......\n### Play a 24 Game using [+ - * /] to get 24\n>>> {_list}\n')
    cass_zip = main_preprocess()
    print('###### cass_zip from main_process output')
    main_cluster(cass_zip_input=cass_zip)
    print('###### kmeans & dbscan runs successfully #######')
