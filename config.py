# ===================================NO CHANGE, NO CHANGE, NO CHANGE, STATIC VARIABLES==============================
# Share_drive_path data source:
SHARE_DRIVE_PATH = 'S:\\OSK-Share\\DEPT\\LOGISTICS\\LC3\\Freight Payment and Reports\\Cognos Reports'
INPUT_ZIPCODE_FILE = 'zipcode_Lon_Lat.csv'

# change following output to SQL server in the future
OUTPUT_SUPPLIER_CLUSTER_FILE = 'cass_zip_cluster.csv'
OUTPUT_SUPPLIER_CLUSTER_PKL = 'cass_zip_cluster.pkl'

OUTPUT_CASSZIP_CSV = 'suppliers_geo.csv'
OUTPUT_CASSZIP_FILE_KLABELS = 'geo_cass.csv'
OUTPUT_K_SELECTION = 'k_selection.csv'
OUTPUT_EXCEED_VEHICLE_LIMIT = 'shipment_weight_over_truck_limit.csv'
OUTPUT_ROUTE_BY_FREQUENCY = 'route_test_2.csv'

HUB_LIST = {'GREENVILLE': ['54942', 'GREENVILLE_WH', 'freq', 0, 0, 0, 0, '54942', -88.53557, 44.293820,
                           'WI', 'label_no', 'shipping_start'],

            'CHANBERSBURG': ['17201', 'CHANBERSBURG_WH', 'freq', 0, 0, 0, 0, '17201', -77.6614, 39.93112,
                             'PA', 'label_no', 'shipping_start'],

            'APPLETON': ['54912', 'APPLETON_WH', 'freq', 0, 0, 0, 0, '54912', -88.42000, 44.26000,
                         'WI', 'label_no', 'shipping_start'],

            'NEENAH': ['54956', 'NEENAH_WH', 'freq', 0, 0, 0, 0, '54956', -88.5179, 44.2004,
                         'WI', 'label_no', 'shipping_start']}


# ==================================CHANGE, CHANGE, CHANGE,INPUT PATHS==========================
INPUT_PATH = r'S:\OSK-Share\DEPT\LOGISTICS\LC3\Projects\Logistics Optimization Tool\Route_Optimization_Tool\FEA_MilkRun\data_input'
OUTPUT_PATH = r'S:\OSK-Share\DEPT\LOGISTICS\LC3\Projects\Logistics Optimization Tool\Route_Optimization_Tool\FEA_MilkRun\data_output\data_output_appleton_54912'

# ======= CASS SELECTION PARAMETERS =======
FY_SELECTION = 'FY2*Invoice Detail.{}'

# CLUSTERING PARAMETER
HUB_NAME = 'APPLETON'
SHIPPING_WINDOW_START = '2019-09-01'
SHIPPING_WINDOW_SPAN = 900
SHIPPING_INDICATOR = 'INBOUND'
SOURCE_STATE = 'all'
SOURCE_COUNTRY = 'US'
DESTINATION_DEPORT_ZIP = HUB_LIST[HUB_NAME][0]

# ======= MODEL SELECTION HYPER-PARAMETER =======

FUZZY_LEVEL = 86  # similarity level of supplier's names
PERIOD_FREQ_LEVEL = 0.5  # frequency of gap allowing
RESAMPLE_LIST = ['W', '2W', 'M']  # frequency categories

# CLUSTERING HYPER-PARAMETERS
K_MAX = 6  # suppliers groups to cluster in first PowerBI page
EPS = 80  # DBSCAN supplier to supplier distance max iin between
MIN_SAMPLES = 2  # number of suppliers in same group using DBSCAN
METRIC = 'precomputed'  # distance matrix to be utilized
LEAF_SIZE = 30

# VRP HYPER-PARAMETERS
VEHICLE_CAPACITY = 45000  # max single trailer capacity
CLUSTER_RANK = 1
VEHICLE_COUNTS = 30
VEHICLE_STOPS = 7
ROUTE_LOCATION_COUNTS = 5
PENALTY = 10000

# VRP PARAMETERS
TRUCK_MODE = 'TL'
CLUSTER_STATES = 'TX,IA,OH,NC,WI,IL,PA'