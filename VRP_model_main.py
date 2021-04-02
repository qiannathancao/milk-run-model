import numpy as np
import pandas as pd
import os
import sys
from collections import Counter
from collections import defaultdict
from sklearn.cluster import KMeans

sys.path.insert(0, '../main_generalized_run_rev_1')

import config

# from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class VrpPrep:
    """ to complete convert ipynb to py """

    def __init__(self, saved_output_path=None):
        self.path = saved_output_path

    def distance_on_sphere_numpy(self, coordinate_df):
        """
        Compute a distance matrix of the coordinates using a spherical metric.
        :param coordinate_df: numpy.ndarray with shape (n,2); latitude is in 1st col, longitude in 2nd.
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
        angle = (np.sin(phi_1) * np.sin(phi_2) * np.cos(theta_diff_mat) +
                 np.cos(phi_1) * np.cos(phi_2))
        arc = np.arccos(angle)

        # Multiply by earth's radius to obtain distance in km
        return np.nan_to_num(arc * EARTH_RADIUS * km2mile_ratio)

    def load_data(self, file):
        """
        :param
            file: to load OUTPUT_SUPPLIER_CLUSTER_FILE = 'cass_zip_cluster.csv'
        :return:
            dataframe: excluding cluster = -1
        """
        df = pd.read_pickle(os.path.join(self.path, file))
        df_copy = df.copy()
        df_copy = df_copy[df_copy.label != -1]  # drop label(cluster)=-1, which do not belong to any group
        df_copy['zip_code'] = df_copy['zip_code'].astype('str')
        df_copy['shipping_date'] = config.SHIPPING_WINDOW_START

        df_exceeds_capacity = df_copy[df_copy['ship_weight_freq_median'] >= config.VEHICLE_CAPACITY].reset_index(
            drop=True)
        df_exceeds_capacity.to_csv(os.path.join(self.path, config.OUTPUT_EXCEED_VEHICLE_LIMIT), index=False)

        df_copy = df_copy[df_copy['ship_weight_freq_median'] < config.VEHICLE_CAPACITY].reset_index(drop=True)
        df_copy.columns = ['shipper_zip', 'shipper_name', 'freq', 'ship_weight', 'ship_weight_annum',
                           'shipment_count_annum', 'billed_amount_annum', 'zip_code', 'longitude', 'latitude',
                           'state_abbreviation', 'label', 'shipping_date']
        print('load data completed')
        return df_copy


class VrpModel:
    def __init__(self):
        pass

    def create_data_model(self,
                          distance_matrix=None,
                          ship_weight_list=None,
                          each_vehicle_capacity=45000,
                          num_vehicles=30,
                          nrlocations=9):
        """

        :param distance_matrix: generated from distance_on_sphere_numpy
        :param ship_weight_list: from sliced cass_cluster's shipping weight
        :param each_vehicle_capacity: from config file
        :param num_vehicles: from config file
        :param nrlocations: from config file
        :return: dictionary of data for solver input
        """

        data = defaultdict()
        data['distance_matrix'] = distance_matrix
        data['demands'] = ship_weight_list
        data['vehicle_capacities'] = [each_vehicle_capacity] * num_vehicles
        data['num_vehicles'] = num_vehicles
        data['depot'] = 0
        data['nrLocations'] = nrlocations
        return data

    def print_solution(self, data, manager, routing, assignment):
        """Prints assignment on console."""
        total_distance = 0
        total_load = 0

        vehicle_routes = defaultdict()  # for list out the same truck pick zip codes

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            plan_output_backward = 'Route for vehicle {}:\n'.format(vehicle_id)  # if backward is shorter path
            route_distance = 0
            route_load = 0
            edge_distance = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                plan_output_backward += ' {0} Load({1}) <- '.format(node_index,
                                                                    route_load)  # if backward is shorter path
                previous_index = index
                index = assignment.Value(routing.NextVar(index))

                if vehicle_id in vehicle_routes:
                    vehicle_routes[vehicle_id].append(node_index)  # adding zip codes to same truck
                else:
                    vehicle_routes[vehicle_id] = [node_index]

                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                edge_distance.append(routing.GetArcCostForVehicle(previous_index, index, vehicle_id))

            # adding destination to entire route

            """ this situation is Fudging Headacheeeeeeee"""
            # distance from Greenville to first supplier is larger than last supplier to Greenville,
            # truck starts from first supplier, remove first span of driving from VRP
            if edge_distance[0] >= edge_distance[-1]:
                vehicle_routes[vehicle_id].append(0)
                vehicle_routes[vehicle_id].pop(0)
                route_distance = route_distance - edge_distance[0]
                plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index), route_load)
                plan_output += 'Distance of the route: {} miles\n'.format(route_distance)
                plan_output += 'Load of the route: {}\n'.format(route_load)

                total_distance += route_distance
                total_load += route_load

            # truck starts form last supplier,remove last span of driving from VRP
            else:
                route_distance = route_distance - edge_distance[-1]
                vehicle_routes[vehicle_id] = vehicle_routes[vehicle_id][::-1]
                plan_output_backward += ' {0} Load({1})\n'.format(manager.IndexToNode(index), route_load)
                plan_output_backward += 'Distance of the route: {} miles\n'.format(route_distance)
                plan_output_backward += 'Load of the route: {}\n'.format(route_load)
                total_distance += route_distance
                total_load += route_load
        return vehicle_routes

    def route_schedule(self, vehicle_routes, id_label=''):
        """
        generate route schedule as a readable format: {truck: supplier_index}
        """

        df = pd.DataFrame()
        for k in vehicle_routes.keys():
            if len(vehicle_routes[k]) == 1:  # this step eliminate dummy trucks like #0,#1 trucks doing nothing
                continue
            for v in vehicle_routes[k]:
                df = pd.concat([df, pd.DataFrame({'truck_number': [str(k) + id_label], 'pick_node': [v]})])
        _route_schedule = df.reset_index(drop=True)
        return _route_schedule

    def vrp_main_process(self, distance_matrix, ship_weight_list, id_label=''):
        """
        this vrp_main_process will iterate within 'vrp_route_in_frequency()'
        :param distance_matrix: generated from distance_on_sphere_numpy
        :param ship_weight_list: from sliced cass_cluster's shipping weight
        :param id_label: from sliced cass_cluster's label
        :return:
        """

        _data = self.create_data_model(distance_matrix=distance_matrix,
                                       ship_weight_list=ship_weight_list,
                                       each_vehicle_capacity=config.VEHICLE_CAPACITY,
                                       num_vehicles=config.VEHICLE_COUNTS,
                                       nrlocations=config.ROUTE_LOCATION_COUNTS)

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(len(_data['distance_matrix']), _data['num_vehicles'], _data['depot'])

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Register transit callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return _data['distance_matrix'][from_node][to_node]

        # Define cost of each arch
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add count_stops constraint
        count_stop_callback = routing.RegisterUnaryTransitCallback(lambda index: 1)
        dimension_name = 'Counter'
        routing.AddDimension(evaluator_index=count_stop_callback, slack_max=0, capacity=config.VEHICLE_CAPACITY,
                             fix_start_cumul_to_zero=True, name='Counter')

        # Add solver to count stop numbers
        counter_dimension = routing.GetDimensionOrDie(dimension_name)
        for vehicle_id in range(config.VEHICLE_COUNTS):
            index = routing.End(vehicle_id)
            solver = routing.solver()
            solver.Add(counter_dimension.CumulVar(index) <= config.VEHICLE_STOPS)

        # Add Capacity constraint
        def demand_callback(from_index):
            from_code = manager.IndexToNode(from_index)
            return _data['demands'][from_code]

        # Add Capacity constraint
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(evaluator_index=demand_callback_index,
                                                slack_max=0,
                                                vehicle_capacities=_data['vehicle_capacities'],
                                                fix_start_cumul_to_zero=True,
                                                name='Capacity')

        # Adding penalty for loading weight exceeds truck capacity
        penalty = config.PENALTY
        for node in range(1, len(_data['distance_matrix'])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        # Solve the problem_applying solver
        assignment = routing.SolveWithParameters(search_parameters)

        if assignment:
            _vehicle_routes = self.print_solution(_data, manager, routing, assignment)
            _route_schedule = self.route_schedule(_vehicle_routes, id_label=str(id_label))
            return _route_schedule
        return None


def vrp_route_in_frequency(cluster_copy, model=VrpModel(), hub_list=config.HUB_LIST, hub_name=config.HUB_NAME):
    """

    :param cluster_copy: output from VrpPrep().load_data()
    :param model: default as VrpModel()
    :param hub_list: default from config.HUB_LIST
    :param hub_name: default from config.HUB_NAME
    :return: dataframe: routed schedule with frequency
    """

    def distance_index(df, x):
        """
        param:
            df: distance matrix with UNIQUE index & columns
            x: truck location source and truck location next-stop
        return:
            DataFrame: distance matrix
        """

        try:
            result = round(df.loc[x[0], x[1]])
        except:
            result = 0

        return result

    route_by_freq = pd.DataFrame()
    freqs = cluster_copy.freq.unique()
    for f in freqs:
        ranks = sorted(cluster_copy[cluster_copy.freq == f]['label'].unique())
        route_by_rank = pd.DataFrame()

        for i, r in enumerate(ranks):
            label_no = Counter(cluster_copy[cluster_copy.freq == f]['label']).most_common()[r][0]
            cluster = cluster_copy[(cluster_copy.label == label_no) & (cluster_copy.freq == f)]
            cluster = cluster.sort_values(by='ship_weight', ascending=False)
            selected_hub = hub_list[hub_name]
            adjusted_hub_list = [
                f if i == 'freq' else label_no if i == 'label_no' else config.SHIPPING_WINDOW_START if i == 'shipping_start' else i
                for i in selected_hub]
            selected_hub_update = pd.DataFrame([adjusted_hub_list], columns=cluster.columns)

            cass_zip_cluster_copy = selected_hub_update.append(cluster).reset_index(drop=True)
            vrp_size = cass_zip_cluster_copy.shape[0]
            if vrp_size > 100:
                k = (vrp_size // 100) + 1
                route_over_100 = pd.DataFrame()

                element_counts_max = 101
                while element_counts_max >= 100:
                    kmeans = KMeans(k, random_state=0).fit(cass_zip_cluster_copy.loc[:, ['longitude', 'latitude']])
                    id_labels = kmeans.labels_
                    elements, element_counts = np.unique(id_labels, return_counts=True)
                    element_counts_max = element_counts.max()
                    cass_zip_cluster_copy['k_label'] = id_labels
                    k += 1

                for id_label in set(id_labels):
                    cass_zip_under_100 = cass_zip_cluster_copy[(cass_zip_cluster_copy.k_label == id_label) &
                                                               (cass_zip_cluster_copy.shipper_name != hub_name)]

                    cass_zip_toy = selected_hub_update.append(cass_zip_under_100).reset_index(drop=True)
                    distance_matrix_toy = VrpPrep().distance_on_sphere_numpy(cass_zip_toy)

                    unique_cass_zip_toy = cass_zip_toy.drop_duplicates(subset=['zip_code'])
                    unique_distance_matrix_toy = VrpPrep().distance_on_sphere_numpy(unique_cass_zip_toy)
                    df_unique_distance_matrix = pd.DataFrame(unique_distance_matrix_toy,
                                                             index=unique_cass_zip_toy.zip_code,
                                                             columns=unique_cass_zip_toy.zip_code)
                    ship_weight_list_toy = cass_zip_toy.ship_weight.tolist()
                    route_schedule_result = model.vrp_main_process(distance_matrix=distance_matrix_toy,
                                                                   ship_weight_list=ship_weight_list_toy,
                                                                   id_label=id_label)

                    route_in_weight = route_schedule_result.merge(cass_zip_toy, left_on='pick_node', right_index=True,
                                                                  how='left')

                    route_in_weight['next_zip_code'] = route_in_weight.groupby(['truck_number'])['zip_code'].shift(-1)

                    route_in_weight['next_shipper_name'] = route_in_weight.groupby(['truck_number'])['shipper_name']. \
                        shift(-1)

                    route_in_weight.loc[:, 'milk_run_distance'] = route_in_weight[['zip_code', 'next_zip_code']]. \
                        apply(lambda x: distance_index(df_unique_distance_matrix, x), axis=1)

                    route_in_weight['stop_number'] = route_in_weight.groupby('truck_number').cumcount()

                    route_over_100 = pd.concat([route_over_100, route_in_weight])

                route_by_rank = pd.concat([route_by_rank, route_over_100])

            else:
                cass_zip_toy = cass_zip_cluster_copy.copy().reset_index(drop=True)
                distance_matrix_toy = VrpPrep().distance_on_sphere_numpy(cass_zip_toy)

                unique_cass_zip_toy = cass_zip_toy.drop_duplicates(subset=['zip_code'])

                unique_distance_matrix_toy = VrpPrep().distance_on_sphere_numpy(unique_cass_zip_toy)

                df_unique_distance_matrix = pd.DataFrame(unique_distance_matrix_toy, index=unique_cass_zip_toy.zip_code,
                                                         columns=unique_cass_zip_toy.zip_code)

                ship_weight_list_toy = cass_zip_toy.ship_weight.tolist()

                route_schedule_result = model.vrp_main_process(distance_matrix=distance_matrix_toy,
                                                               ship_weight_list=ship_weight_list_toy)

                route_in_weight = route_schedule_result.merge(cass_zip_toy, left_on='pick_node', right_index=True,
                                                              how='left')

                route_in_weight['next_zip_code'] = route_in_weight.groupby(['truck_number'])['zip_code'].shift(-1)

                route_in_weight['next_shipper_name'] = route_in_weight.groupby(['truck_number'])['shipper_name']. \
                    shift(-1)

                route_in_weight.loc[:, 'milk_run_distance'] = route_in_weight[['zip_code', 'next_zip_code']].apply(
                    lambda x: distance_index(df_unique_distance_matrix, x), axis=1)

                route_in_weight['stop_number'] = route_in_weight.groupby('truck_number').cumcount()

                route_by_rank = pd.concat([route_by_rank, route_in_weight])

        route_by_freq = pd.concat([route_by_freq, route_by_rank])

    return route_by_freq


if __name__ == "__main__":
    vrp_prep = VrpPrep(saved_output_path=config.OUTPUT_PATH)
    cass_cluster = vrp_prep.load_data(config.OUTPUT_SUPPLIER_CLUSTER_PKL)
    vrp_model = VrpModel()
    _route_by_freq = vrp_route_in_frequency(cluster_copy=cass_cluster, model=vrp_model)
    _route_by_freq.to_csv(os.path.join(config.OUTPUT_PATH, config.OUTPUT_ROUTE_BY_FREQUENCY),
                          index_label='time_sequence')

    print('VRP_model_main complete !')
