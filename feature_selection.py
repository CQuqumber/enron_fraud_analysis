from explore_data import poiemails, odd_payments
import matplotlib.pyplot as plt
import numpy as np


def test_kbest(features,labels,features_list):

    from sklearn.feature_selection import SelectKBest

    x = 0

    kbest = SelectKBest()
    kbestrans = kbest.fit_transform(features,labels)

    for f in features_list:
        if x > 0:
            print(f,'KBEST',kbest.scores_[x-1])
        x = x + 1

    return kbest

def new_features(data_dict):

    for items in data_dict:
        data_dict[items]['key_payments'] = data_dict[items]['salary'] + data_dict[items]['bonus'] + data_dict[items]['other']
        data_dict[items]['deferral_balance'] = data_dict[items]['deferral_payments'] + data_dict[items]['deferred_income']
        data_dict[items]['retention_incentives'] = data_dict[items]['long_term_incentive'] + data_dict[items]['total_stock_value']
        data_dict[items]['total_of_totals'] = data_dict[items]['total_payments'] + data_dict[items]['total_stock_value']
        data_dict[items]['poi_emails'] = poiemails(data_dict[items])
        data_dict[items]['odd_payments'] = odd_payments(data_dict[items],data_dict[items]['poi'],items)
        if data_dict[items]['salary'] != 0:
            data_dict[items]['bonus/salary'] = data_dict[items]['bonus']/data_dict[items]['salary']
            data_dict[items]['exercised_stock_options/salary'] = data_dict[items]['exercised_stock_options']/data_dict[items]['salary']
        else:
            data_dict[items]['bonus/salary'] = 0
            data_dict[items]['exercised_stock_options/salary'] = 0
        if data_dict[items]['key_payments'] != 0:
            data_dict[items]['retention_incentives/key_payments'] = data_dict[items]['retention_incentives']/data_dict[items]['key_payments']
        else:
            data_dict[items]['retention_incentives/key_payments'] = 0
        messagetotal = data_dict[items]['from_messages'] + data_dict[items]['to_messages']
        if messagetotal > 0:
            poitotal = data_dict[items]['from_this_person_to_poi'] + data_dict[items]['from_poi_to_this_person']
            data_dict[items]['poi_emailratio'] = poitotal/messagetotal
        else:
            data_dict[items]['poi_emailratio'] = 0

    return data_dict
