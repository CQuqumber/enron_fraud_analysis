import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#Various lists

finlist = ['director_fees','exercised_stock_options','restricted_stock_deferred']

emaillist = ['to_messages','from_messages','shared_receipt_with_poi',
'from_this_person_to_poi','from_poi_to_this_person']

poiemaillist = ['shared_receipt_with_poi','from_this_person_to_poi','from_poi_to_this_person']


finlistall = ['total_payments','deferred_income', 'salary','director_fees','long_term_incentive',
'expenses','exercised_stock_options','restricted_stock_deferred','restricted_stock',
'loan_advances','bonus','total_stock_value','deferral_payments','other']

newfeaturelist = ['exercised_stock_options/salary','bonus/salary','retention_incentives/key_payments',
'no_dirdefrestock','odd_payments','key_payments','deferral_balance','retention_incentives','total_of_totals','poi_emails']


#Function Type 1: Searching for outliers

def totalPOIs(list):
#this function counts the total POIs
    count = 0

    for items in list:
        if list[items]['poi']==1:
            count +=1
    return count

def findnulls(list):
#this function counts the null values for all and just POI's for any list
    nullcount = defaultdict(int)
    poi_nullcount = defaultdict(int)

    for items in list:
        for i in list[items]:
            if list[items][i] == "NaN":
                nullcount[i] +=1
                if list[items]['poi']==1:
                    poi_nullcount[i] +=1

    return nullcount,poi_nullcount

def isnulllist(data,list):
#this function complies a list of null values (all & POI) for the given feature list

    allitems = True
    mylist = []
    mypoilist = []
    for items in data:
        for l in list:
            if data[items][l] != "NaN":
                allitems = False
        if allitems == True:
            mylist.append(items)
            if data[items]['poi']==1:
                mypoilist.append(items)
        allitems = True

    return mylist, mypoilist

def nullPOInames(data,var):
#this function is used to list the names of POIs with NaN values in the given list

    mylist = []

    for items in data:
        if data[items][var] == "NaN" and data[items]['poi']==1:
            mylist.append(items)

    return mylist

def nullPOIexceptions(data,var):
#this function is used to list the names of POIs WITHOUT NaN values on the given list
    mylist = []

    for items in data:
        if data[items][var] != "NaN" and data[items]['poi']==1:
            mylist.append(items)

    return mylist


def printhighnonPOIs(data_dict):
#this function prints names of high non-POI values that I identified from the graphs

    ishigh = False

    for items in data_dict:

        for i in data_dict[items]:
            if data_dict[items]["poi"]==0:
                if i == 'bonus':
                    if data_dict[items][i] > 6000000:
                        ishigh = True
                if i == 'expenses':
                    if data_dict[items][i] > 200000:
                        ishigh = True
                if i == 'deferral_payments':
                    if data_dict[items][i] > 6000000:
                        ishigh = True
                if i == 'salary':
                    if data_dict[items][i] > 1000000:
                        ishigh = True
                if i == 'exercised_stock_options':
                    if data_dict[items][i] > 4000000:
                        ishigh = True
                if i == 'restricted_stock':
                    if data_dict[items][i] > 10000000:
                        ishigh = True
                if i == 'long_term_incentive':
                    if data_dict[items][i] > 3000000:
                        ishigh = True
                if i == 'total_stock_value':
                    if data_dict[items][i] > 20000000:
                        ishigh = True
                if i == 'other':
                    if data_dict[items][i] > 6000000:
                        ishigh = True
                if i == 'deferred_income':
                    if data_dict[items][i] < -3000000:
                        ishigh = True
        if ishigh == True:
            print(items,': ',i,' are ',data_dict[items][i])
            ishigh = False


def checktotals(data_dict):
#checking whether total payments and total stock value adds up...

    for items in data_dict:
        total = data_dict[items]['other']+data_dict[items]['deferral_payments']+data_dict[items]['long_term_incentive']+data_dict[items]['salary']+data_dict[items]['bonus']+data_dict[items]['expenses']+data_dict[items]['loan_advances']+data_dict[items]['deferred_income']+data_dict[items]['director_fees']
        stocktotal = data_dict[items]['exercised_stock_options']+ data_dict[items]['restricted_stock_deferred'] + data_dict[items]['restricted_stock']
        if total != data_dict[items]['total_payments']:
            print (items,'payments dont tally, poi?:',data_dict[items]['poi'])
            #for i in finlistall:
            #    print(i, ":",data_dict[items][i])
        if stocktotal != data_dict[items]['total_stock_value']:
            print (items,'stock doesnt tally, poi?:',data_dict[items]['poi'])
            for i in finlistall:
                print(i, ":",data_dict[items][i])

        for items in data_dict:
            if data_dict[items]['deferral_payments'] < 0:
                print('defpay under 0',items, data_dict[items])
            if data_dict[items]['restricted_stock']<0:
                print('reststock under 0',items, data_dict[items])

#Function Type 2: Printing Outlier Search Results

def printresults(data_dict):
#this function goes through all the functions above and prints the results

    #key data facts
    print('No.of Players: ', len(data_dict))
    print('No.of POIs: ', totalPOIs(data_dict))

    #list the null values per field (all and POI)
    nullcount,poinullcount = findnulls(data_dict)
    for items in nullcount:
        print("NaN's in ",items, ":",nullcount[items])
        if items in poinullcount:
            print("POI NaN's in ",items, ":",poinullcount[items])

    #null groups identified in finance and email.
    finn,poi_finn = isnulllist(data_dict,finlist)
    emailn,poi_emailn = isnulllist(data_dict,emaillist)

    #prints out the null values summaries

    print('total with email and no message:', len(emailn))
    print('POIs email no message:',len(poi_emailn))
    print('total missing finance options:', len(finn))
    print('POIs missing finance options:',len(poi_finn))

    print('POIs email, no message', poi_emailn)
    print('POIs no salary', nullPOInames(data_dict,'salary'))
    print('POIs no restricted stock', nullPOInames(data_dict,'restricted_stock'))
    print('POIs no bonus', nullPOInames(data_dict,'bonus'))
    print('POIs no long term incentive', nullPOInames(data_dict,'long_term_incentive'))
    print('POIs no exercised_stock_options/3 fin fields', poi_finn)
    print('POIs no deferred income ', nullPOInames(data_dict,'deferred_income'))
    print('POIs deferral payments all null EXCEPT:', nullPOIexceptions(data_dict,'deferral_payments'))
    print('POIs loan advances all null EXCEPT:', nullPOIexceptions(data_dict,'loan_advances'))



    #checking these two values as they have a lot of financial nulls.
    print(data_dict['HIRKO JOSEPH'])
    print(data_dict['YEAGER F SCOTT'])

    #checking these two values which I'd identified as potential outliers.
    print(data_dict['THE TRAVEL AGENCY IN THE PARK'])
    print(data_dict['TOTAL'])

#Function Type 3: updating nulls and outliers

def updatenulls(data):

    for items in data:
        for i in data[items]:
            if data[items][i] == "NaN":
                data[items][i] = 0

def update_data_errors(data_dict):

    #function to update data entry errors
    for items in data_dict:
        if items == 'BELFER ROBERT':
            data_dict[items]['total_payments'] = 3285
            data_dict[items]['deferred_income'] = -102500
            data_dict[items]['director_fees'] = 102500
            data_dict[items]['expenses'] = 3285
            data_dict[items]['exercised_stock_options'] = 0
            data_dict[items]['restricted_stock_deferred'] = -44093
            data_dict[items]['restricted_stock'] = 44093
            data_dict[items]['total_stock_value'] = 0
            data_dict[items]['deferral_payments'] = 0
        if items == 'BHATNAGAR SANJAY':
            data_dict[items]['total_payments'] = 137864
            data_dict[items]['other'] = 0
            data_dict[items]['director_fees'] = 0
            data_dict[items]['expenses'] = 137864
            data_dict[items]['exercised_stock_options'] = 15456290
            data_dict[items]['restricted_stock_deferred'] = -2604490
            data_dict[items]['restricted_stock'] = 2604490
            data_dict[items]['total_stock_value'] = 15456290

#Function Type 4: drawing graphs

def drawboxes(finstats,poi_finstats):
#this function draws a descriptive stats graph for all our features
#next time I'll draw box plots instead as they would be show results better

    for items in finstats:

        #run bar graph function for totals
        fd,ftotal = des_stats(finstats[items])
        poi_fd, ptotal = des_stats(poi_finstats[items])

        fig, ax = plt.subplots()
        data_to_plot = []
        data_to_plot.append(finstats[items])
        data_to_plot.append(poi_finstats[items])
        p1 = ax.boxplot(data_to_plot,patch_artist=True)
        for box in p1['boxes']:
            # change outline color
            box.set( color='#7570b3', linewidth=2)
            # change fill color
            box.set( facecolor = '#1b9e77' )

        ## change color and linewidth of the whiskers
        for whisker in p1['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        for cap in p1['caps']:
            cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
        for median in p1['medians']:
            median.set(color='#b2df8a', linewidth=2)

        ## change the style of fliers and their fill
        for flier in p1['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        ax.set_xticklabels(['Non POIs','POIs'])
        total = str(ftotal + ptotal)
        mytitle = 'Box plot for ' + items + '(' + total + ')'
        ax.set_title(mytitle,fontsize=20)
        update_ylabels(ax)
        plt.show()

def update_ylabels(ax):
#function used by drawgraphs to add thousand delimiters
    ylabels = [format(label, ',.0f') for label in ax.get_yticks()]
    ax.set_yticklabels(ylabels,fontsize=16)

def graphstats(data,mylist):
#this function compiles the stats dictionarys which will be used in graphstats
#the default dictionary holds the feature names in the given list and the values

    stats = defaultdict(list)
    poi_stats = defaultdict(list)

    for items in data:
        for i in data[items]:
            if i in mylist:

                if data[items]['poi']==1:
                    poi_stats[i].append(data[items][i])
                else:
                    stats[i].append(data[items][i])

    return stats, poi_stats

def drawbars(finstats,poi_finstats):
#original function which draws a descriptive stats bar graph for all our features
#now using drawboxes and only with financials as no real info gain from emailstats
#next time I'll draw box plots instead as they would be show results better

    for items in finstats:
        fd,ftotal = des_stats(finstats[items])
        poi_fd, ptotal = des_stats(poi_finstats[items])
        ind = np.arange(len(fd))
        width = 0.3
        fig, ax = plt.subplots()
        p1 = ax.bar(ind,fd.values(),width,color='red')
        p2 = ax.bar(ind + width,poi_fd.values(),width,color='yellow')

        ax.set_xticks((0.3,1.3,2.3,3.3))
        ax.set_xticklabels(fd.keys(),fontsize=16)
        update_ylabels(ax)
        total = str(ftotal + ptotal)
        mytitle = 'Descriptive stats for ' + items + '(' + total + ')'
        ax.set_title(mytitle,fontsize=20)
        ax.legend((p1[0],p2[1]),('Non POIs','POIs'),bbox_to_anchor=(1.10, 1))
        plt.show()


    for items in emailstats:

        ed,etotal = des_stats(emailstats[items])

        plt.bar(range(len(ed)),ed.values())
        plt.xticks((0.5,1.5,2.5,3.5),ed.keys(),fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Descriptive stats for ' + items + '(' + str(etotal) + ')',fontsize=20)
        plt.show()


def des_stats(stats):
#old function used by drawbars to get descriptive stats.
    total = 0
    d = {}
    d['Min']= int(np.min(stats))
    d['Max']= int(np.max(stats))
    d['Mean']= int(np.mean(stats))
    d['Median']= int(np.median(stats))

    for items in stats:
        if items > 0 or items < 0:
            total +=1

    return d,total


#Function Type 5: Developing new features

def poiemails(mylist):
    #function for poi_emails feature
    val = False
    truecount= 0

    for m in mylist:
        if m in poiemaillist:
            if mylist[m] > 0:
                val = True
                truecount +=1
    return val

def odd_payments(data,poi,name):

    #function for detecting entries with very extreme payments
    odd = False

    d = {}

    #I started with a large combination but it turns out that a
    #combination of retention_incentives > 28,000,000 and
    #deferral_balance < -1,500,000 nets the same 5 POIs

    #d['other'] = 8000000
    #d['bonus'] = 4500000
    #d['loan_advances'] = 5000000
    #d['exercised_stock_options'] = 16000000
    #d['total_payments'] = 21000000
    #d['total_of_totals'] = 32000000
    #d['retention_incentives/key_payments'] = 1000

    d['retention_incentives'] = 28000000

    if data['deferral_balance'] < -1500000:
        odd = True

    for i in data:
        for ditems in d:
            if i == ditems:
                if data[i] > d[ditems]:
                    odd = True

    return odd

def featureimportance(data,feature_list):
    from sklearn.ensemble import RandomForestClassifier
    df= pd.DataFrame(data)
    X_train = df.iloc[:,1:]
    y_train = df.iloc[:,0]
    feat_labels = np.array(feature_list)
    forest = RandomForestClassifier(n_estimators=100,
                                    criterion = 'entropy',
                                    max_features = None,
                                    max_depth = 5,
                                    random_state=None,
                                    n_jobs=-1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    print(X_train.shape[1])
    plt.title('Feature Importances via Random Forest')
    plt.bar(range(X_train.shape[1]), 
            importances[indices],
            color='lightblue', 
            align='center')

    plt.xticks(range(X_train.shape[1]), 
               feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()
