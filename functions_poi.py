# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:37:24 2017

@author: Abir
"""
## Functions for creating new features
## Feature 1:bonus:salary ratio

def salary_Bonus(data_dict):
    for name,feature in data_dict.iteritems():
        if feature['bonus']!='NaN' and feature['salary'] != 'NaN':
            feature['bonus_salary_sum']=float(feature['bonus'])+float(feature['salary'])
        else:
            feature['bonus_salary_sum']='NaN'
    return(data_dict)
            
## Feature 2:ratio of 'from_poi_to_this_person' and 'from_messages(fm)'
def ratio_fm(data_dict):
    for name,feature in data_dict.iteritems():
        if feature['from_poi_to_this_person'] !='NaN' and feature['from_messages'] !='NaN':
            feature['poi_proportion_fm']=float(feature['from_poi_to_this_person'])/float(feature['from_messages'])
        else:
            feature['poi_proportion_fm']='NaN'
    return(data_dict)

## Feature 3:ratio of 'from_this_person_to_poi' and 'to_messages(tm)'
def ratio_tm(data_dict):
    for name,feature in data_dict.iteritems():
        if feature['from_this_person_to_poi'] !='NaN' and feature['to_messages'] != 'NaN':
            feature['poi_proportion_tm']=float(feature['from_this_person_to_poi'])/float(feature['to_messages'])
        else:
            feature['poi_proportion_tm']='NaN'
    return(data_dict)

