#!/usr/bin/env python
# -*- coding: utf-8
# # algorithm:

# ### 1. import modules
import networkx as nx
import os
from tqdm import tqdm
tqdm.pandas()
import cx_Oracle as cx
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, String, Integer
import yaml

with open(r'config.yml', 'rb') as f:
    config = yaml.load(f)


# load configuration
# parameters
group_size_threshold = config['parameter']['max_group_number']
# database
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
user_name1 = config['database']['engine1']['user']
password1 = config['database']['engine1']['password']
ip1 = config['database']['engine1']['ip_address']
port1 = config['database']['engine1']['port']

user_name2 = config['database']['engine2']['user']
password2 = config['database']['engine2']['password']
ip2 = config['database']['engine2']['ip_address']
port2 = config['database']['engine2']['port']

t_temp = config['database']['table1']['table_pairs_person_id']
table_name = config['database']['table1']['table_registration']
table_result = config['database']['table1']['table_result']
table_output = config['database']['table2']['table_output']

# database connection
engine1 = create_engine(
    'oracle://%s:%s@%s:%s/orcl?charset=utf8' % (user_name1, password1, ip1, port1))
conn1 = cx.connect('%s/%s@%s/orcl' % (user_name1, password1, ip1))
cursor1 = conn1.cursor()

engine2 = create_engine(
    'oracle://%s:%s@%s:%s/orcl?charset=utf8' % (user_name2, password2, ip2, port2))
conn2 = cx.connect('%s/%s@%s/orcl' % (user_name2, password2, ip2))
cursor2 = conn2.cursor()


def find_subgraphs(data):
    """
    use networkx to find sub graphs of a dataframe
    Keyword Arguments:
    data -- dataframe, (id1, id2)

    Return:
    all_sub -- list of sub graph nodes
    """
    G = nx.Graph()
    for i in range(data.shape[0]):
        G.add_edge(data.iloc[i, 0], data.iloc[i, 1], weight=1)
    data_sub = nx.connected_component_subgraphs(G, copy=True)
    all_sub = [list(i.nodes()) for i in data_sub]
    del G
    return all_sub


def long_term_hospitalization(dataframe, days=30):
    """
    是否长时间住院
    dataframe: data
    index: multi index in groups
    columns: [check in date column name, check out date column name]
    days: hospitalization days
    """
    df_checkout_checkin = dataframe
    diff_checkout_checkin = df_checkout_checkin.diff(
        axis=1).iloc[:, -1] / np.timedelta64(1, 'D')
    if (diff_checkout_checkin > days).any():
        return True
    else:
        return False


def continuous_checkin_checkout(dataframe, days=1):
    """
    是否连续出入院
    dataframe: data
    index: multi index in groups
    columns: [check in date column name, check out date column name]
    days: minimum date difference number in days
    """
    diff_checkout_next_checkin = pd.DataFrame()
    diff_checkout_next_checkin['date'] = dataframe.stack().sort_values()
    diff_checkout_next_checkin['diff'] = diff_checkout_next_checkin[
        'date'].diff() / np.timedelta64(1, 'D')
    # diff_checkout_next_checkin['diff'].fillna(0, inplace=True)
    if (diff_checkout_next_checkin['diff'] <= days).any():
        return True
    else:
        return False


def parse_data(dataframe):
    """
    stack paired data
    Keyword Arguments:
    dataframe -- DataFrame, (id1, id2)

    Return:
    df_patient -- DataFrame, one patient per row
    """
    dataframe.sort_values(['个人ID1', '个人ID2', '入院日期1'], inplace=True)
    groups = find_subgraphs(daframe)

    # ### parse data for person per row
    df_person_1 = dataframe[[
        '个人ID1', '姓名1', '就诊号1', '机构', '证件号1', '入院日期1', '出院日期1', '出院诊断名称1', '出院诊断代码1',
        '总费用1'
    ]]
    df_person_2 = dataframe[[
        '个人ID2', '姓名2', '就诊号2', '机构', '证件号2', '入院日期2', '出院日期2', '出院诊断名称2', '出院诊断代码2',
        '总费用2'
    ]]

    df_patient = pd.DataFrame(
        data=np.concatenate([df_person_1.values, df_person_2.values]),
        columns=[
            '个人ID', '姓名', '就诊号', '机构', '证件号', '入院日期', '出院日期', '出院诊断名称', '出院诊断代码', '总费用'
        ])

    df_patient.sort_values(['个人ID', '入院日期'], inplace=True)
    df_patient.drop_duplicates(['个人ID', '入院日期'], inplace=True)

    df_patient.set_index('个人ID', inplace=True)
    for key, value in enumerate(groups):
        df_patient.loc[value, 'group_sn'] = key
    df_patient['subgroup_sn'] = 0
    df_patient['risk_score'] = 0
    df_patient['group_type'] = '0'
    return df_patient


def analize_all_data(df_patient):
    df_patient.reset_index(inplace=True)
    df_patient.set_index('就诊号', inplace=True)
    df_patient.sort_values(['group_sn', '入院日期'], inplace=True)
    for group, data in df_patient.groupby('group_sn'):
        # data.sort_values('入院日期', inplace=True)
        # df_group_sn_temp = df_patient[df_patient['group_sn']==group]
        data['checkin_diff'] = data['入院日期'].diff() / np.timedelta64(1, 'D')
        index_org = data.index
        data = data.reset_index()
        index_s = data[data['checkin_diff'] > 3].index

        for k, v in enumerate(index_s):
            try:
                # print(index_s[k], index_s[k+1], k+1)
                # print(index_org[v:index_s[k+1]])
                df_patient.loc[index_org[v:index_s[k + 1]],
                               'subgroup_sn'] = k + 1
                # print(data.iloc[index[k], -1])
            except IndexError:
                df_patient.loc[index_org[v:], 'subgroup_sn'] = k + 1
    df_patient.reset_index(inplace=True)
    df_patient.set_index('个人ID', inplace=True)
    group_size = df_patient.reset_index().groupby(
        ['group_sn'])['个人ID'].unique().apply(len)
    big_groups = df_patient[df_patient['group_sn'].isin(
        group_size[group_size >= 8].index)]
    small_groups = df_patient[df_patient['group_sn'].isin(
        group_size[group_size < 8].index)]
    df_long_term = small_groups.groupby('个人ID').progress_apply(
        lambda x: long_term_hospitalization(x[['入院日期', '出院日期']], days=30))
    df_cont = small_groups.groupby('个人ID').progress_apply(
        lambda x: continuous_checkin_checkout(x[['入院日期', '出院日期']]))
    return df_patient, df_long_term, df_cont, small_groups, big_groups


# ### export data
def export(df_patient, df_long_term, df_cont, small_groups, big_groups):
    all_small_groups = np.unique(small_groups['group_sn'].values)
    hospital_size = small_groups.groupby(
        ['group_sn'])['机构'].unique().apply(len)
    all_small_groups_in_same_hospital = hospital_size[hospital_size < 2].index
    all_small_groups_not_in_same_hospital = hospital_size[hospital_size >= 2].index

    groups_long_term = small_groups.loc[df_long_term, 'group_sn'].unique()
    groups_long_term_in_same_hospital = [
        x for x in groups_long_term if x in all_small_groups_in_same_hospital]

    groups_cont = small_groups.loc[df_cont, 'group_sn'].unique()
    groups_cont_in_same_hospital = [
        x for x in groups_cont if x in all_small_groups_in_same_hospital]

    groups_reg = [x for x in all_small_groups_in_same_hospital
                  if x not in groups_long_term_in_same_hospital
                  and x not in groups_cont_in_same_hospital]

    cont_groups_except_lt = [x for x in all_small_groups_in_same_hospital
                             if x not in groups_long_term_in_same_hospital
                             and x in groups_cont_in_same_hospital]

    reg_groups_except_lt_cont = [x for x in all_small_groups_in_same_hospital
                                 if x not in groups_long_term
                                 and x not in groups_cont and x in groups_reg]
    output_columns = df_patient.reset_index().columns.tolist()
    small_groups.reset_index(inplace=True)
    l_risky_disease = ['肾衰竭', '手术后状态', '精神', '恶性肿瘤', '癌', '肿瘤', '艾滋病', '骨折', '骨肉瘤',
                       '白血病', '脑瘫', '瘫痪', '心力衰竭', '梅尼埃', '美尼尔', '红斑狼疮', '肺结核', '淋巴瘤',
                       '血友病', '骨坏死', '恶性']
    small_groups['risky_disease'] = 10
    for d in l_risky_disease:
        risky_index = small_groups[small_groups['出院诊断名称'].str.contains(
            d)].index
        small_groups.loc[risky_index, 'risky_disease'] = 5
    average_group_diagnosis_number = small_groups.groupby('group_sn')['出院诊断名称'].unique().apply(len) /\
        small_groups.groupby('group_sn')['个人ID'].unique().apply(len)
    max_average_group_diagnosis_number = max(average_group_diagnosis_number)

    average_group_cost = small_groups.groupby('group_sn')['总费用'].sum(
    ) / small_groups.groupby('group_sn')['个人ID'].unique().apply(len)
    max_average_group_cost = max(average_group_cost)

    small_groups['LOS'] = (small_groups['出院日期'] -
                           small_groups['入院日期']) / np.timedelta64(1, 'D')
    average_group_LOS = small_groups.groupby('group_sn')['LOS'].sum(
    ) / small_groups.groupby('group_sn')['个人ID'].unique().apply(len)
    max_average_group_LOS = max(average_group_LOS)

    small_groups['risky_disease_avg'] = small_groups.groupby('group_sn')['risky_disease'].sum(
    ) / small_groups.groupby('group_sn')['个人ID'].unique().apply(len)

    group_risky_disease = small_groups.groupby('group_sn')['risky_disease'].sum(
    ) / small_groups.groupby('group_sn')['姓名'].apply(len)

    score_rule = list(range(1, 11, 1))
    group_risky_score = average_group_LOS.apply(lambda x: score_rule[int(np.ceil(x * 10 / max_average_group_LOS) - 1)]) + \
        average_group_diagnosis_number.apply(lambda x: score_rule[int(np.ceil(x * 10 / max_average_group_diagnosis_number) - 1)]) +\
        average_group_cost.apply(lambda x: score_rule[int(np.ceil(x * 10 / max_average_group_cost) - 1)]) +\
        group_risky_disease
    small_groups['risk_score'] = small_groups['group_sn'].map(
        group_risky_score)

    df_long_result = small_groups[small_groups['group_sn'].isin(
        groups_long_term_in_same_hospital)].set_index('就诊号')
    df_cont_result = small_groups[small_groups['group_sn'].isin(
        cont_groups_except_lt)].set_index('就诊号')
    df_reg_same_hos_result = small_groups[small_groups['group_sn'].isin(
        reg_groups_except_lt_cont)].set_index('就诊号')
    df_reg_not_same_hos_result = small_groups[small_groups['group_sn'].isin(
        all_small_groups_not_in_same_hospital)].set_index('就诊号')

    df_patient.reset_index(inplace=True)
    df_patient.set_index('就诊号', inplace=True)

    df_patient.loc[df_long_result.index,
                   'risk_score'] = df_long_result['risk_score']
    df_patient.loc[df_cont_result.index,
                   'risk_score'] = df_cont_result['risk_score']
    df_patient.loc[df_reg_same_hos_result.index,
                   'risk_score'] = df_reg_same_hos_result['risk_score']
    df_patient.loc[df_reg_not_same_hos_result.index,
                   'risk_score'] = df_reg_not_same_hos_result['risk_score']

    df_patient.loc[df_long_result.index, 'group_type'] = '1'
    df_patient.loc[df_cont_result.index, 'group_type'] = '2'
    df_patient.loc[df_reg_same_hos_result.index, 'group_type'] = '3'
    df_patient.loc[df_reg_not_same_hos_result.index, 'group_type'] = '4'

    sql_insert = """INSERT INTO %s   (PATIENT_ID, MEDICAL_CLINIC_ID, HOSPITAL_ID, ADMISSION_DATE,
               DISCHARGE_DATE, DIAGNOSIS, GROUP_SN, GROUP_TYPE, RISK_SCORE, SUBGROUP_SN, DISEASE_CODE)
               VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11)"""
    df_patient.reset_index(inplace=True)
    counter = 0
    for idx, data in df_patient.iterrows():
        counter += 1
        if counter % 50 == 0:
            print(counter)
            # print(data)
            # break
        try:
            cursor2.execute(sql_insert % table_output, (str(data['个人ID']),
                                                        str(data['就诊号']),
                                                        str(data['机构']),
                                                        data['入院日期'],
                                                        data['出院日期'],
                                                        str(data['出院诊断名称']),
                                                        str(data['group_sn']),
                                                        str(data['group_type']),
                                                        str(data['risk_score']),
                                                        str(data['subgroup_sn']),
                                                        str(data['出院诊断代码'])
                                                        ))
        except cx.IntegrityError:
            # print('number error')
            pass
        # except cx.DatabaseError:
            # pass
    conn2.commit()


def prepare_data(t_temp, table_name, table_result):
    """prepare paired data, a pair of person must appears more than 3 times in a hospital at the same time in a time period.
    Keyword Arguments:
    t_temp       -- id pairs
    table_name   -- registration table name
    table_result -- paired person data table
    """
    try:
        cursor1.execute('drop table %s' % t_temp)
    except cx.DatabaseError as e:
        error, = e.args
        print(e)
        if error.code == 1017:
            print('表或视图不存在')
    sql_create_temp = """
   create table {0} as
select t1.person_id person_1, t2.person_id person_2
  from {1} t1
  join {2} t2 on t1.med_ser_org_no = t2.med_ser_org_no
                and t1.person_id > t2.person_id
                and abs(to_date(substr(t1.in_hosp_date, 1, 8), 'yyyymmdd') -
                        to_date(substr(t2.in_hosp_date, 1, 8), 'yyyymmdd')) <= 2
                and abs(to_date(substr(t1.out_hosp_date, 1, 8), 'yyyymmdd') -
                        to_date(substr(t2.out_hosp_date, 1, 8), 'yyyymmdd')) <= 2
 where t1.clinic_type = '2'
   and t2.clinic_type = '2'
   and t1.med_ser_org_no = t2.med_ser_org_no
   and t1.in_hosp_date like '201802%'
   and t2.in_hosp_date like '201802%'
  and not exists (select 1
          from t_disease t3
         where t1.out_diag_dis_cd = t3.disease_code
         and cheat_flag = '1')
   and not exists (select 1
          from t_disease t4
         where t2.out_diag_dis_cd = t4.disease_code
         and cheat_flag = '1')
 group by t1.person_id, t2.person_id
having count(distinct t1.med_clinic_id) >= 3 and count(distinct t2.med_clinic_id) >= 3
"""
    try:
        cursor1.execute(
            sql_create_temp.format(t_temp, table_name, table_name))
    except cx.DatabaseError as e:
        print(e)
        error, = e.args
    # drop final result table
    try:
        cursor1.execute('drop table %s' % table_result)
    except cx.DatabaseError as e:
        print(e)
        error, = e.args
        if error.code == 955:
            print('表或视图不存在')
    sql_final_result = """
    create table %s as
    select t2.%s 个人ID1,
           t3.%s 个人ID2,
           t2.%s 姓名1,
           t3.%s 姓名2,
           t2.%s 就诊号1,
           t3.%s 就诊号2,
           t2.%s 机构,
           t2.%s 证件号1,
           t3.%s 证件号2,
           to_date(substr(t2.%s, 1, 8), 'YYYYMMDD') 入院日期1,
           to_date(substr(t3.%s, 1, 8), 'YYYYMMDD') 入院日期2,
           to_date(substr(t2.%s, 1, 8), 'YYYYMMDD')  出院日期1,
           to_date(substr(t3.%s, 1, 8), 'YYYYMMDD')  出院日期2,
           t2.out_diag_dis_nm 出院诊断名称1,
           t3.out_diag_dis_nm 出院诊断名称2,
           t2.out_diag_dis_cd 出院诊断代码1,
           t3.out_diag_dis_cd 出院诊断代码2,
           t2.med_amout 总费用1,
           t3.med_amout 总费用2
      from %s t1
      join %s t2 on t1.person_1 = t2.person_id
      join %s t3 on t1.person_2 = t3.person_id
                    and abs(to_date(substr(t2.in_hosp_date, 1, 8), 'yyyymmdd') -
                            to_date(substr(t3.in_hosp_date, 1, 8), 'yyyymmdd')) <= 2
                    and abs(to_date(substr(t2.out_hosp_date, 1, 8), 'yyyymmdd') -
                            to_date(substr(t3.out_hosp_date, 1, 8), 'yyyymmdd')) <= 2
                    and t2.med_ser_org_no = t3.med_ser_org_no
     where t2.clinic_type = '2'
       and t3.clinic_type = '2'
    """ % (table_result,
           config['database']['columns']['person_id'],
           config['database']['columns']['person_id'],
           config['database']['columns']['person_name'],
           config['database']['columns']['person_name'],
           config['database']['columns']['medical_id'],
           config['database']['columns']['medical_id'],
           config['database']['columns']['hospital_id'],
           config['database']['columns']['person_dentification'],
           config['database']['columns']['person_dentification'],
           config['database']['columns']['admission_date'],
           config['database']['columns']['admission_date'],
           config['database']['columns']['discharge_date'],
           config['database']['columns']['discharge_date'],
           t_temp, table_name, table_name)
    try:
        cursor1.execute(sql_final_result)
    except cx.DatabaseError as e:
        error, = e.args
        print(e)
    # cursor1.close()
    conn1.commit()


if __name__ == '__main__':
    print('prepare data')
    prepare_data(t_temp, table_name, table_result)

    sql_read_city = """
    select *  from %s"""
    print('load data')
    df_city = pd.read_sql_query(sql_read_city % (table_result), engine1)

    print('parse data')
    df_patient = parse_data(df_city)

    print('analize data')
    df_patient, df_long_term, df_cont, small_groups, big_groups = analize_all_data(
        df_patient)

    print('export data')
    export(df_patient, df_long_term, df_cont, small_groups, big_groups)
