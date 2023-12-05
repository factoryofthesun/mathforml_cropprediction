import pandas as pd
import numpy as np
import us
import os
import gc
import re
def irrigation_intake():
    data = pd.read_csv('irrigation.csv')
    data.drop(columns='Name', inplace=True)
    # print(data.columns)
    data = data.groupby(by='stfips', ).mean()
    data.reset_index(inplace=True )
    # print(data)
    data = data[['stfips', 'i1950002', 'i1954002', 'i1959002', 'i1964002', 'i1969002',
                 'i1974002', 'i1978002', 'i1982002', 'i1987002', 'i1992002', 'i1997002', 'i2002002', 'i2007002', 'i2012002']]
    data = data.groupby(by='stfips').mean()
    data.reset_index(inplace=True)
    data = pd.wide_to_long(data, stubnames='i', i=['stfips'], j='year')
    data.reset_index(inplace=True)
    data = data.sort_values('stfips')
    fips_to_state = us.states.mapping('fips', 'name')
    str_key = []
    for i in fips_to_state.keys():
        if i is not None:
            str_key.append(int(i))
    state_to_fips = dict(map(lambda i, j: (i, j), str_key, fips_to_state.values()))
    data['stfips'] = data['stfips'].map(state_to_fips)
    data['year'] = data['year'].map(lambda x: int(str(x)[0:4]))
    for i in data['year']:
        print(type(i))
    data.columns = ['NAME', 'DATE', 'IRRIGATION']
    # data['NAME'] = data['stfips'].apply(state_to_fips)
    # fips_to_state = {str_key : fips_to_state.values}
    # fips_to_state.keys = int(fips_to_state.keys)

    # print(data.head)
    data.to_csv('out.csv')


def drought_importer():
    drought_data = pd.read_csv('temp_max.csv', sep='\s+', header=None, dtype={0:str} )
    drought_data['annual'] = drought_data.astype(float).iloc[:, 1:13].mean(axis=1)
    drought_data['year'] = drought_data[0]
    drought_data['state'] = drought_data[0]
    states = us.states.STATES

    for i, j in enumerate(drought_data[0]):
        drought_data['year'][i] = int(j[-4:])
        drought_data['state'][i] = int(j[0:3])
    drought_data = drought_data[drought_data.year > 1965]
    drought_data = drought_data[drought_data.state <= 51]
    drought_data.reset_index(inplace=True)
    states2 = []
    for i in states:
        if i.name != 'Hawaii' and i.name != 'Alaska':
            states2.append(i)
    states = states2
    for i in range(len(drought_data['state'])):
        code = drought_data.at[i, 'state']
        if code == 1:
            drought_data.at[i, 'state'] = 'Alabama'
        elif code == 50:
            drought_data.at[i, 'state'] = 'Alaska'
        else:
            drought_data.at[i, 'state'] = states[code-1].name
    drought_data = drought_data[['annual', 'state', 'year']]
    drought_data.to_csv('temp_max_data.csv')
    yield_data = pd.read_csv('yield_data.csv')
    # print(yield_data['State'].unique())
    u_states = pd.DataFrame(yield_data['State'].unique()).to_csv('ustates.csv')
folder = '/Users/chrisdoyle/Desktop/The Big Lebowski (1998) (1080p BluRay x265 HEVC 10bit HDR AAC 7.1 afm72)/'


def intake(folder):
    files = os.walk(folder)
    data = []
    for f in files:
        data += f
    root = data[0]
    data = data[2]
    for i in data:
        if '.DS_Store' in i:
            break
        date = pd.read_csv(root + '/' + i)
        if date['DATE'].min() < 1966:
            intake = pd.read_csv(root + '/' + i)
            intake.to_csv('/Users/chrisdoyle/Desktop/Freeze data_2/' + i)

# intake(folder)
def merge(folder, code):
    # print(os.listdir(folder))
    files = os.walk(folder)
    master_df_list = []
    data = []
    for f in files:
        data += f
    root = data[0]
    data = data[2]
    data = sorted(data)
    # print(data)
    prev_state = 'AL'
    state_df = pd.DataFrame()
    df_list = []
    first = True
    for i in data:
        if '.DS_Store' in i:
            continue
        date = pd.read_csv(root + '/' + i)
        # print(date.columns)
        if code not in date.columns:
            continue
        if first:
            state_df = date
            first = False
            continue
        # print(date.shape)
        date = date.loc[(date['DATE'] >= 1965) & (date['DATE'] < 2009)]
        date = date[date[code].notna()]
        # print(date['FZF0'].isna().sum())
        date.reset_index(inplace=True)
        if date.empty:
            continue
        # print(date['NAME'])
        # date = date[data['DATE'] < 2009]
        state = re.search(' [A-Z][A-Z] US', date['NAME'][0])[0][1:3]
        if state == 'AH':
            print(date['NAME'][0])
        # print(date['NAME'][0])
        # print(state)
        if state == prev_state:
            df_list.append(date[['NAME', 'DATE', code]])
            # print(pd.concat([state_df, date], keys=['NAME'], axis=0))
            # print(date.columns)
            # state_df = state_df.merge(date[['NAME', 'DATE', 'FZF0']], on='DATE')  # , 'FZF5'
            prev_state = state
        else:
            for j in df_list:
                # print(state_df.shape)
                state_df = pd.concat([state_df, j])
                # print(state_df.shape)
            state_df = state_df[['DATE', code]].groupby(by='DATE').mean()
            print('- - - --')
            state_df['NAME'] = prev_state
            master_df_list.append(state_df)
            # state_df.to_csv('/Users/chrisdoyle/Desktop/agg_data/' + prev_state + '.csv')
            prev_state = state
            # del df_list
            df_list = []
            df_list.append(date[['NAME', 'DATE', code]])
    files = os.walk('/Users/chrisdoyle/Desktop/agg_data/')
    data = []
    # for f in files:
    #     data += f
    # root = data[0]
    # data = data[2]
    # data = sorted(data)
    final_df = master_df_list[0]
    # print(data[1])
    for k in master_df_list[1:]:
        # if '.DS_Store' in k:
        #     continue
        # if k == '.csv':
        #     continue
        # print(k)
        # new_intake = k
        final_df = pd.concat([final_df, k], axis=0)
        # print(final_df)
    # final_df[['DATE','NAME', code]].to_csv('/Users/chrisdoyle/Desktop/final.csv')
    final_df.reset_index(inplace=True)
    return final_df[['DATE','NAME', code]]


folder = '/Users/chrisdoyle/Desktop/The Big Lebowski (1998) (1080p BluRay x265 HEVC 10bit HDR AAC 7.1 afm72)/'
# code_list = ['TAVG', 'TMIN', 'TMAX']
# out_df = merge(folder, code_list[0])
# out_df.to_csv('/Users/chrisdoyle/Desktop/indicators/' + code_list[0] + '.csv')
# # print(type(out_df))
# for j in code_list[1:]:
#     print(j)
#     out_df = out_df.merge(merge(folder, j), on=['NAME', 'DATE'], how='outer').fillna(0)
# print(out_df.shape)
# out_df.to_csv('/Users/chrisdoyle/Desktop/finalfinal.csv')
# irrigation_intake()


def stitcher():
    y_data = pd.read_csv('yield_data.csv')[['Year', 'State', 'Value']]
    y_data.columns = ['DATE','NAME', 'YIELD']
    snow_data = pd.read_csv('snow-precip_data.csv')
    min_max_data = pd.read_csv('min-max_temp_data.csv')
    freeze_data = pd.read_csv('freeze_data.csv')
    irrigation_data = pd.read_csv('irrigation_data.csv')
    temp_data = pd.read_csv('temp_data.csv')
    fert_data = pd.read_csv('fert.csv')
    drought_data = pd.read_csv('drought.csv')
    drought_data.columns = ['Unnamed: 0','DROUGHT', 'NAME', 'DATE']
    final_data = y_data
    final_data['NAME'] = final_data['NAME'].map(lambda u: u.strip())
    abbv_to_state = us.states.mapping('abbr', 'name')
    # final_data['NAME'] = final_data['NAME'].map(abbv_to_state)
    snow_data['NAME'] = snow_data['NAME'].map(abbv_to_state)
    snow_data['NAME'] = snow_data['NAME'].map(lambda u: u.upper().strip())
    temp_data['NAME'] = temp_data['NAME'].map(abbv_to_state)
    temp_data['NAME'] = temp_data['NAME'].map(lambda u: u.upper().strip())
    min_max_data['NAME'] = min_max_data['NAME'].map(abbv_to_state)
    min_max_data['NAME'] = min_max_data['NAME'].map(lambda u: u.upper())
    freeze_data['NAME'] = freeze_data['NAME'].map(abbv_to_state)
    freeze_data['NAME'] = freeze_data['NAME'].map(lambda u: u.upper().strip())
    irrigation_data['NAME'] = irrigation_data['NAME'].map(lambda u: u.upper().strip())
    fert_data['NAME'] = fert_data['NAME'].map(lambda u: u.upper().strip())
    drought_data['NAME'] = drought_data['NAME'].map(lambda u: u.upper().strip())
    fert_data.to_csv('/Users/chrisdoyle/Desktop/fert.csv')
    final_data = pd.merge(left=final_data, right=snow_data, on=['NAME', 'DATE'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data = pd.merge(left=final_data, right=min_max_data, on=['NAME', 'DATE'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data = pd.merge(left=final_data, right=freeze_data, on=['NAME', 'DATE'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data = pd.merge(left=final_data, right=irrigation_data, on=['DATE', 'NAME'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data = pd.merge(left=final_data, right=fert_data, on=['DATE', 'NAME'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data = pd.merge(left=final_data, right=drought_data, on=['DATE', 'NAME'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data = pd.merge(left=final_data, right=temp_data, on=['DATE', 'NAME'], how='left').drop(columns='Unnamed: 0').fillna(0)
    final_data.sort_values(['NAME', 'DATE'])
    final_data = final_data[(final_data['DATE'] > 1964)]
    final_data.to_csv('/Users/chrisdoyle/Desktop/final_data.csv')





stitcher()

def fert_importer():
    nitro_data = pd.read_excel('nitro_fertilizer.xlsx')
    pot_data = pd.read_excel('pot_fertilizer.xlsx')
    phos_data = pd.read_excel('phos_fertilizer.xlsx')
    fert_data_raw = [nitro_data, pot_data, phos_data]
    fert_data = []
    for i in fert_data_raw:
        i.reset_index(inplace=True)
        fert_data.append(pd.melt(i, id_vars='State', value_vars=i.columns.values.tolist()[2:]))
        # fert_data.append(pd.wide_to_long(i, stubnames='', i=['State'], j='DATE' ))
    fert_df = fert_data[0]
    fert_df.columns = ['NAME', 'DATE', 'NITRO']
    count = 0
    type_list = ['POT', 'PHOS']
    for i in fert_data[1:]:
        i.columns = ['NAME', 'DATE', type_list[count]]
        count += 1
        fert_df = pd.merge(left=fert_df, right=i, on=['NAME', 'DATE'])
    fert_df = fert_df.sort_values(['NAME', 'DATE'])
    fert_df.to_csv('/Users/chrisdoyle/Desktop/fert.csv')