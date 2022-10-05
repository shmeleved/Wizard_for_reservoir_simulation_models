from itertools import chain
import ecl_data_io as eclio
import pandas as pd
import os
import numpy as np
import codecs
path_to_model = ''
model_name = ''
path_to_fip_xlx = ''
path_cps_export = ''
path_reports_export = ''

def dimens(): # Размерность модели
    file_elements = []
    with codecs.open(path_to_model + '/' + model_name + '.DATA', 'r',"utf_8_sig") as my_file:
        for line in my_file:
            if 'DIMENS' in line:
                for inner_line in my_file:
                    inner_line = inner_line.split()
                    if "/" in inner_line:
                        inner_line.remove("/")
                        file_elements.append(inner_line)
                        file_elements = list(chain.from_iterable(file_elements))
                        break
                    file_elements.append(inner_line)
    return file_elements

def ts_reading(): # Шаги в модели
    dict_steps = {}
    with open(path_to_model + "/" + 'RESULTS/' + model_name + '/' + 'outsol_ts_maps.meta', 'r') as my_file:
        for line in my_file:
            if '[timesteps]' in line:
                for inner_line in my_file:
                    if len(inner_line) == 1:
                        break
                    inner_line = inner_line.strip(' ')  # Удаляем пробел с двух сторон
                    inner_line = inner_line.strip('\n')  # Удаляем перенос с двух сторон
                    inner_line = inner_line.split(' = ')  # Разделение данных
                    try:
                        while True:
                            inner_line.remove('')  # Удаление лишних пробелов
                    except ValueError:
                        abc = 0
                    dict_steps[inner_line[0]] = inner_line[1]  # Заполнение словаря дата:расчетный шаг
    return dict_steps

def unpacking(path, property): # Распаковка файлов tNav
    file_elements = []
    with open(path, 'r') as my_file:
        for line in my_file:
            if property in line:
                for inner_line in my_file:
                    if inner_line.split()[0]=='--':
                        continue
                    inner_line = inner_line.split()
                    if "/" in inner_line:
                        inner_line.remove("/")
                        file_elements.append(inner_line)
                        file_elements = sum(file_elements, [])
                        break
                    file_elements.append(inner_line)
    unpacked = []
    for i in file_elements:
        if "*" in i:
            i = i.split('*')
            i[0] = int(i[0])
            i[1] = float(i[1])
            for item in range(i[0]):
                unpacked.append(i[1])
        else:
            unpacked.append(float(i))
    return unpacked

def init_reading(property): # Чтение INIT файла
    list_name=0
    for item in eclio.lazy_read(path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + '.INIT'):
        if property in item.read_keyword():
            list_name = pd.Series(item.read_array().astype(float))
    return list_name

def x_files_list():
    x_files = []
    for root, dirs, files in os.walk(path_to_model + "/" + 'RESULTS/' + model_name):
        for filename in files:
            if model_name+'.X' in filename:
                x_files.append(filename[-4:])
    return sorted(x_files)

def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k

def unrst_reading(property, time_step): # Чтение unrst файла
    list_name = []
    dict_steps = ts_reading()
    if int(dict_steps[time_step]) < 10:
        format = '.X000' + str(dict_steps[time_step])
    elif 10 <= int(dict_steps[time_step]) < 100:
        format = '.X00' + str(dict_steps[time_step])
    elif 100 <= int(dict_steps[time_step]) < 1000:
        format = '.X0' + str(dict_steps[time_step])
    elif 1000 <= int(dict_steps[time_step]) < 10000:
        format = '.X' + str(dict_steps[time_step])
    for item in eclio.lazy_read(path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + format):
        if property in item.read_keyword():
            # list_name = item.read_array().astype(float)
            list_name = item.read_array()
    return list_name

def check_cache(cache):
    check_ijk_file = os.path.exists(cache)
    return check_ijk_file

def clear_cache():
    mydir = 'cached/'
    filelist = [f for f in os.listdir(mydir)]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

def read_ijk():
    df_ijk = pd.DataFrame()
    if not check_cache('cached/'+'df_ijk_big'+'_'+model_name):
        actnum = []
        for item in eclio.lazy_read(path_to_model + '/RESULTS/' + model_name + '/' + model_name + '.EGRID'):
            if 'ACTNUM' in item.read_keyword():
                actnum = item.read_array()

        df_ijk['actnum'] = actnum

        i_max = int(dimens()[0])
        j_max = int(dimens()[1])
        k_max = int(dimens()[2])

        coor_i = np.tile(np.arange(1, i_max + 1), j_max * k_max)
        coor_j = np.tile(np.repeat(np.arange(1, j_max + 1), i_max), k_max)
        coor_k = np.repeat(np.arange(1, k_max + 1), i_max * j_max)

        df_ijk['i'] = coor_i
        df_ijk['j'] = coor_j
        df_ijk['k'] = coor_k

        coor_i_cps = coor_i[coor_k == 1]
        coor_j_cps = coor_j[coor_k == 1]
        df_ijk_cps = pd.DataFrame()
        df_ijk_cps['i'] = coor_i_cps
        df_ijk_cps['j'] = coor_j_cps

        df_ijk = df_ijk.query('actnum != 0')

        df_ijk.to_pickle('cached/'+'df_ijk_big' + '_' + model_name)
        df_ijk_cps.to_pickle('cached/'+'df_ijk_cps' + '_' + model_name)
    else:
        df_ijk = pd.read_pickle('cached/'+'df_ijk_big' + '_' + model_name)
        df_ijk_cps = pd.read_pickle('cached/'+'df_ijk_cps' + '_' + model_name)
    return df_ijk, df_ijk_cps

def get_static_property():
    property_list = []
    for item in eclio.lazy_read(path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + '.INIT'):
        property_list.append(item.read_keyword())
    return property_list

def get_dynamic_property():
    ts_number = int(x_files_list()[0])
    format = 0
    if ts_number < 10:
        format = '.X000' + str(ts_number)
    elif 10 <= ts_number < 100:
        format = '.X00' + str(ts_number)
    elif 100 <= ts_number < 1000:
        format = '.X0' + str(ts_number)
    elif 1000 <= int(ts_number) < 10000:
        format = '.X' + str(ts_number)
    property_list = []
    for item in eclio.lazy_read(path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + format):
        property_list.append(item.read_keyword())
    return property_list

def get_fip_dict(path_to_fip_xlx):
    df = pd.read_excel(path_to_fip_xlx, engine='openpyxl')
    unique_fip = df['object'].unique()
    dict_fip = {}
    for i in unique_fip:
        dict_fip[i] = list(df.query('object == @i')['region'])
    return dict_fip

def get_dznet():
    for item in eclio.lazy_read(
            path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + '.INIT'):
        if 'DZ' in item.read_keyword():
            dz = item.read_array().astype(float)
        elif 'NTG' in item.read_keyword():
            ntg = item.read_array().astype(float)
    return dz * ntg

def get_volume():
    for item in eclio.lazy_read(
            path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + '.INIT'):
        if 'DX' in item.read_keyword():
            dx = item.read_array().astype(float)
        elif 'DY' in item.read_keyword():
            dy = item.read_array().astype(float)
        elif 'DZ' in item.read_keyword():
            dz = item.read_array().astype(float)
    return pd.Series(dx * dy * dz)

def export_cps(export_combo,grid,property,
               df_cps,time,fip,df_ijk_all,region_text,method):
    max_i = int(dimens()[0])
    max_j = int(dimens()[1])

    x_depart = init_reading('DX')[0]
    y_depart = init_reading('DY')[0]

    x = []
    y = []
    list_name = 0
    for item in eclio.lazy_read(path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + '.EGRID'):
        if 'COORD' in item.read_keyword():
            list_name = item.read_array()

    x.append(list_name[0])  # min_x
    x.append(list_name[-3])  # max_x

    y.append(list_name[1] * (-1))  # max_y
    y.append(list_name[-2] * (-1))  # min_y

    if not os.path.isdir(path_cps_export):
        os.mkdir(path_cps_export)

    if not os.path.isdir(path_cps_export+'/01.ALL/'):
        os.mkdir(path_cps_export+'/01.ALL/')

    if not os.path.isdir(path_cps_export+'/02.Separated/'):
        os.mkdir(path_cps_export+'/02.Separated/')

    if export_combo == 'Раздельно':
        max_val = grid[property].astype(float).max()
        min_val = grid[property].astype(float).min()

        df_cps = df_cps.merge(grid, on=['i', 'j'], how='outer')
        df_cps[property] = df_cps[property].fillna('1e30')

        f = open(path_cps_export + '/02.Separated/' +
            time + '_' + fip + '_' + method + '_' + property + '.cps',
            'a')
        row_len = max_j
        frm = 0
        to = max_j
        with f as my_file:
            my_file.write('FSASCI 0 1 COMPUTED 0 1.000000e+30')
            my_file.write('\n')
            my_file.write('FSATTR 0 0')
            my_file.write('\n')
            my_file.write(
                'FSLIMI' + ' ' + ' ' + str(min(x)+x_depart/2) + ' ' + str(max(x)-x_depart/2) + ' ' + str(
                    min(y)+y_depart/2) + ' ' + str(max(y)-y_depart/2) + ' ' + str(min_val) + ' ' + str(max_val))
            my_file.write('\n')
            my_file.write('FSNROW' + ' ' + str(max_j) + ' ' + str(max_i))
            my_file.write('\n')
            my_file.write('FSXINC' + ' ' + str(x_depart) + ' ' + str(y_depart))
            my_file.write('\n')
            my_file.write('->Surface')
            my_file.write('\n')
            for i in range(int(len(df_cps) / row_len)):
                count_columns = 0
                for j in range(frm, to):
                    count_columns += 1
                    my_file.write(str(df_cps[property][j]))
                    my_file.write(' ')
                    if count_columns == 5:
                        my_file.write('\n')
                        count_columns = 0
                my_file.write('\n')
                frm += row_len
                to += row_len
    elif export_combo == 'Все регионы':
        if path_to_fip_xlx is not None:
            dict_fip = get_fip_dict(path_to_fip_xlx)
            for object in dict_fip.keys():
                fip = dict_fip[object]
                df_ijk_all_export = df_ijk_all.query('region in @fip')
                df_ijk_all_export = df_ijk_all_export.groupby(['i', 'j'])[
                    property].mean().reset_index()

                max_val = df_ijk_all_export[property].astype(float).max()
                min_val = df_ijk_all_export[property].astype(float).min()

                df_cps_all = df_cps.merge(df_ijk_all_export, on=['i', 'j'], how='outer')
                df_cps_all[property] = df_cps_all[
                    property].fillna(
                    '1e30')
                f = open(path_cps_export + '/01.ALL/' +
                         time + '_' + object + '_' + method + '_' + property + '.cps',
                         'a')
                row_len = max_j
                frm = 0
                to = max_j
                with f as my_file:
                    my_file.write('FSASCI 0 1 COMPUTED 0 1.000000e+30')
                    my_file.write('\n')
                    my_file.write('FSATTR 0 0')
                    my_file.write('\n')
                    my_file.write(
                        'FSLIMI' + ' ' + ' ' + str(min(x) + x_depart / 2) + ' ' + str(
                            max(x) - x_depart / 2) + ' ' + str(
                            min(y) + y_depart / 2) + ' ' + str(max(y) - y_depart / 2) + ' ' + str(min_val) + ' ' + str(
                            max_val))
                    my_file.write('\n')
                    my_file.write('FSNROW' + ' ' + str(max_j) + ' ' + str(max_i))
                    my_file.write('\n')
                    my_file.write('FSXINC' + ' ' + str(x_depart) + ' ' + str(y_depart))
                    my_file.write('\n')
                    my_file.write('->Surface')
                    my_file.write('\n')
                    for i in range(int(len(df_cps_all) / row_len)):
                        count_columns = 0
                        for j in range(frm, to):
                            count_columns += 1
                            my_file.write(str(df_cps_all[property][j]))
                            my_file.write(' ')
                            if count_columns == 5:
                                my_file.write('\n')
                                count_columns = 0
                        my_file.write('\n')
                        frm += row_len
                        to += row_len
        elif path_to_fip_xlx is None:
            fip = []
            for item in eclio.lazy_read(path_to_model + "/" + 'RESULTS/' + model_name + '/' + model_name + '.INIT'):
                if region_text in item.read_keyword():
                    fip.append(list(set(item.read_array())))
            for object in fip[0]:
                object = int(object)
                df_ijk_all_export = df_ijk_all.query('region == @object')
                df_ijk_all_export = df_ijk_all_export.groupby(['i', 'j'])[
                    property].mean().reset_index()

                max_val = df_ijk_all_export[property].astype(float).max()
                min_val = df_ijk_all_export[property].astype(float).min()

                df_cps_all = df_cps.merge(df_ijk_all_export, on=['i', 'j'], how='outer')
                df_cps_all[property] = df_cps_all[
                    property].fillna('1e30')
                f = open(path_cps_export + '/01.ALL/' + time + '_' + str(object) + '_' + method + '_' + property + '.cps',
                         'a')
                row_len = max_j
                frm = 0
                to = max_j
                with f as my_file:
                    my_file.write('FSASCI 0 1 COMPUTED 0 1.000000e+30')
                    my_file.write('\n')
                    my_file.write('FSATTR 0 0')
                    my_file.write('\n')
                    my_file.write(
                        'FSLIMI' + ' ' + ' ' + str(min(x) + x_depart / 2) + ' ' + str(
                            max(x) - x_depart / 2) + ' ' + str(
                            min(y) + y_depart / 2) + ' ' + str(max(y) - y_depart / 2) + ' ' + str(min_val) + ' ' + str(
                            max_val))
                    my_file.write('\n')
                    my_file.write('FSNROW' + ' ' + str(max_j) + ' ' + str(max_i))
                    my_file.write('\n')
                    my_file.write('FSXINC' + ' ' + str(x_depart) + ' ' + str(y_depart))
                    my_file.write('\n')
                    my_file.write('->Surface')
                    my_file.write('\n')
                    for i in range(int(len(df_cps_all) / row_len)):
                        count_columns = 0
                        for j in range(frm, to):
                            count_columns += 1
                            my_file.write(str(df_cps_all[property][j]))
                            my_file.write(' ')
                            if count_columns == 5:
                                my_file.write('\n')
                                count_columns = 0
                        my_file.write('\n')
                        frm += row_len
                        to += row_len