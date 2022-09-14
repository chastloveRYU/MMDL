from osgeo import gdal
import netCDF4 as nc
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import os
from matplotlib import colors
from dbfread import DBF


def read_icetype(path):

    fields = ['CT', 'CA', 'CB', 'CC', 'SA', 'SB', 'SC', 'POLY_TYPE']
    for field in fields:
        if field == 'POLY_TYPE':
            # land mask
            _, _, globals()[field], value, record = read_tif(os.path.join(path, field, field + '.tif'), field)
            lib = {x: y for x, y in zip(record, value)}
        else:
            lon, lat, globals()[field] = read_tif(os.path.join(path, field, field + '.tif'), field)

    # plt.figure()
    # plt.imshow(SA)
    # plt.show()

    # 确定主海冰类型：CA/CB/CC最大值对应的海冰类型SA/SB/SC
    sic_parts = np.dstack((CA, CB, CC))
    icetype = np.dstack((SA, SB, SC))
    # print(value)
    # print(record)
    A, B = np.meshgrid(np.arange(sic_parts.shape[1]), np.arange(sic_parts.shape[0]))
    ind = (B, A, np.argmax(sic_parts, 2))
    icetype = icetype[ind]

    # try:
    #     icetype[POLY_TYPE == lib['L']] = 120
    #     icetype[POLY_TYPE == lib['W']] = 55
    # except:
    #     print('The sea ice type in charts is either sea ice or open water!')
    # Land Mask
    icetype[POLY_TYPE == lib['L']] = 120

    icetype[CT < 10] = 55
    CT[POLY_TYPE == lib['L']] = 120

    return lon, lat, icetype, CT


def read_tif(filename, field):
    # read ASI SIC
    data = gdal.Open(filename)
    row = data.RasterYSize
    col = data.RasterXSize
    bands = data.RasterCount
    sic = data.ReadAsArray(0, 0, col, row)
    sic = sic.astype(np.float32)
    GeoTransform = data.GetGeoTransform()  # 投影转换信息
    Projection = data.GetProjection()  # 投影信息
    pixel, line = range(0, col), range(0, row)
    pixel, line = np.meshgrid(pixel, line)
    lon = GeoTransform[0] + pixel * GeoTransform[1] + line * GeoTransform[2]
    lat = GeoTransform[3] + pixel * GeoTransform[4] + line * GeoTransform[5]

    # 读取数据表
    dbf = filename + '.vat.dbf'
    value, record = read_dbf(dbf, field)

    # 有些值value没有记录，找出来赋为nan
    nan_values = list(set(sic.flatten().tolist()) ^ set(value))
    for nan_value in nan_values:
        sic[sic == nan_value] = np.nan

    # print(nan_value)
    # print(type(nan_value))

    if field == "POLY_TYPE":
        # sic[sic == 0] = np.nan
        return lon, lat, sic, value, record

    else:
        for i in range(len(value)):
            sic[sic == value[i]] = record[i]
        # sic[sic == 15] = np.nan
        return lon, lat, sic


def read_dbf(dbf, field):
    # 读取数据表
    table = DBF(dbf, load=True)
    value = []
    CT = []
    for record in table.records:
        value.append(record['Value'])
        try:
            if field != 'POLY_TYPE':
                CT.append(int(record[field]))
            else:
                CT.append(str(record[field]))
        except:
            CT.append(np.nan)
    # print(value)
    # print(CT)

    return value, CT



if __name__ == "__main__":

    path = r'E:\sea_ice_classification\data\charts_gridded\Western Arctic\2019'.replace('\\', '/')
    filelist = os.listdir(path)
    filelist = [os.path.join(path, file, 'CT', 'CT.tif.vat.dbf') for file in filelist]
    print(filelist)

    code = []
    for file in filelist:
        value, record = read_dbf(file, 'CT')
        code.extend(record)

    code = list(set(code))
    code = [x for x in code if not np.isnan(x)]
    code.sort()
    print(code)
        # colorlist=[(153/255,205/255,242/255,1),
        #            (70/255,102/255,245/255,1),
        #
        #            (11/255,93/255,140/255,1),
        #
        #            (24/255,155/255,13/255,1),
        #
        #            (170/255,226/255,67/255,1),
        #
        #            (252/255,242/255,4/255,1),
        #
        #            (255/255,165/255,79/255,1),
        #
        #            (237/255,60/255,0/255,1),
        #            (225/255,13/255,28/255,1),
        #            (178/255,4/255,0/255,1)]
        # cmap = colors.ListedColormap(colorlist)

        # plt.figure()
        # plt.imshow(sic)
        # plt.colorbar()
        # plt.show()
