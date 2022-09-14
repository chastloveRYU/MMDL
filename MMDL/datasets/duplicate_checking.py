import os

data_dir_west = r'E:\sea_ice_classification\data\S1\Western Arctic\201910to202009_hdf_ASI_cut_full_info'.replace('\\', '/')
data_dir_east = r'E:\sea_ice_classification\data\S1\Eastern Arctic\201910to202009_sar_asi_charts'.replace('\\', '/')

filelist_west = os.listdir(data_dir_west)
filelist_east = os.listdir(data_dir_east)

duplicate_filelist = list(set(filelist_west) & set(filelist_east))

print(len(duplicate_filelist))