import math
from collections import Counter
import pandas as pd
import numpy as np
import sys


def load_csv(filename):                                                        #Đọc file csv
    dataset = pd.read_csv(filename)                                            #filename  : styled-line.csv ,school_earnings.csv, nz_weather.csv
    return dataset

def minmax_normalize(dataset,prop,Min,Max):                                             #Chức năng: Chuẩn hóa min max
    df = pd.DataFrame()                                                                 #Tham số truyền vào: dataframe ,tên thuộc tính, Min , Max (Min,Max do người dùng nhập) 
    type_new = pd.Series([])                                                            #Trả về: df chứa của 1 thuộc tính 
    if (dataset[prop].dtype == np.float64 or dataset[prop].dtype == np.int64):
        for i in range(len(dataset[prop])):
            type_new[i] = (dataset[prop][i]-Min)/(Max-Min)
        df.insert(0,prop,type_new)
        return df
    else:
        print('Cant execute!')
        Emptydf = pd.DataFrame()
        return Emptydf

def column_means(df_num):
    means = [0 for i in range(len(df_num[0]))]                                                  #Tìm giá trị trung bình của cột 
    for i in range(len(df_num[0])):                                                             #Tham số đầu vào: dataframe của  một cột
        col_values = [row[i] for row in df_num]                                                 #Trả về: giá trị trung bình của cột
        means[i] = sum(col_values) / np.float64(len(df_num))
    return means


def column_stdevs(df_num, means):                                                               #Tìm độ lệch chuẩn của các  thuộc tính
    stdev = [0 for i in range(len(df_num[0]))]                                                  #Tham số đầu vào:   daframe của thuộc tính, giá trị trung bình của thuộc tính đó
    for i in range(len(df_num[0])):                                                             #Trả về: Độ lệch chuẩn của thuộc tính
        variance = [pow(row[i] - means[i], 2) for row in df_num]                                
        stdev[i] = sum(variance)
    stdev = [np.sqrt(x / (np.float64(len(df_num) - 1))) for x in stdev]
    return stdev


def zscore_normalize(dataset):                                                                  #Chuẩn hóa z-score
    df_num = dataset.select_dtypes(include=[np.float])                                          #Tham số đầu vào: Dataframe
    name_col = list(df_num.columns)                                                             #Trả về: dataframe đã đc chuẩn hóa
    data_list = df_num.values.tolist()
    mean = column_means(data_list)
    stdev = column_stdevs(data_list, mean)
    for row in data_list:
        for i in range(len(row)):
            row[i] = (row[i] - mean[i]) / stdev[i]
    num_data = pd.DataFrame(data_list)
    num_data.columns = name_col
    nonnum_data = dataset.select_dtypes(exclude=[np.number])
    result = pd.concat([nonnum_data, num_data], axis=1, sort=False)
    return result


def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):           
        bins.append((low, low + width))
    return bins


def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


def equal_width(dataset, prop, n):                                                                  #Rời rạc hóa dữ liệu bằng phương pháp chia giỏ theo độ rộng trên danh sách thuộc tính
    if (dataset[prop].dtype == np.float64 or dataset[prop].dtype == np.int64):                      #Tham số đầu vào: dataframe,tên thuộc tính,number do người dùng nhập
        lower = min(dataset[prop])                                                                  #Trả về: dataframe của thuộc tính đã được chuẩn hóa, nếu không thực hiện đc thì trả về df rỗng
        upper = max(dataset[prop])
        bins = create_bins(round(lower), round((upper - lower)/n), n)
        binned = []
        for value in dataset[prop]:
            bin_index = find_bin(value, bins)
            binned.append((value, bins[bin_index]))
        df = pd.DataFrame(binned)
        df.columns = [prop, 'bin']
        return df
    else:
        print('Cant execute!')
        Emptydf = pd.DataFrame()
        return Emptydf

def create_bins1(dataset,prop,n):
    bins = []
    df = dataset
    df_list = df[prop].tolist()
    df_list.sort()
    lower = df_list[0]
    depth = round(len(df_list)/n)
    count = 1
    for i in range(0, len(df_list)):
        if i % depth == 0 and count < n and i != 0:
            bins.append((lower, df_list[i]))
            lower = df_list[i]
            count = count + 1
        if count == n:
            bins.append((lower, df_list[len(df_list)-1]))
            break
    return bins

def removeData(dataset,prop):                                   #xóa các dòng bị thiếu dữ liêu của thuộc  tính
    index = []                                                  #Tham số: df,tên thuộc tính
    na_rows = dataset[dataset[prop].isnull()]                   #Trả về: Df đã được chuẩn hóa của thuộc tính
    for j in range(len(na_rows)):
            index.append(na_rows.index[j])
    index = list(dict.fromkeys(index))
    for i in range(len(index)):
         dataset = dataset.drop(index[i])
    return dataset

def equal_depth(dataset, prop, n):
    if (dataset[prop].dtype == np.float64 or dataset[prop].dtype == np.int64):          #Rời rạc hóa dữ liệu bằng phương pháp chia giỏ theo độ sâu trên danh sách thuộc tính
        bins = create_bins1(dataset, prop, n)                                           #Tham số:  df,tên thuộc tính,number do người dùng nhập
        binned = []                                                                     #Kết quả : df đã đc rời rạc 
        for value in dataset[prop]:
            bin_index = find_bin(value, bins)
            binned.append((value, bins[bin_index]))
        df = pd.DataFrame(binned)
        df.columns = [prop, 'bin']
        return df
    else:
        print('Cant execute!')
        Emptydf = pd.DataFrame()
        return Emptydf


def mode_column(dataset, prop):
    data = Counter(dataset[prop])
    data.most_common()  # Returns all unique items and their counts                                     #Tìm data có tần số xuất hiện cao nhất của thuộc tính
    get_mode = data.most_common(2)                                                                      #Tham số:df,tên thuộc tính
    if math.isnan(get_mode[0][0]):                                                                      #trả về các mode 
        return get_mode[1][0]
    else:
        return get_mode[0][0]


def fill_missingvalue(dataset, prop):
    df = dataset
    if (df[prop].dtype == np.float64 or df[prop].dtype == np.int64):                            #Điền vào các giá trị thiếu của dữ liệu nếu là thuộc tính liên tục  thì mean nếu là thuộc tính rời rác thì điền mode
        mean = np.nanmean(df[prop], axis = 0)                                                   #Tham số: df,tên thuộc tính
        for i in range(0, len(df[prop])):
            if math.isnan(df[prop][i]):                                                         #Trả về: df đã điền  các dữ liệu bị thiếu
                df.at[i,prop]= mean
    else:
        mode = mode_column(dataset,prop)
        for i in range(0, len(df[prop])):
            if pd.isnull(df[prop][i]):
                df.at[i,prop] = mode
    return df[prop]


def action(i,dataset):
    if(i == "min-max" ):
        Str_d = getpropList()
        List_d = splitstring(Str_d)                         # Đã sửa ghi  trực tiếp lên out.csv
        MinMax = getMinMax()                                #24_B1.py school_earnings.csv out.csv min-max 10-20 60-110 14-165 propList{School,Men,Women}                                                                
        MyEmptydf_d = pd.DataFrame()                        # Đã sửa cho phép nhập min max như trên nếu dữ liệu != float or int thì ko thực hiện
        for i in range(len(List_d)):                            
            MyEmptydf_d = pd.concat([MyEmptydf_d,minmax_normalize(dataset,List_d[i],int(MinMax[i][0]),int(MinMax[i][1]))], axis=1, sort=False)
        MyEmptydf_d.to_csv("out.csv", index=None)      
    elif(i  == "z-score"):                                   #Đã sửa ghi  trực tiếp lên out.csv
        Str_d = getpropList()                                #Chức năng: thực hiện các chức năng dựa trên tham số dòng lệnh đã nhập (argv[3])  và lưu vào csv
        List_d = splitstring(Str_d)                          #Tham số: lệnh cần  thực hiện,df 
        df = zscore_normalize(dataset)                       #Trả về:Không
        df[List_d].to_csv("out.csv",index = None)            #Ví  dụ chuẩn Hóa minmax: 24_B1.py school_earnings.csv out.csv min-max 10-20 60-110 14-165 propList{School,Men,Women}     
    elif(i == "equal-width"):                                #Chuẩn hóa z-score: 24_B1.py styled-line.csv out.csv z-score propList{Low 2007,High 2014} 
        bin_w = getBin()                                     #Rời rạc hóa dữ liệu theo chiều rộng: 24_B1.py school_earnings.csv out.csv equal-width bin-5 propList{School,Men,Women} 
        Str_w = getpropList()                                #Rời rạc hóa dữ liệu theo chiều sâu: 24_B1.py school_earnings.csv out.csv equal-depth bin-5 propList{School,Men,Women} 
        List_w = splitstring(Str_w)                          #Xóa các dòng bị thiếu dữ liệu: 24_B1.py school_earnings.csv out.csv remove propList{Men} (kết quả ghi lại  là df trong đó các dòng dữ liệu bị thiếu của thuộc tính 'Men' đã bị xóa còn lại giữ nguyên)
        MyEmptydf_w = pd.DataFrame()                         #Điền dữ liệu bị thiếu: 24_B1.py school_earnings.csv out.csv fill-missingvalue propList{Men,Gap}
        for i in range(len(List_w)):
            MyEmptydf_w = pd.concat([MyEmptydf_w, equal_width(dataset, List_w[i], int(bin_w[1]))], axis=1, sort=False)
        MyEmptydf_w.to_csv("out.csv", index=None)                      # thiếu tên cột để dọc ngược lên lại
    elif(i == "equal-depth"):
        bin_d = getBin()
        Str_d = getpropList()
        List_d = splitstring(Str_d)
        MyEmptydf_d = pd.DataFrame()
        for i in range(len(List_d)):
            MyEmptydf_d = pd.concat([MyEmptydf_d, equal_depth(dataset, List_d[i], int(bin_d[1]))], axis=1, sort=False)
        MyEmptydf_d.to_csv("out.csv", index=None)                       # thiếu tên cột để dọc ngược lên lại
    elif(i == "remove"):
        Str = getpropList()
        List = splitstring(Str)
        if(len(List) == 0):
            df = pd.read_csv(sys.argv[1])
            List  = df.columns.tolist()
        for i in range(len(List)):
            dataset = removeData(dataset,List[i])
        dataset.to_csv("out.csv", index=None)
    elif(i == "fill-missingvalue"):
        Str_d = getpropList()
        List_d = splitstring(Str_d)
        MyEmptydf = pd.DataFrame()
        for i in range(len(List_d)):
            MyEmptydf = pd.concat([MyEmptydf, fill_missingvalue(dataset,List_d[i])], axis=1, sort=False)
        MyEmptydf.to_csv("out.csv", index=None)
    else:
        print("Khong thuc hien hanh dong nao")

def getpropList():                                                              #Lấy ra danh sách các thuộc  tính
    i  = 0                                                                      #trả về chuỗi propList(....)
    flag = 0
    while(i < len(sys.argv)):
        temp = sys.argv[i].split("{")
        if(temp[0] == "propList"):
           flag = i
           i+=1
        else: i+=1
    if(flag == 0):
        return sys.argv[i-1]
    if(flag == len(sys.argv)):
        return sys.argv[flag]
    else:
        for j in range(flag+1,i):
            sys.argv[flag]+= " "+ sys.argv[j]
        return sys.argv[flag]


def checkProplist(Str):                                         #Kiểm tra xem có phải chuỗi propList(....) hay ko
    temp = Str.split("{")
    if(temp[0] == "propList"):
        return True
    else: return False


def splitstring(Str):
    if(checkProplist(Str) == True):                             #Tách chuỗi propList(...) để đưa các thuộc tính bên trong thành list
        temp = Str.split("{")                                   #trả về list danh sách thuộc tính đã nhập
        temp[1] = temp[1].split("}")
        tempStr = temp[1][0].split(",")
        return tempStr
    else: return {}


def getBin():
    i = 0
    while(i<len(sys.argv)):
        temp = sys.argv[i].split('-')          # qui định cú pháp      bin-number
        if(temp[0] == "bin"):                  
            return temp                        #Lấy ra list(bin,number)
        else:i+=1                              
    return {}

def getMinMax():
    i = 4                           # vị trí action min-max = argv[3]
    MinMax = list()
    while(sys.argv[i].find("propList{")==-1):
        Str = sys.argv[i].split("-")
        MinMax.append([Str[0],Str[1]])
        i+=1
    return MinMax

dataset = load_csv(sys.argv[1])
action(sys.argv[3],dataset)
print("Xem kết quả tại file output đã nhập")