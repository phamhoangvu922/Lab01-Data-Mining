import pandas as pd
import numpy as np
from collections import namedtuple
import sys

def load_txt(filename):                                                                                                         #Đọc dữ liệu từ file txt vào df và ghi lại vao file test1.csv
    Item = namedtuple('Item', ['country', 'name', 'longName', 'foundingDate', 'population', 'capital', 'largestCity',           #Tham số: file txx
                               'area'])                                                                                        
    items = []
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
        country = np.nan
        name = np.nan
        longName = np.nan
        foundingDate = np.nan
        population = np.nan
        capital = np.nan
        largestCity = np.nan
        area = np.nan
        for i in range(8, len(lines)):
            line = lines[i]
            l = line.rstrip('\n')
            if l.startswith("country="):
                country = l.strip('country=')
            elif l.startswith("name="):
                name = l.strip('name=')
            elif l.startswith("longName="):
                longName = l.strip('longName=')
            elif l.startswith("foundingDate="):
                foundingDate = l.strip('foundingDate=')
            elif l.startswith("population="):
                population = l.strip('population=')
            elif l.startswith("capital="):
                capital = l.strip('capital=')
            elif l.startswith("largestCity="):
                largestCity = l.strip('largestCity=')
            else:
                area = l.strip('area=')
            if i<len(lines)-1:
                nextline = lines[i + 1]
            else:
                nextline = line
            if nextline.startswith("country=") or nextline==line:
                items.append(Item(country, name, longName, foundingDate, population, capital, largestCity, area))
                country = np.nan
                name = np.nan
                longName = np.nan
                foundingDate = np.nan
                population = np.nan
                capital = np.nan
                largestCity = np.nan
                area = np.nan

    df = pd.DataFrame.from_records(items, columns=['country', 'name', 'longName', 'foundingDate', 'population',
                                                   'capital', 'largestCity', 'area'])
    df.to_csv('test1.csv',index = None)
def formatArea(dataset):                                                                                #format  lại dữ liệu có dạng "or NUMBERmi" thành dạng chuẩn  
    List = list(dataset["area"])                                                                        #Tham số :df
    for i in range(len(List)):                                                                          #Trả về: df có dạng dữ liệu chuẩn
        if(List[i].find("or ") != -1 ):
            Str  = List[i].split("or ") 
            Str[1] = Str[1].replace(",",".")
            dataset = dataset.replace(to_replace = List[i], value = Str[1]) 
    return dataset

def changeUnit(dataset):                                                                        
    dataset = formatArea(dataset)
    List = list(dataset["area"])
    for i in range(len(List)):
        if(List[i].find("mi")!=-1):                    # đổi từ mi : m^2 sang km : km^2  
            Str = List[i].split("mi")                  #Tham số: df
            number =  float(Str[0])/1000000            #Trả về: df đã được đổi đơn vị
            Str2 = str(number)+"km"
            dataset = dataset.replace(to_replace = List[i], value = Str2) 
    return dataset
def removeData(dataset,prop):                                   # xóa 1 đối tượng chỉ định giống như hàm 24_B1.py
    index = []
    na_rows = dataset[dataset[prop].isnull()]
    for j in range(len(na_rows)):
            index.append(na_rows.index[j])
    index = list(dict.fromkeys(index))
    for i in range(len(index)):
         dataset = dataset.drop(index[i])
    return dataset
def drop_duplicaterows(df):                                                             #Xóa  dữ liệu bị trùng (khác mã coutry còn  lại giống)
    prop = 'check'                                                                      #Tham số: df
    df[prop] = 0                                                                        #Ghi  kết quả vào file người dùng nhập (argv[2])
    Index = df.index
    for a in range(0, len(Index)):
        if df.at[Index[a],prop] == 1:
            df = df.drop(Index[a])
        else:
            row1 = df.loc[Index[a], ['name', 'longName', 'foundingDate', 'population', 'capital', 'largestCity', 'area']]
            for b in range(a+1, len(Index)):
                if df.at[Index[b], prop] == 1:
                    continue
                else:
                    row2 = df.loc[
                        Index[b], ['name', 'longName', 'foundingDate', 'population', 'capital', 'largestCity', 'area']]
                    if row1.equals(row2):
                        df.at[Index[b], prop] = 1
    df = df.drop(columns = 'check')
    df.to_csv(sys.argv[2],index = None)   
def remove_empty_rows(df):                                                                                  #Xóa các mẫu chỉ có mã country
    for row in df.iterrows():                                                                               #Tham số : df
        i = df[df['longName'].isnull() & df['foundingDate'].isnull() & df['population'].isnull()            #Trả  về:df
               & df['capital'].isnull() & df['largestCity'].isnull() & df['area'].isnull()].index
        df = df.drop(i)
    return df


load_txt(sys.argv[1])           # load file countries.txt vào  dataframe và dữ liệu ghi  vào file  test1.csv
df = pd.read_csv("test1.csv")
df = removeData(df,"area")                                 #remove dữ liệu thiếu diện tích      
df = changeUnit(df)                                        # đổi đơn vị từ m^2 -> km^2
df = remove_empty_rows(df)                                 #xóa dữ liệu chỉ có mã coutry        
drop_duplicaterows(df)                                    #Xóa dữ liệu trùng         # chạy trong khoảng 3 phút và ghi kết quả ra output.csv
print("Xem kết quả ở file output đã nhập")

