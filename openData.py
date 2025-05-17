import os
import re
import pandas as pd
from datetime import datetime


class OpenData(object):

    def __init__(self, path) -> None:
        self.path = path

    def check_filename_format(self, filename):
        # Регулярное выражение для проверки формата
        pattern = re.compile(r'^[A-Za-z0-9]+_[A-Za-z0-9.]+_\d{2}\.\d{2}\.\d{4}\.xlsx$')

        # Проверка соответствия формату
        if pattern.match(filename):
            return True
        else:
            return False

    def readFiles(self):
        files = {}
        for file in os.listdir(self.path):
            if self.check_filename_format(file):
                files[file] = self.path+'\\'+file

        return files
    

class MergeData(object):

    def __init__(self, files):
        self.file = files

    def openDataFrames(self, files):
        self.dataframes = {}
        for key, value in files.items():
            self.dataframes[key] = pd.read_excel(value)
            self.dataframes[key].columns =  self.dataframes[key].columns.astype(str)

    def parseDataColumns(self):
        for key, value in self.dataframes.items():
            new_cols = {}
            for col in value.columns:
                if '-tetra' in col:
                    new_col = col.replace('tetra', 'Gs')
                    if '-Fc' in col:
                        new_col = new_col.replace('-Fc', '')
                    new_cols[col] = new_col
            if len(new_cols) > 0:
                self.dataframes[key] = value.rename(columns = new_cols)

        for key, value in self.dataframes.items():
            new_cols = {}
            for col in value.columns:
                if '-bis' in col:
                    new_col = col.replace('-bis', '')
                    new_cols[col] = new_col
            if len(new_cols) > 0:
                self.dataframes[key] = value.rename(columns = new_cols)

        for key, value in self.dataframes.items():
            new_cols = {}
            for col in value.columns:
                if 'Gs' not in col:
                    if '58-57' in col:
                        new_col = col.replace('58-57', '57-58')
                        new_cols[col] = new_col
                    if '57-21' in col:
                        new_col = col.replace('57-21', '21-57')
                        new_cols[col] = new_col
                    if '57-7' in col:
                        new_col = col.replace('57-7', '7-57')
                        new_cols[col] = new_col
                    if '21-7' in col:
                        new_col = col.replace('21-7', '7-21')
                        new_cols[col] = new_col
                    if '58-7' in col:
                        new_col = col.replace('58-7', '7-58')
                        new_cols[col] = new_col
                    if '58-21' in col:
                        new_col = col.replace('58-21', '21-58')
                        new_cols[col] = new_col
            if len(new_cols) > 0:
                self.dataframes[key] = value.rename(columns = new_cols)
    
    def parseFileNames(self):
        self.names = {}
        for key in self.dataframes.keys():
            self.names[key] = key.replace('BsAb_', '').replace('mono_', '').replace('homo_', '').replace('_', ' ').replace('.xlsx', '')

    def process_dict(self, data):
        df = pd.DataFrame(columns=['serum', 'date'])

        for key, value in data.items():
            parts = value.split(' ')

            date = parts[-1]

            virus = ' '.join(parts[:-1])

            if ' II' in virus:
                virus = virus.replace(' II', '')

            if ' I' in virus:
                virus = virus.replace(' I', '')

            df.loc[key] = [virus, date]

        return df
    
    def parseMultipleDfs(self, dfs):
        parsedDfs = []
        for name, df in dfs.items():
            for col in df.columns:
                if "." in col:
                    spl_col = col.split(".")[0]
                    df[spl_col] = (df[spl_col] + df[col]) / 2
                    df = df.drop(columns = [col])
            parsedDfs.append(self.parseDf(df, name))

        return pd.concat(parsedDfs, axis=0, ignore_index=True)

    def  parseDf(self, df, name):
        dfs = []
        for col in df.columns:
            if col != 'X':
                dfs.append(self.createDfPerColumn(df[col], col, df['X'], name))

        return pd.concat(dfs, axis=0, ignore_index=True)

    def createDfPerColumn(self, df_col, col_name, x_col, virus):
        return pd.DataFrame(data={
            'virus': [col_name for _ in range(len(df_col))],
            'serum': [virus for _ in range(len(df_col))],
            'replicate': [1 for _ in range(len(df_col))],
            'concentration': x_col.values,
            'fraction infectivity': df_col.values,
        })
    
    def processViruses(self, dataframe, parsedVirusNames):
        dataframe = dataframe.merge(parsedVirusNames, left_on='serum', right_index=True, suffixes=('', '_new'))
        dataframe['serum'] = dataframe['serum_new']
        dataframe.drop(columns=['serum_new'], inplace=True)
        return dataframe
    
    def generate_process_combinations(self, dataframe):
        combinations = []
        for i in range(len(dataframe)):
            combinations.append((dataframe.loc[i, 'serum'], dataframe.loc[i, 'virus']))
        
        self.combinations = list(set(combinations))

    def create_serum_virus_dict(self, dataframe):
        serum_virus_dict = {}
        for _, row in dataframe.iterrows():
            serum = row['serum']
            virus = row['virus']
            if serum not in serum_virus_dict:
                serum_virus_dict[serum] = {}
            serum_virus_dict[serum][virus] = True
        return serum_virus_dict

    def process_replicates(self, dataframe):
        dct = {}

        for combo in self.combinations:
            if len(set(dataframe[(dataframe['serum'] == combo[0]) & (dataframe['virus'] == combo[1])]['date'].values)) > 1:
                date_objects = [datetime.strptime(date, '%d.%m.%Y') for date in list(set(dataframe[(dataframe['serum'] == combo[0]) & (dataframe['virus'] == combo[1])]['date'].values))]
                sorted_dates = sorted(date_objects)
                numbered_dates = {date.strftime('%d.%m.%Y'): i + 1 for i, date in enumerate(sorted_dates)}
                for key, val in numbered_dates.items():
                    dct[(combo[0], combo[1], key)] = val

        for key, value in dct.items():
            for idx in dataframe[(dataframe['serum'] == key[0]) & (dataframe['virus'] == key[1]) & (dataframe['date'] == key[2])].index:
                dataframe.loc[idx, 'replicate'] = int(value)