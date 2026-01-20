# -*- coding: utf-8 -*-

# Log reading class - 09/2019
# Julius Andretti

import os, inspect, sys
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import mode
from io import StringIO  
import pandas as pd
import matplotlib.pyplot as plt

from check_consistency import consistency_check,datetime_diff

def identify_lines_to_ignore(string_io,identify_until_reached,encoding="latin-1",return_ignored_lines=False):
    """Gets the location of a line containing a pattern in the log and saves what came before

    Parameters
    ----------
    string_io : io.StringIO
        Actiware log in StringIO format
    identify_until_reached : str
        The lines before a line that contains this string will be ignored
    encoding : str, default "utf-8"
        The file character encoding
    return_ignored_lines : boolean, default False
        If True, the lines that will be ignored are returned as a single string

    Returns
    -------
    lines_to_ignore : int
        The number of lines to be ignored
    ignored_lines : str, only if return_ignored_lines==True
        The lines that will be ignored as a single string
    pattern_found : boolean
        Indicates if the identify_until_reached pattern was found in any line
    """

    identify_until_reached_length = len(identify_until_reached)
    number_of_lines = len(string_io.readlines())
    string_io.seek(0) # Returning the stream to the start

    lines_to_ignore = 0

    ignored_lines = string_io.readline() # This variable will store the initial section that will be ignored
    if ignored_lines.find(identify_until_reached) < 0:
        line = string_io.readline()         
        line_length = len(line)

        lines_to_ignore = 1
        while (lines_to_ignore < number_of_lines-1) and ((line_length < identify_until_reached_length) or (line.find(identify_until_reached) < 0)): # Everything above this line will be ignored
            ignored_lines += line
            lines_to_ignore += 1
            line = string_io.readline()
            line_length = len(line)

    string_io.seek(0)

    pattern_found = True
    if lines_to_ignore == number_of_lines-1:
        pattern_found = False

    if return_ignored_lines:
        return ignored_lines, lines_to_ignore, pattern_found
    else:
        return lines_to_ignore, pattern_found

class LogRead:
    def data_update(self):
        self.timestamps = self.data.index

        if 'MS' in self.columns:
            self.ms = self.data['MS'].to_numpy() 

        if 'EVENT' in self.columns:
            self.event = self.data['EVENT'].to_numpy()

        if 'TEMPERATURE' in self.columns:
            self.temperature = self.data['TEMPERATURE'].to_numpy()

        if 'EXT TEMPERATURE' in self.columns:
            self.ext_temperature = self.data['EXT TEMPERATURE'].to_numpy()

        if 'ORIENTATION' in self.columns:
            self.orientation = self.data['ORIENTATION'].to_numpy()

        if 'PIM' in self.columns:
            self.pim = self.data['PIM'].to_numpy()

        if 'PIMn' in self.columns:
            self.pim_n = self.data['PIMn'].to_numpy()

        if 'TAT' in self.columns:
            self.tat = self.data['TAT'].to_numpy()

        if 'TATn' in self.columns:
            self.tat_n = self.data['TATn'].to_numpy()

        if 'ZCM' in self.columns:
            self.zcm = self.data['ZCM'].to_numpy()

        if 'ZCMn' in self.columns:
            self.zcm_n = self.data['ZCMn'].to_numpy()

        if 'LIGHT' in self.columns:
            self.light = self.data['LIGHT'].to_numpy()

        if 'AMB LIGHT' in self.columns:
            self.amb_light = self.data['AMB LIGHT'].to_numpy()

        if 'RED LIGHT' in self.columns:
            self.red_light = self.data['RED LIGHT'].to_numpy()

        if 'GREEN LIGHT' in self.columns:
            self.green_light = self.data['GREEN LIGHT'].to_numpy()

        if 'BLUE LIGHT' in self.columns:
            self.blue_light = self.data['BLUE LIGHT'].to_numpy()

        if 'IR LIGHT' in self.columns:
            self.ir_light = self.data['IR LIGHT'].to_numpy()

        if 'UVA LIGHT' in self.columns:
            self.uva_light = self.data['UVA LIGHT'].to_numpy()

        if 'UVB LIGHT' in self.columns:
            self.uvb_light = self.data['UVB LIGHT'].to_numpy()

        if 'STATE' in self.columns:
            self.state = self.data['STATE'].to_numpy()

    def check_consistency(self):
        self.consistency = consistency_check(self.data,self.duration)

    def solve_consistency(self,report_file="",max_gap=20*60):
        def drop_rows(removes):
            remove = []
            for r in removes:
                gap = 0
                if (r > 0) and (r < len(self.data)-1):
                    if self.data.index[r].year == 1970:
                        if ((self.data.index[r+1].year != 2000) and (self.data.index[r-1].year != 2000)):
                            gap = (self.data.index[r+1]-self.data.index[r-1]).total_seconds()
                    else:
                        if ((self.data.index[r+1].year != 1970) and (self.data.index[r-1].year != 1970)):
                            gap = (self.data.index[r+1]-self.data.index[r-1]).total_seconds()

                if gap <= max_gap:
                    if not isinstance(report_file, str):
                        message = "row "+str(r)+" (timestamp="+str(self.data.index[r])+") was dropped during inconsistency solving\n"
                        report_file.write(message)
                else:
                    remove.append(r)

            for r in remove:
                removes.remove(r)

            self.data["num"] = range(len(self.data))
            
            for r in removes:
                self.data = self.data[self.data["num"] != r]

            self.data.drop(columns=["num"],inplace=True)
            self.timestamps = self.data.index

        def check_year(year):
            check_year = self.consistency[self.consistency["desc"] == year]
            num_year = len(check_year) 

            if num_year > 0:
                idx = check_year["index"].to_numpy()
                removes = []
                if num_year > 2:
                    diff = np.diff(idx)
                    non_seq = []

                    for i in range(len(diff)):
                        if (i > 0) and (i < len(diff)-1):
                            if ((diff[i] > 1) and (diff[i+1] > 1)):
                                non_seq.append(i)
                        else:
                            if (diff[i] > 1):
                                non_seq.append(i)

                    if len(non_seq) > 0:
                        if non_seq[0] == 0:
                            removes.append(idx[0])
                        for i in range(len(non_seq)):
                            removes.append(idx[non_seq[i]+1])

                elif num_year == 2:
                    diff = np.diff(idx)
                    if diff[0] > 1:
                        for idxx in idx:
                            removes.append(idxx)

                else:
                    removes.append(idx[0])

                drop_rows(removes)

        self.check_consistency()
        if len(self.consistency) > 0:
            check_year("1970")
            self.check_consistency()
            if len(self.consistency) > 0:
                check_year("2000")
                self.check_consistency()
                if len(self.consistency) > 0:
                    for i in range(len(self.consistency)):
                        if self.consistency.at[i,"desc"] == "gap":
                            if self.consistency.at[i,"gap"] <= max_gap:
                                self.consistency.at[i,"ignore"] = True

                                if not isinstance(report_file, str):
                                    message = "gap from "+str(self.consistency.at[i,"from"])+" to "+str(self.consistency.at[i,"to"])+" was processed and ignored\n"
                                    report_file.write(message)

    def __init__(self,file,ignore_rows=0,duration=0,encoding="latin-1",decimal=".",delimiter=";",dayfirst=True,check_consistency=True,header_pattern="DATE/TIME;",footer_pattern="",parse_dates=['DATE/TIME'],header_dict=True):
        # file is a string containing the path to the log file
        # header indicates if the log file has a header

        if not isinstance(file, str):
            raise Exception("Log path must be a string!")


        log = open(file,encoding=encoding)        
        string_io = StringIO(log.read())
        log.close()

        header = []
        lines_to_ignore = 0
        if len(header_pattern) > 0:
            header, lines_to_ignore, pattern_found = identify_lines_to_ignore(string_io,header_pattern,return_ignored_lines=True)

        self.encoding = encoding
        self.file_name = file
        
        self.header = header

        if header_dict:
            header_dict = {}
            header_lines = header.split("\n")
            if len(header_lines) > 3:
                header_lines = header_lines[1:len(header_lines)-2] 
                for line in header_lines:
                    split = line.split(" : ")
                    if len(split) > 1:
                        key,value = split
                    else:
                        key = split[0]
                        value = ""
                    header_dict[key] = value
        else:
            header_dict = {}

        self.header_dict = header_dict
        
        # print(header)
        # print(header_lines)
        # print(header_dict)

        if len(footer_pattern) == 0:
            data = pd.read_csv(file,
                            skiprows=lines_to_ignore,
                            delimiter=delimiter,
                            encoding=encoding,
                            parse_dates=parse_dates,
                            dayfirst=dayfirst,
                            decimal=decimal,
                            )
        else:
            lines_to_footer, pattern_found = identify_lines_to_ignore(string_io,footer_pattern,return_ignored_lines=False)
            print(lines_to_ignore)
            print(lines_to_footer)
            data = pd.read_csv(file,
                       skiprows=lines_to_ignore,
                       nrows=lines_to_footer-lines_to_ignore-ignore_rows,
                       delimiter=delimiter,
                       encoding=encoding,
                       parse_dates=parse_dates,
                       dayfirst=dayfirst,
                       decimal=decimal,
                       )
            
        parse_dates = np.array(parse_dates)
        if parse_dates.size == 1:
            datetime_index = parse_dates[0]
        elif parse_dates.size == 2:
            datetime_index = parse_dates[0,0]+"_"+parse_dates[0,1]

        data.set_index(data[datetime_index], inplace=True)

        self.data = data
        self.columns = self.data.columns
        self.data_update()

        n = data.shape[0]
        if duration == 0:
            if n > 1:
                self.duration = mode(datetime_diff(data.index),keepdims=True).mode[0]   # Epoch duration, most repeated interval between measures
            else:
                self.duration = duration
        else:
            self.duration = duration

        self.consistency = None
        if check_consistency:
            self.consistency = consistency_check(data,self.duration)

    def plotter(self):
        for i in range(len(self.columns)):
            print(str(i+1)+'. '+self.columns[i])
        plot = int(input('Enter the number assigned to the variable you want to plot: '))
        plt.figure()
        self.data[self.columns[plot]].plot()
        plt.show() 
    
    def list(self):
        print('List of variables read from log file:')
        for i in range(len(self.columns)):
            print('- '+self.columns[i])

    def rewrite(self,data,name,sep=';'):
        self.data = data
        
        log = open(name,encoding=self.encoding,mode="w")
        log.write(self.header)
        log.close()

        self.data.to_csv(path_or_buf=name,sep=sep,header=True,index=False,date_format="%d/%m/%Y %H:%M:%S",mode="a")