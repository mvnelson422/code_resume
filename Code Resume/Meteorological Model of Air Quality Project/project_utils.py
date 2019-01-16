import pandas as pd
import numpy as np
import os.path
from matplotlib import pyplot as plt

var_dict = {'Ozone' :'44201', # Pollutants
            'SO2'   :'42401',
            'CO'    :'42101',
            'NO2'   :'42602',
            'PM2.5 Mass'         :'88101', # Particulates
            'PM2.5 non FRM Mass' :'88502',
            'PM10 Mass'          :'81102',
            'PM2.5 Speciation'   :'SPEC',
            'PM10 Speciation'    :'PM10SPEC',
            'Wind'     :'WIND', # Meteorological
            'Temp'     :'TEMP',
            'Pressure' :'PRESS',
            'Dewpoint' :'RH_DP',
            'HAPs'     :'HAPS', # Toxics
            'VOCs'     :'VOCS',
            'NONOxNOy' :'NONOxNOy',
            'Lead'     :'LEAD',
            'AQI_cbsa'   :'aqi_by_cbsa', # Air Quality Index
            'AQI_county' :'aqi_by_county'} #unfortunately neither of these
                                           #have lattitude and longitude
    
columns_wanted_slim = ['Wind_Wind Speed - Resultant_Arithmetic Mean',
       'Wind_Wind Speed - Resultant_1st Max Value', 
       'Temp_Arithmetic Mean',
       'Temp_1st Max Value', 'Pressure_Arithmetic Mean',
       'Pressure_1st Max Value', 'Dewpoint_Dew Point_Arithmetic Mean',
       'Dewpoint_Dew Point_1st Max Value',
       'Dewpoint_Relative Humidity _Arithmetic Mean',
       'Dewpoint_Relative Humidity _1st Max Value',
       'season_fall', 'season_spring', 'season_winter',
       'sin_direction','cos_direction']

columns_wanted_full = ['Wind_Wind Speed - Resultant_Arithmetic Mean',
    'Wind_Wind Speed - Resultant_1st Max Value',
    'Wind_Wind Direction - Resultant_Arithmetic Mean',
    'Temp_Arithmetic Mean',
    'Temp_1st Max Value',
    'Pressure_Arithmetic Mean',
    'Pressure_1st Max Value',
    'Dewpoint_Dew Point_Arithmetic Mean',
    'Dewpoint_Dew Point_1st Max Value',
    'Dewpoint_Relative Humidity _Arithmetic Mean',
    'Dewpoint_Relative Humidity _1st Max Value',
    'season_fall',
    'season_spring',
    'season_winter',
    'sin_direction','cos_direction',
    'Wind_Wind Speed - Resultant_1st Max Value_3_day_max',
    'Temp_1st Max Value_3_day_max',
    'Pressure_1st Max Value_3_day_max',
    'Dewpoint_Dew Point_1st Max Value_3_day_max',
    'Dewpoint_Relative Humidity _1st Max Value_3_day_max',
    'Wind_Wind Speed - Resultant_Arithmetic Mean_3_day_mean',
    'Wind_Wind Direction - Resultant_Arithmetic Mean_3_day_mean',
    'Temp_Arithmetic Mean_3_day_mean',
    'Pressure_Arithmetic Mean_3_day_mean',
    'Dewpoint_Dew Point_Arithmetic Mean_3_day_mean',
    'Dewpoint_Relative Humidity _Arithmetic Mean_3_day_mean',
    'Wind_Wind Speed - Resultant_1st Max Value_7_day_max',
    'Temp_1st Max Value_7_day_max',
    'Pressure_1st Max Value_7_day_max',
    'Dewpoint_Dew Point_1st Max Value_7_day_max',
    'Dewpoint_Relative Humidity _1st Max Value_7_day_max',
    'Wind_Wind Speed - Resultant_Arithmetic Mean_7_day_mean',
    'Temp_Arithmetic Mean_7_day_mean',
    'Pressure_Arithmetic Mean_7_day_mean',
    'Dewpoint_Dew Point_Arithmetic Mean_7_day_mean',
    'Dewpoint_Relative Humidity _Arithmetic Mean_7_day_mean',
    'Wind_Wind Speed - Resultant_1st Max Value_14_day_max',
    'Temp_1st Max Value_14_day_max',
    'Pressure_1st Max Value_14_day_max',
    'Dewpoint_Dew Point_1st Max Value_14_day_max',
    'Dewpoint_Relative Humidity _1st Max Value_14_day_max',
    'Wind_Wind Speed - Resultant_Arithmetic Mean_14_day_mean',
    'Wind_Wind Direction - Resultant_Arithmetic Mean_14_day_mean',
    'Temp_Arithmetic Mean_14_day_mean',
    'Pressure_Arithmetic Mean_14_day_mean',
    'Dewpoint_Dew Point_Arithmetic Mean_14_day_mean',
    'Dewpoint_Relative Humidity _Arithmetic Mean_14_day_mean']
    
def fetch_tables(root_url='https://aqs.epa.gov/aqsweb/airdata/',
                 folder='data', clean=True, gpd_bool=True, verbose=True,
                 frequency = 'daily', years=['2014','2015','2016','2017','2018'],
                 vars_to_get = ['Ozone','SO2','CO','NO2','Wind','Temp','Pressure',
                                'Dewpoint','PM2.5 Mass','PM10 Mass']):
    """
    For each year listed:
        Fetches table for and the specified frequency
        possible frequencies include 'daily', 'hourly', and '8hour'.
        
    For each variable listed:
        If true, fetch all variables of that category.
        If false, fetch no tables of that cateogry.
        If list, fetch only those specified.
        
    Tables are scraped if not already present in folder specified.
    
    columns are dropped as specified and otherwise cleaned as necessary
    """
    
    if gpd_bool:
        from shapely.geometry import Point
    
    tables = dict()
    date_columns = ['Date Local','Date of Last Change']
    
    # collect specified variables over correct frequencies and years
    for var_name in vars_to_get:
        for year in years:
            var = var_dict[var_name]
            path = frequency + '_' + var + '_' + year
            filename = folder + '/' + path + '.csv'
            dict_key = var_name

            temp_df = pd.DataFrame()

            # if file already present in folder, load it to csv and store in tables dictionary
            if os.path.exists(filename):
                if verbose:
                    print('\n' + dict_key + '_' + year,'found.')
                temp_df = pd.read_csv(filename,parse_dates=date_columns)

            # otherwise, scrape it from root_url using built in pandas.read_csv functionality 
            # and store in tables dictionary. Note raw files are .zip, but turns out pd.read_csv
            # is okay with that given a url to download the file from. 
            # Tried it on a hunch and it got rid of a lot of unneeded code.
            else:
                if verbose:
                    print('\n' + dict_key + '_' + year,'not found. Beginning download.')
                url_loc = root_url + path + '.zip'
                
                temp_df = pd.read_csv(url_loc,parse_dates=date_columns)

                if verbose:
                    print(dict_key + '_' + year,'downloaded.')
                temp_df.to_csv(filename, index=False, encoding='utf-8') #index false, otherwise increases size when saving
                
            # for each type of measurement in temp_df add to separate table using parameter name as part of key
            # if more than one measurement in csv
            params = list(set(temp_df['Parameter Name']))
            for param in params:
                if len(params) > 1:
                    dict_key_n = dict_key + '_' + param
                else:
                    dict_key_n = dict_key
                    
                temp_df_n = temp_df[temp_df['Parameter Name']==param]
                
                # if table already in tables append to it, otherwise add it. Allows adding multiple years together.
                if dict_key_n in tables.keys():
                    if verbose:
                        print('Appending to table ' + dict_key_n + '.')
                    tables[dict_key_n] = pd.concat([tables[dict_key_n],temp_df_n])
                else:
                    if verbose:
                        print('Adding table ' + dict_key_n + '.')
                    tables[dict_key_n] = temp_df_n

    if clean:
        if verbose:
            print('Cleaning Tables')
        to_drop = ['State Code', 'County Code', 'Site Num', 'Parameter Code', 
       'POC', 'Datum', 'Sample Duration', '1st Max Hour',
       'Pollutant Standard', 'Event Type', 'Units of Measure', 'Parameter Name',
       'Method Code', 'Method Name', 'Observation Count', 'Observation Percent',
       'Local Site Name', 'Address', 'State Name', 'County Name', 'City Name',
       'CBSA Name', 'Date of Last Change'] # 'Units of Measure', 'Parameter Name'
        
        print('Units of measure:')
        
        keys = var_dict.keys()
        for key in tables.keys():
            print('    ' + key + ':',set(tables[key]['Parameter Name']),' | ',set(tables[key]['Units of Measure']), '\n')
            tables[key] = tables[key].drop(to_drop, axis=1)
            
            #drop AQI from meteorological tables
            if key in ['Temp', 'Pressure', 'Dewpoint_Relative Humidity ', 'Dewpoint_Dew Point', 
                       'Wind_Wind Direction - Resultant', 'Wind_Wind Speed - Resultant']:
                
                tables[key] = tables[key].drop(['AQI'], axis=1)
            
            if gpd_bool:
                #convert to geopandas dataframe
                tables[key]['Coordinates'] = list(zip(tables[key].Longitude, tables[key].Latitude))
                tables[key]['Coordinates'].apply(Point)
                #tables[key] = geopandas.GeoDataFrame(tables[key], geometry='Coordinates')
        
    return tables



def fill_AQI(tables, new_column_name = 'AQI'):
    """
    Replaces or creates new colunn with specified name. Fills with calculated
    AQI based on Wikipedia article above.
    
    PM2.5 Mass aligns almost perfectly with previously calculated values, but the other two do not. I suspect this is because the specified period of measurement is not 24 hours, which is how all data used was collected. Investigate this further next semester.
    """
    def sub_fill(key, values, round_precision):
        concentrations = np.round(tables[key]['Arithmetic Mean'] ,round_precision)
        
        tables[key][new_column_name] = 0
        
        for value_set in values:
            Clow, Chigh, Ilow, Ihigh = value_set
            c_loc = (concentrations >= Clow) & (concentrations <= Chigh)
            
            tables[key][new_column_name].loc[c_loc] = np.ceil((Ihigh-Ilow)/
                            (Chigh-Clow)*(concentrations[c_loc]-Clow) + Ilow)
    
    if 'SO2' in tables.keys():
        # SO2
        # values for Clo2, Chigh, Ilow, Ihigh in each bracket - units ppb
        values_SO2 = [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), 
                      (186, 304, 151, 200), (305, 604, 201, 300), 
                      (605, 804, 301, 400), (805, 1004, 401, 500)]
        sub_fill('SO2', values_SO2, 0)
    
    if 'CO' in tables.keys():
        # CO
        # values for Clo2, Chigh, Ilow, Ihigh in each bracket - units ppm
        values_CO = [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), 
                     (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), 
                     (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)]
        sub_fill('CO', values_CO, 1)
    
    if 'PM2.5 Mass' in tables.keys():
        # PM2.5
        # values for Clo2, Chigh, Ilow, Ihigh in each bracket 
        # units micro-grams per cubic meter
        values_PM = [(0, 12.0 , 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                     (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), 
                     (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
        sub_fill('PM2.5 Mass', values_PM, 1)
    

def max_means(tables, periods=[3,7,14,28], means=['Arithmetic Mean'],
              maxs=['1st Max Value'],
              keys=['Wind_Wind Direction - Resultant',
                    'Wind_Wind Speed - Resultant','Temp','Pressure',
                    'Dewpoint_Dew Point','Dewpoint_Relative Humidity ']):
    for key in keys:
        for period in periods:
            for mean in means:
                newname = mean+' ('+str(period)+' day prior mean)'
                pt = pd.pivot_table(tables[key],index=['Coordinates',
                                'Date Local'])[mean].unstack(0)
                pt = pt.rolling(window=period).mean().stack().swaplevel(0,
                                            1).reset_index(name=newname) 
                
                tables[key]  = tables[key].merge(pt,on=['Coordinates',
                                                        'Date Local'])
            for max_ in maxs:
                newname = max_+' ('+str(period)+' day prior max)'
                pt = pd.pivot_table(tables[key],index=['Coordinates',
                                                       'Date Local'])[max_]
                pt = pt.unstack(0).rolling(window=period).max().stack().swaplevel(0,
                                                    1).reset_index(name=newname) 
                tables[key]  = tables[key].merge(pt,on=['Coordinates',
                                                        'Date Local'])
                
                
def region_2D_interp(tables, keys, dates, value_keys=['Arithmetic Mean'], 
                     longitude_bounds=(-130, -65), latitude_bounds=(25, 50),
                     plot=True, N = 1000, M = 1000, cmap='viridis',
                     return_bool=True,abs_min_max=True):
    
    from scipy.interpolate import LinearNDInterpolator
    
    if plot:
        from mpl_toolkits.basemap import Basemap
    
    min_x, max_x = longitude_bounds
    min_y, max_y = latitude_bounds
    
    functions = dict()
    
    x = np.linspace(min_x, max_x, N)
    y = np.linspace(min_y, max_y, M)

    X, Y = np.meshgrid(x, y)

    for key in keys:
        table = tables[key]
        
        # if min or max values are not defined set them to the min or max
        if min_x is None:
            min_x = np.min(table['Longitude'])
        if max_x is None:
            max_x = np.max(table['Longitude'])
        if min_y is None:
            min_y = np.min(table['Latitude'])
        if max_y is None:
            max_y = np.max(table['Latitude'])
        
        dayfuncs = dict()
        for date in dates:
            datestr = str(date.month) + '-' + str(date.day) + '-' + str(date.year)
            
            table_day = table[table['Date Local'] == date]
            
            valuefuncs = dict()
            for value_key in value_keys:
                
                if abs_min_max:
                    value_max = np.max(table[value_key])
                    value_min = np.min(table[value_key])
                else:
                    value_max = np.max(table_day[value_key])
                    value_min = np.min(table_day[value_key])
                
                values = table_day[(table_day['Longitude'] > min_x) & 
                                   (table_day['Longitude'] < max_x) &
                                   (table_day['Latitude'] > min_y)  &
                                   (table_day['Latitude'] < max_y)][value_key]
                
                points = table_day[(table_day['Longitude'] > min_x) & 
                                   (table_day['Longitude'] < max_x) &
                                   (table_day['Latitude'] > min_y)  & 
                                   (table_day['Latitude'] < max_y)][['Longitude',
                                                                     'Latitude']]

                # interpolate linearly on available points
                g = LinearNDInterpolator(points, values, fill_value=value_min)
                
                if plot:
                    title_str = key + ': ' + value_key + ', ' + datestr + '\nsamples: '+str(len(values))
                    
                    plt.figure(figsize=(20,6))
                    
                    m = Basemap(llcrnrlon=min_x, llcrnrlat=min_y, urcrnrlon=max_x,
                                urcrnrlat=max_y, suppress_ticks=False)
                    m.drawcoastlines(color='#FFFFFF',linewidth=1)
                    m.drawcountries(color='#FFFFFF',linewidth=1)
                    m.drawstates(color='#FFFFFF',linewidth=1)
                    
                    p = plt.pcolormesh(X, Y, g(X,Y), cmap=cmap, vmax=value_max, 
                                       vmin=value_min)
                    plt.colorbar(p)
                    plt.title(title_str)
                    plt.xlabel('Longitude')
                    plt.ylabel('Lattitude')
                    plt.show()
                
                valuefuncs[value_key] = g
            dayfuncs[datestr] = valuefuncs
        functions[key] = dayfuncs
    
    if return_bool:
        return X, Y, functions

def animate_region(tables, table_key, var_key, dates, value_min, value_max, 
                   longitude_bounds=(-130, -65), latitude_bounds=(25, 50), 
                   M=1000, N=1000, cmap='viridis', filename='animation.mp4'):
    """
    It is assumed that dates are given in the order they should be animated in.
    """
    
    from mpl_toolkits.basemap import Basemap
    from matplotlib import animation
    
    print('Collecting interpolating functions')
    
    X, Y, gfuncs = region_2D_interp(tables, [table_key], dates, value_keys=[var_key], 
                    longitude_bounds=longitude_bounds, latitude_bounds=latitude_bounds, 
                    plot=False, N=N, M=M, cmap=cmap, abs_min_max=True)
    
    print('Generating animation')
    
    gfuncs = gfuncs[table_key]
    date_strs = list(gfuncs.keys()) #They were added by date and still in order
    
    #generate figure and set properties
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)

    min_x, max_x = longitude_bounds
    min_y, max_y = latitude_bounds
    
    m = Basemap(llcrnrlon=min_x, llcrnrlat=min_y, urcrnrlon=max_x,
                urcrnrlat=max_y, suppress_ticks=False)
    m.drawcoastlines(color='#FFFFFF',linewidth=1)
    m.drawcountries(color='#FFFFFF',linewidth=1)
    m.drawstates(color='#FFFFFF',linewidth=1)
    
    g = gfuncs[date_strs[0]][var_key]
    pmesh = ax.pcolormesh(X, Y, g(X,Y)[:-1, :-1], cmap=cmap, vmax=value_max, 
                          vmin=value_min)

    #update function gets called repeatedly to create new frames in animation
    def update(i):
        z = gfuncs[date_strs[i]][var_key](X,Y)
        z = z[:-1, :-1]
        pmesh.set_array(z.ravel())
        plt.title(table_key + ': ' + var_key + ' ' + date_strs[i])

        return pmesh

    plt.ioff() # Turn off interactive mode to hide rendering animations
    #plt.title(table_key + ': ' + var_key)
    plt.colorbar(pmesh)
    plt.xlabel('Longitude')
    plt.ylabel('Lattitude')

    animation.writer = animation.writers['ffmpeg']
    
    #save the animation to a file outside the notebook
    ani = animation.FuncAnimation(fig, update, frames=range(len(dates)))
    ani.save(filename)