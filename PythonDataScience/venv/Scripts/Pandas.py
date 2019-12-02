import numpy as np
import pandas as pd

# pandas series objects
data = pd.Series([0.25,0.5,0.75,1.0])
data.values
data.index

data[1:3]

data = pd.Series([0.25,0.5,0.75,1.0],
                 index=['a','b','c','d'])

# turn dictionary into a series
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population['Texas']

# pandas dataframes
area_dict = {'California': 423967,
             'Texas': 695662,
             'New York': 141297,
             'Florida': 170312,
             'Illinois': 149995}
area = pd.Series(area_dict)
area

states = pd.DataFrame({'population':population,
                       'area':area})
states
states.index
states.columns

# dataframes as specialized directories

states['area']

# construct dataFrames
# from single series objects
pd.DataFrame(population, columns=['population'])

# from list of dictionaries
data = [{'a':i,'b': 2*i}
        for i in range(3)]
pd.DataFrame(data)

#missing keys
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

# from a dictionary of series
pd.DataFrame({'population': population,
              'area': area})

# from 2-dim NumPy arrays
pd.DataFrame(np.random.rand(3,2),
             columns=['foo','bar'],
             index=['a','b','c'])

# from NumPy strictured array
A = np.zeros(3,dtype=[('A','i8'),('B', '<f8')])
pd.DataFrame(A)

# pandas index object
ind = pd.Index([2,3,5,7,11])
ind

# index as immutable array
ind[1]
ind[::2]

print(ind.size,ind.shape,ind.ndim,ind.dtype)

# index as ordered sets
indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])

indA & indB
indA | indB
indA ^ indB

# Indexing and Selection

# series as dictionary
data = pd.Series([0.25,0.5,0.75,1],
                 index=['a','b','c','d'])

data['b']
'a' in data
data.keys()
list(data.items())
# new row
data['e'] = 1.25

# series as 1-dim array
data['a':'c']
data[0:2]
data[(data > 0.3) & (data < 0.8)]
data[['a','e']]

# indexers ; loc,iloc,ix
data = pd.Series(['a','b','c'],
                 index=[1,3,5])
data[1]
data[1:3]

#loc
data.loc[1]
data.loc[1:3]

#iloc
data.iloc[1]
data.iloc[1:3]

# data selection in DataFrames
# DataFrame as a dictionary
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

data['area']
data.area
data.area is data['area']

# false since pop is a method. conflict!
data.pop is data['pop']

# new column
data['density'] = data['pop']/data['area']

# DataFrames as 2-dim arrays
data.values
# transpose
data.T
# treat like NumPy array
data.values[0]

data.iloc[:2,:2]
data.loc[:'Texas', :'pop']

# ix is a hybrid
#data.ix[:3,:'pop']

# masking and fancy indexing
data.loc[data.density > 100 ,['pop','density']]

data.iloc[0,2]

data['Texas':'Florida']
data[data.density > 100]

# data operations Pandas
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0,10,4))
ser

df = pd.DataFrame(rng.randint(0,10,(3,4)),
                  columns= ['A','B','C','D'])

np.exp(ser)
np.sin(df * np.pi / 4)

# ufuncs index alignment
# Series
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

population / area
area.index & population.index

A = pd.Series([2,4,6], index=[0,1,2])
B = pd.Series([1,3,5], index=[1,2,3])

A+B
A.add(B)
A.add(B, fill_value=0)

# DataFrames
A = pd.DataFrame(rng.randint(0,20,(2,2)),
                 columns = list('AB'))

B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))

A + B

# fill with average values in A
fill = A.stack().mean()
A.add(B,fill_value=fill)

# ufuncs between series and DataFrames
A = rng.randint(10,size=(3,4))
A - A[0]

df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]

df.subtract(df['R'], axis=0)

halfrow = df.iloc[0, ::2]
df - halfrow

# handling missing data
# None: pythonic missing data

vals1 = np.array([1,None,3,4])
vals1

for dtype in ['object','int']:
    print("dtype=", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()

vals1.sum()

vals2 = np.array([1,np.nan,3,4])
#vals2.dtype

1+np.nan

vals2.sum(), vals2.min(), vals2.max()
#np.nansum(vals2)

#NaN and None in Pandas
pd.Series([1,np.nan,2,None])

data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]

# drop nulls
data.dropna()
# DataFrame
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df

df.dropna()
df.dropna(axis='columns')

df[3] = np.nan
df

df.dropna(axis='columns',how='all')
df.dropna(axis='rows',thresh=3)

# fill null values
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data

data.fillna(0)
data.fillna(method='ffill')
data.fillna(method='bfill')

# DataFrames
df.fillna(method='ffill',axis=1)

# hierarchical indexing
# tha bad way

index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = ([33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561])
pop = pd.Series(populations, index = index)

# the right way - Pandas multiIndex
index = pd.MultiIndex.from_tuples(index)
index

pop=pop.reindex(index)
pop

pop[:,2010]

# multiindex as extra dimention
pop_df = pop.unstack()
pop_df

pop_df.stack()

# add extra column
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
pop_df

f_u18 = pop_df['under18']/pop_df['total']
f_u18.unstack()

# methods of multiindex creation
# list of indexes
df = pd.DataFrame(np.random.rand(4, 2),
                  index = [['a','a','b','b'],[1,2,1,2]],
                  columns=['data1','data2'])

# dictionary
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)

# explicit multiindex constructors
pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])
pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
pd.MultiIndex.from_product([['a','b'],[1,2]])

# naming indexes
pop.index.names = ['state','year']
pop

# Multiindex for columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

health_data = pd.DataFrame(data, index = index , columns = columns)
health_data['Guido']

# indexing and slicing
pop
pop['California', 2000]
pop['California']
pop.loc['New York':'Texas']

pop[:,2000]

pop[pop > 22000000]
pop[['Texas','California']]

# Multiply indexed DataFrames
health_data
health_data['Guido','HR']

health_data.iloc[:2, :2]

idx = pd.IndexSlice
health_data.loc[idx[:,1] , idx[:, 'HR']]

# sort indexing
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data

data = data.sort_index()
data['a':'b']

# stack and unstack indexes
pop.unstack(level=0)
pop.unstack(level=1)

# index setting and reset
pop_flat = pop.reset_index(name='population')

pop_flat.set_index(['state','year'])

# aggregations on multi indexes
health_data
data_mean= health_data.mean(level='year')
data_mean

data_mean.mean(axis=1,level='subject')

# concat and append
def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data,ind)

make_df('ABC', range(3))

# class to display side by side objects
class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""

    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# concat NumPy arrays
x = [1,2,3]
y = [4,5,6]
z = [7,8,9]
np.concatenate([x,y,z])

x=[[1,2],
  [3,4]]
np.concatenate([x,x], axis=1)

# simple pandas concatenation
ser1 = pd.Series(['A','B','C'], index = [1,2,3])
ser2 = pd.Series(['D','E','F'], index = [4,5,6])
pd.concat([ser1,ser2])

df1 = make_df('AB', [1,2])
df2 = make_df('AB', [3,4])
display('df1', 'df2', 'pd.concat([df1, df2])')

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])

pd.concat([df3,df4], axis = 1)
display('df3', 'df4', "pd.concat([df3, df4], axis=1)")

# duplicate indices
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
x.index = y.index
pd.concat([x,y])

# catch dup index
try:
    pd.concat([x,y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)

#ignore index
pd.concat([x,y], ignore_index=True)

# add multiple keys
pd.concat([x,y], keys=['x','y'])

# concat with joins
df5 = make_df('ABC',[1,2])
df6 = make_df('BCD',[3,4])
pd.concat([df5, df6])
pd.concat([df5,df6], join='inner')
pd.concat([df5,df6], join='outer')

# with axes to keep input
pd.concat([df5,df6], join_axes=[df5.columns])

# append method
df1.append(df2)

# merges and joins
# one to one join
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})

df3 = pd.merge(df1,df2)

# many to one
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})

pd.merge(df3,df4)

# many to many
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})

pd.merge(df1,df5)
display('df1', 'df5', "pd.merge(df1, df5)")

# specify merge key with 'on'
pd.merge(df1,df2,on='employee')

# left_on and right_on - different column names
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
pd.merge(df1,df3,
         left_on="employee",
         right_on="name")

# drop dup column
pd.merge(df1,df3,
         left_on="employee",
         right_on="name").drop('name', axis=1)

# left and right index keywords
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
df1a
df2a

pd.merge(df1a, df2a ,left_index= True, right_index=True)

# join method
df1a.join(df2a)

# combine indexes and columns
pd.merge(df1a,df3, left_index= True, right_on= 'name')

# join set arithmetic
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])

pd.merge(df6,df7) # inner is default
pd.merge(df6,df7, how='left')

# overllaping colum names - suffixes keywords
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
pd.merge(df8,df9, on = 'name')
pd.merge(df8,df9,on = 'name', suffixes = ["_L","_R"])

# real example US data
pop = pd.read_csv('venv/Scripts/data/state-population.csv')
areas = pd.read_csv('venv/Scripts/data/state-areas.csv')
abbrevs = pd.read_csv('venv/Scripts/data/state-abbrevs.csv')

merged = pd.merge(pop, abbrevs , how = 'outer',
                  left_on='state/region',right_on='abbreviation')
merged.head(20)
merged.isnull().any()

merged[merged['population'].isnull().head()]
merged.loc[merged['state'].isnull(), 'state/region'].unique()

merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()

final = pd.merge(merged, areas,
                 on = 'state' ,
                 how='left')
final.head(5)
final.isnull().any()

final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna(inplace=True)

data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()

data2010.set_index('state',inplace=True)
density = data2010['population']/data2010['area (sq. mi)']
density.sort_values(ascending=False,inplace=True)
density.head()

# groupings and aggregations
import seaborn as sns
planets = sns.load_dataset('planets')
planets.head()

# simple aggregation pandas
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser

ser.sum()
ser.mean()

df =  pd.DataFrame({'A' : rng.rand(5),
                    'B' : rng.rand(5)})
df

df.mean()
df.mean(axis='columns')

planets.dropna().describe()

#group by
# split apply combine

df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data': range(6)},
                  columns= ['key','data'])
df

df.groupby('key').sum()

# the group by object
planets.groupby('method')['orbital_period'].median()

# iteretion over groups #manual
 for (method,group) in planets.groupby('method'):
     print("{0:30s} shape={1}".format(method, group.shape))


# dispatch methods
planets.groupby('method')['year'].describe().unstack()

#aggregate filter transform apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key' : ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1' : range(6),
                   'data2' : rng.randint(0,10,6)},
                  columns=['key','data1','data2'])
df

# aggregation
df.groupby('key').aggregate(['min', np.median, max])

df.groupby('key').aggregate({'data1': min,
                             'data2': max})

# filtering
def filter_func(x):
    return x['data2'].std() > 4

df.groupby('key').filter(filter_func)

# transformation

df.groupby('key').transform(lambda x: x- x.mean())

# apply
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x

df.groupby('key').apply(norm_by_data2)

# split key
L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum()

df.groupby(df['key']).sum()

 # map with dictionaries
 df2 = df.set_index('key')
 mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
 df2.groupby(mapping).sum()

 #any python function
 df2.groupby(str.lower).mean()
 df2.groupby([str.lower,mapping]).mean()

 #example
 decade = 10 * (planets['year'] // 10)
 decade = decade.astype(str) + 's'
 decade.name = 'decade'
 planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

 # pivot tables
 titanic = sns.load_dataset('titanic')
 titanic.head()

 #pivot by hand
 titanic.groupby('sex')[['survived']].mean()
 titanic.groupby(['sex','class'])['survived'].aggregate('mean').unstack()

 #pivot syntax
 titanic.pivot_table('survived', index = 'sex' , columns= 'class')

 #multilevel pivot tables
 age = pd.cut(titanic['age'], [0,18,80])
 titanic.pivot_table('survived', ['sex', age], 'class')

 fare = pd.qcut(titanic['fare'],2)
 titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

 # more options in pivot
 titanic.pivot_table(index ='sex', columns ='class',
                     aggfunc={'survived' : sum, 'fare' : 'mean'})

 titanic.pivot_table('survived', index = 'sex' , columns = 'class' , margins= True)

 # example Birthrate Data

births = pd.read_csv('venv/Scripts/data/births.csv')
births.head()

births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births',index = 'decade' , columns= 'gender' , aggfunc= 'sum')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
births.pivot_table('births',index = 'year' , columns= 'gender' , aggfunc= 'sum').plot()
plt.ylabel('total births per year')

# vectorized string operations
x = np.array([2,3,5,7,11,13])
x * 2

data = ['peter' , 'Paul' , 'MARY' , 'gUIDO']
[s.capitalize() for s in data]

# pandas series
names = pd.Series(data)
names

names.str.capitalize()

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])

monte.str.lower()
monte.str.len()
monte.str.startswith('T')
monte.str.split()


# regex
monte.str.extract('([A-Za-z]+)',expand = False)
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')

monte.str[0:3]

monte.str.split().str.get(-1)

full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
full_monte

full_monte['info'].str.get_dummies('|')

# working with time series
from datetime import datetime
datetime(year = 2015 ,month =7 ,day=4)

from dateutil import parser
date = parser.parse("4th of July, 2015")
date

date.strftime('%A')

date = np.array('2015-07-04' , dtype = np.datetime64)
date

date + np.arange(12)

np.datetime64('2015-07-04')
np.datetime64('2015-07-04 12:00')
np.datetime64('2015-07-04 12:59:59.50', 'ns')

# date time in pandas
date = pd.to_datetime("4th of July, 2015")
date

date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')

# indexing time series
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])

data = pd.Series([0,1,2,3], index=index)
data

data['2014-07-04':'2015-07-04']
data['2015']

dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                       '2015-Jul-6', '07-07-2015', '20150708'])
dates

dates - dates[0]

# regular sequences
pd.date_range('2015-07-03', '2015-07-10')

pd.date_range('2015-07-03', periods= 8)
pd.date_range('2015-07-03', periods= 8, freq= 'H')

pd.timedelta_range(0,periods=10,freq='H')

# example
data = pd.read_csv('venv/Scripts/data/FremontBridge.csv', index_col= 'Date',parse_dates= True)
data.head()
data.describe()
data=data.iloc[:,0:2 ]

data.columns = ['West', 'East']
data['Total'] = data.eval('West + East')
data.dropna().describe()

import seaborn
seaborn.set()

data.plot()
plt.ylabel('Hourly Bicycle Count')

weekly = data.resample('W').sum()
weekly.plot(style=[':', '--', '-'])
plt.ylabel('Weekly bicycle count')

daily = data.resample('D').sum()
daily.rolling(30 , center=True).sum().plot(style=[':', '--', '-'])
plt.ylabel('mean hourly count');

daily.rolling(50 , center = True,
              win_type= 'gaussian').sum(std=10).plot(style=[':', '--', '-'])

# drill down
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks = hourly_ticks , style=[':', '--', '-'])

by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
by_weekday.plot(style=[':', '--', '-'])

# weekday vs weekend
weekend = np.where(data.index.weekday < 5, 'Weekday' , 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.ix['Weekday'].plot(ax=ax[0], title='Weekdays',
                           xticks=hourly_ticks, style=[':', '--', '-'])
by_time.ix['Weekend'].plot(ax=ax[1], title='Weekends',
                           xticks=hourly_ticks, style=[':', '--', '-'])

# eval and query
import numpy as np
rng = np.random.RandomState(42)
x = rng.rand(1000000)
y = rng.rand(1000000)
%timeit x+y

mask = (x > 0.5) & ( y < 0.5)

# eval

import pandas as pd
nrows, ncols = 100000, 10
rng = np.random.RandomState(42)

df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows,ncols))
                      for i in range(4))
%timeit df1 + df2 + df3 + df4
%timeit pd.eval('df1 + df2 + df3 + df4') # 50% faster

np.allclose(df1 + df2 + df3 + df4,
            pd.eval('df1 + df2 + df3 + df4'))

# eval operations
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0,100,(100,3)))
                           for i in range(5))

# arithmetic
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)

# comparison
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('(df1 < df2) & (df2 <= df3) & (df3 != df4)')
np.allclose(result1,result2)

# bitwise
result1 = (df1 <0.5) & (df2 <0.5) | (df3 <df4)
result2 = pd.eval('(df1 <0.5) & (df2 <0.5) | (df3 <df4)')
np.allclose(result1, result2)

result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1,result3)

# attributes and indices
result1 = df2.T[0] + df3.iloc[1]
result2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)

# pandas column-wise eval
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
df.head()

result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
np.allclose(result1, result2)


result3 = df.eval('(A + B) / (C - 1)')
np.allclose(result1, result3)

# assignment in dataframe.eval()
df.head()

df.eval('D = (A + B) / C',inplace=True)
df.head()

# modify existing column
df.eval('D = (A - B) / C', inplace=True)
df.head()

# local variable in eval
column_mean = df.mean(1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean')
np.allclose(result1, result2)

# DataFrame query method
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1, result2)

result2 = df.query('A < 0.5 and B > 0.5')
np.allclose(result1, result2)

# local variables
Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')
np.allclose(result1, result2)

# performance
# big data --- > eval and query
df.values.nbytes