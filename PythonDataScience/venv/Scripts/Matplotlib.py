import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('classic')
x = np.linspace(0 , 10 , 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()


fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')

# save
fig.savefig('my_figure.png')

# matlab style
plt.figure()
plt.subplot(2,1,1)
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))

# object oriented style
fig, ax= plt.subplots(2)
ax[0].plot(x, sin(x))
ax[1].plot(x, cos(x))

#line plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0,10,1000)
ax.plot(x, sin(x))

plt.plot(x ,np.sin(x))
plt.plot(x ,np.cos(x))

#line colors and styles
plt.plot(x, np.sin(x) ,color ='#E0E1DD')
plt.plot(x, np.cos(x) , linestyle = 'dotted')

#axes limits
plt.plot(x, np.sin(x))

plt.xlim(-1,11)
plt.ylim(-1.5,-1.5);

plt.plot(x, np.sin(x))

plt.xlim(10, 0)
plt.ylim(1.2, -1.2);

plt.plot(x, np.sin(x))
plt.axis([-1,11,-1.5,-1.5]);

#auto fix axis
plt.plot(x, np.sin(x))
plt.axis('tight');

plt.plot(x, np.sin(x))
plt.axis('equal');

#labeling
plt.plot(x, np.sin(x))
plt.title('A sine curve')
plt.xlabel("x")
plt.ylabel("sin(x)");

plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b',label ='cos(x)')
plt.axis('equal')

plt.legend();

# scatter plots
x = np.linspace(0,10,30)
y = np.sin(x)

plt.plot(x,y,'o' ,color = 'black');

# different markers
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd'] :
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label = "marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0, 1.8)

plt.plot(x, y ,'-ok')

plt.plot(x,y, '-p',color = 'gray',
         markersize=15, linewidth=4,
         markerfacecolor = 'white',
         markeredgecolor = 'gray',
         markeredgewidth = 2)
plt.xlim(-1.2, -1.2);

# plt.scatter plots
plt.scatter(x,y,marker = 'o')

rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x,y, c =colors , s=sizes , alpha= 0.2)
plt.colorbar();

from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T


plt.scatter(features[0] , features[1] , alpha= 0.2,
            s = 100*features[3], c =iris.target, cmap='viridis' )

#plt.plot is faster

# visualize errors
x = np.linspace(0,10,50)
dy = 0.8
y= np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x,y,yerr=dy,fmt = '.k')

plt.errorbar(x,y, yerr = dy ,
             fmt ='o',
             color ='black',
             ecolor='lightgray',
             elinewidth=3,
             capsize=0)

# continuous errors
from sklearn.gaussian_process import GaussianProcess

# define the model and draw some data
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# Compute the Gaussian process fit
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                     random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region

# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')

plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.xlim(0, 10);

# 3 dim viz
def f(x,y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0,5,50)
y = np.linspace(0,5,40)

X,Y = np.meshgrid(x,y)
Z = f(X,Y)

plt.contour(X,Y,Z , colors = 'black')
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.colorbar();

# as image
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image');

# with labels
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar();

# histogram binning and density
data = np.random.randn(1000)
plt.hist(data)

plt.hist(data , bins= 30 , normed= True ,alpha = 0.5 ,
         histtype= 'stepfilled', color = 'steelblue',
         edgecolor = 'black');

x1 = np.random.normal(0 , 0.8 , 1000)
x2 = np.random.normal(-2 , 1 , 1000)
x3 = np.random.normal(3 , 2 , 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

# just compute histogram
np.histogram(data, bins=5)

# two dim hist
mean = [0,0]
cov = [[1,1],[1,2]]
x, y = np.random.multivariate_normal(mean, cov,10000).T

plt.hist2d(x,y, bins = 30, cmap = 'Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')

plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')

# plot legends
plt.style.use('classic')
x = np.linspace(0,10,1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b',label ='Sine')
ax.plot(x, np.cos(x), '--r',label ='Cosine')
ax.axis('equal')
leg = ax.legend();

ax.legend(loc='upper left',frameon = False)
fig

ax.legend(frameon = False , loc = ' lower center', ncol =2)
fig

ax.legend(fancybox=True, framealpha=1,shadow=True,borderpad=1)
fig

# choose element legends
y = np.sin(x[:, np.newaxis] + np.pi *np.arange(0 ,2,0.5))
lines = plt.plot(x,y)
plt.legend(lines[:2], ['first', 'second']);

plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True);

# size points legends
import pandas as pd
cities = pd.read_csv('venv/Scripts/data/california_cities.csv')
cities.head()

lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

plt.scatter(lon, lat , label = None,
            c = np.log10(population),
            cmap='viridis',
            s=area ,
            linewidth= 0 ,
            alpha= 0.5)
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label = 'log$_{10}$(population)')
plt.clim(3, 7)

#legend creation
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

plt.title('California Cities: Area and Population');

# colorbars customization
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar()

#discrete color bars
plt.imshow(I, cmap=plt.cm.cmap('Blues',6))
plt.colorbar()
plt.clim(-1,1)

# multiple subplots
plt.style.use('seaborn-white')
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.5,0.8,0.4],
                   xticklabels= [],ylim=(-1.2,1.2))
ax2 = fig.add_axes([0.1,0.5,0.8,0.4],
                   ylim=(-1.2,1.2))
x = np.linspace(0,10)
ax.plot(np.sin(x))
ax.plot(np.cos(x));

for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
             fontsize=18, ha='center')

grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);

# Create some normally distributed data
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal', color='gray')
y_hist.invert_xaxis()

# text and annotation
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd

births = pd.read_csv('venv/Scripts/data/births.csv')
quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)

births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax);


fid, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

#add labels to plot
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

# arrows and annotation
fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

# custom ticks
ax = plt.axes(xscale='log', yscale='log')
ax.grid()

# hide ticks
ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())

# increase or reduce num ticks
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)

# For every axis, set the x and y major locator
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
fig

# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig

# three dim plots
from mpl_toolkits import mplot3d

fig = plt.figure()

ax = plt.axes(projection = '3d')

zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

zdata = 15 * np.random.randn(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# seaborn
import seaborn as sns
sns.set()

# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

plt.plot(x,y)
plt.legend('ABCDEF', ncol=2, loc='upper left');

# histograms
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)

for col in 'xy':
    sns.kdeplot(data[col], shade=True)

sns.distplot(data['x'])
sns.distplot(data['y']);

sns.kdeplot(data);

with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');

# pairplots

iris = sns.load_dataset("iris")
iris.head()

sns.pairplot(iris, hue='species', size = 2.5)

# facete histogram
tips = sns.load_dataset("tips")
tips.head()

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row = "sex" , col = "time" ,margin_titles=True)
grid.map(plt.hist, "tip_pct" , bins = np.linspace(0,40,15))

# factor plots
with sns.axes_style(style = 'ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data = tips, kind = "box")
    g.set_axis_labels("Day" , "Total Bill")

#  joint distributions
with sns.axes_style("white"):
    sns.jointplot("total_bill", "tip", data=tips,kind='reg')

# bar plots
planets = sns.load_dataset('planets')
planets.head()

with sns.axes_style('white'):
    g = sns.factorplot("year", data = planets , aspect = 2 ,
                       kind = "count", color = 'steelblue')
    g.set_xticklabels(step = 5)

with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                           hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')

