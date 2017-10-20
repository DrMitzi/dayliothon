import pandas as pd 
import re
import numpy as np
import datetime

#define path and name of file to read
daylio_csv = 'daylio_export_public.csv'

#read the file, ignoring weekday and time and adding columns as needed
df = pd.read_csv(daylio_csv, header = 0, usecols = [0,1,4,5,6,7], names = ['Year', 'Date', 'Mood', 'Obs', 'Note', 'contA'], skipinitialspace=True, quotechar='"', doublequote=False, lineterminator='"')

#Concatenate Year with Date in the new Date column and remove /n
df['Date'] = df['Year'].astype(str) + " " + df['Date']
df.Date = df.Date.astype('str')
enter = re.compile('\\n')
df.Date = df.Date.replace(enter, '')
df = df.replace(' ', '')

#Remove the old Year column
df = df.drop('Year', 1)

#turn each cell in Obs and Note into a list
df['Obs'] = df.Obs.str.split('|')
df['Note'] = df.Note.str.split('|')
df['contA'] = df.contA.str.split('|')

def add_spaces(cell):
	"""Add a space to string: before for the first item and after for the last item in the cell"""
	if isinstance(cell, list):
		cell[0] = str(" " + cell[0])
		cell[-1] = str((cell[-1]) + " ")
	return cell

#uniformize all list entries 
df = df.applymap(add_spaces) 

df.dropna(axis=0, thresh=2, inplace=True)
df.dropna(axis=0, how='all', inplace=True) # subset=['Date', 'Mood', 'Obs', 'Note', 'ContA'])

#format date and set as index
df.Date = df.Date.apply(lambda x: datetime.datetime.strptime(x, "%Y %B %d"))
df.set_index(df.Date, inplace=True)

#Numerize moods from 1 to 5, remove NaN and turn into int
df.Mood = df.Mood.replace(to_replace=['awful', 'pretty bad', 'meh', 'good', 'awesome'], value=[1, 2, 3, 4, 5])
df.Mood.fillna(0, inplace=True)
df.Mood = df.Mood.astype('int')

df['Note'].fillna('nan', inplace=True)
df['contA'].fillna('nan', inplace=True)

#turn each cell in Note and contA into a list
df['Note'] = df['Note'].apply(lambda cell: [cell])
df['contA'] = df['contA'].apply(lambda cell: [cell])

#turn Obs and Note into one column and get rid of Note
df['Obs'] = df['Obs'] + df['Note'] + df['contA']
df.drop(['Note', 'contA'], axis=1, inplace=True)

#print(df.iloc[0,2])
#print(len(df.iloc[0,2]))

#define listize
def listize(cell):
	"""turn list of lists into one list in each cell of Obs"""
	#print(cell)
	cell_list = []
	for sublist in cell:
		if isinstance(sublist, list):
			for item in sublist:
				cell_list.append(item)				
		elif isinstance(sublist, str):
			cell_list.append(sublist)		
						
	return cell_list

#turn Obs into a single list
df['Obs'] = df['Obs'].apply(listize) 

#Get dummies for visualization
dummy = pd.get_dummies(df.Obs.apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)
df = pd.concat([df, dummy], axis = 1)

#Remove meaningless columns
if 'nan' in list(df):
	df.drop(['nan'], axis=1, inplace=True)
if 'etc' in list(df):
	df.drop(['etc'], axis=1, inplace=True)


#QA
print(df.head(20))
print(df.info())
df.to_csv('daylio_export_tidy.csv')

#export tidy dataframe to csv
#df.to_csv('daylio_export_tidy.csv')
