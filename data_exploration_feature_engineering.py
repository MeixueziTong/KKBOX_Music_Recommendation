#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:33:58 2017

@author: meixuezi
"""

# Some usual imports here
import os
import csv as csv 
import numpy as np
import pandas as pd
import missingno as msno # for missing data visualisation
import lightgbm as lgb
import math
import gc
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

os.chdir('/Users/meixuezi/desktop/Projects/KKBOX_music_recommendation')

# load data
members = pd.read_csv('members.csv',parse_dates = [5,6]) # parse the date columns
song_extra_info = pd.read_csv('song_extra_info.csv')
songs = pd.read_csv('songs.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# merge data
# merge training data 
song_merged = pd.merge(songs,song_extra_info,how = 'outer', on = 'song_id') # song info 
train_song_merged = pd.merge(song_merged, train, how = 'right', on = 'song_id') # song member pairing info 
train_merged = pd.merge(train_song_merged, members, how = 'right', on = 'msno') # member/user info
print('training data info:')
print(train_merged.shape)
print(train_merged.head(5))
# merge test data
test_song_merged = pd.merge(song_merged, test, how = 'right', on = 'song_id')
test_merged = pd.merge(test_song_merged, members, how = 'left', on = 'msno')
print('test data info:')
print(test_merged.shape)
print(test_merged.head(5))




### 1. visualise missing data for each column
plt.figure(1) # visualize missing data for training set
msno.bar(train_merged[train_merged.columns[train_merged.isnull().any()].tolist()],
         figsize=(20,8),
         color='green',
         fontsize=18,
         labels=True)

plt.figure(2) # visualize missing data for test set
fig1 = msno.bar(test_merged[test_merged.columns[test_merged.isnull().any()].tolist()],
         figsize=(20,8),
         color='blue',
         fontsize=18,
         labels=True)

plt.figure(3) # nullity correlation heatmap for training data
msno.heatmap(train_merged[train_merged.columns[train_merged.isnull().any()].tolist()],
         figsize=(20,8),
         fontsize=18,
         labels=True)

plt.figure(4) # nullity correlation heatmap for test data
msno.heatmap(test_merged[test_merged.columns[test_merged.isnull().any()].tolist()],
         figsize=(20,8),
         fontsize=18,
         labels=True)

# fill NA with mean
train_merged.target.fillna(0, inplace=True)
train_merged.bd.fillna(train_merged.bd.mean())
test_merged.bd.fillna(test_merged.bd.mean())
train_merged.song_length.fillna(train_merged.song_length.mean(),inplace=True)
test_merged.song_length.fillna(test_merged.song_length.mean(),inplace=True)



### 2. Explore Users' data

# transform male , female and nan as integer 0, 1 and 2 
def is_male0_female1(gender):
    if gender == 'male':
        return 0
    elif gender == 'female':
        return 1
    elif gender == 'nan':
        return 2

train_merged['is_male0_female1'] = train_merged['gender'].apply(is_male0_female1)

users = train_merged.groupby(['msno']).agg(['mean','count'])

# User Music Listening Event counts

plt.figure(5)
plt.hist(np.log10(users[('registered_via','count')]), 100)
plt.title("Users' Music Listening Event Counts")
plt.xlabel('Log Values of Music Counts')
plt.ylabel('User Counts')
plt.savefig('figures/user_music_listening event counts.png')

users2 = users.copy()
index = users2[('registered_via','count')]==users2[('registered_via','count')].max()

# User age
age = users[('bd','mean')]
age_no0 = age[age != 0] # filter out age = 0 data
plt.figure(6)
plt.hist(age_no0,50,[0,100]) # plot a histgram
plt.title("Users' Age")
plt.xlabel('Age')
plt.ylabel('User Counts')
plt.savefig('figures/user_age_distribution.png')

# User gender

gender = users[('is_male0_female1','mean')]
plt.figure(7)
gender[gender != 2].value_counts().plot('bar',width = 0.2, color = ('red','green')) # filter out nan data

plt.title("Users' Gender")
plt.xticks([0,1],['Male', 'Female'], rotation = 'horizontal')
plt.ylabel('User Counts')
plt.savefig('figures/user_gender.png')

# correlation of gender and age and music listening occurrence counts

# maskout gender = nan, age = 0 and age >= 100 
user_gender_age = users[(users[('is_male0_female1','mean')] != 2)& 
                        (users[('bd','mean')] != 0) & 
                        (users[('bd','mean')] < 100)]


plt.figure(8)
male = user_gender_age[user_gender_age[('is_male0_female1','mean')] == 0]
female = user_gender_age[user_gender_age[('is_male0_female1','mean')] == 1]

plt.hist(np.log10(male[('registered_via','count')]), 100, alpha=0.5, label='Male')
plt.hist(np.log10(female[('registered_via','count')]), 100, alpha=0.5, label='Female')
plt.legend(loc='upper right')
plt.title("Users' Music Listening Occurrence Counts")
plt.xlabel('Log Values of Music Listening Occurrence Counts')
plt.ylabel('User Counts')

plt.savefig('figures/gender_specific_listening_occurrance.png')


plt.figure(9)
plt.hist(male[('bd','mean')], 100, alpha=0.5, label='Male')
plt.hist(female[('bd','mean')], 100, alpha=0.5, label='Female')
plt.legend(loc='upper right')
plt.title("Users' Age")
plt.xlabel('Age')
plt.ylabel('User Counts')

plt.savefig('figures/gender_age_distribution.png')




### 3. Explore Music Data

# add song_year variable
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

train_merged['song_year'] = train_merged['isrc'].apply(isrc_to_year)
test_merged['song_year'] = test_merged['isrc'].apply(isrc_to_year)

# song, artist, composer, lyricist
songs = train_merged.groupby('song_id').agg(['mean','count','sum'])
artists = train_merged.groupby('artist_name').agg(['mean','count','sum'])
composers = train_merged.groupby('composer').agg(['mean','count','sum'])
lyricists = train_merged.groupby('lyricist').agg(['mean','count','sum'])
# 'target': replay True or False, 1 or 0
# ('target', 'count') total play counts
# ('target', 'sum') total replay counts

# print song summary
print('song played summary')
print(songs[('target','count')].describe())
print(songs[('target','sum')].describe())

# top 10 most played
songs_top10 = songs[('target','count')].nlargest(10)
artists_top10 = artists[('target','count')].nlargest(10)
composers_top10 = composers[('target','count')].nlargest(10)
lyricists_top10 = lyricists[('target','count')].nlargest(10)
top10 = pd.DataFrame({'song_id':songs_top10.index,
                      'song_id_play_counts':songs_top10.values,
                   'artist':artists_top10.index,
                   'artist_play_counts':artists_top10.values,
                   'composer':composers_top10.index,
                   'composer_play_counts':composers_top10.values,
                   'lyricist':lyricists_top10.index,
                   'lyricist_play_counts':lyricists_top10.values
                   })

top10.to_csv('top10_most_played.csv', index = False)


plt.figure(10)
plt.hist(np.log10(songs[('target','count')]),20, alpha = 0.5, color = 'green')
plt.ylabel('Log Value of Music Played Counts')
plt.title('Music Played Counts')
plt.savefig('figures/song_play_count_distribution.png')

plt.figure(11)
plt.hist(np.log10(artists[('target','count')]),20, alpha = 0.5, color = 'blue')
plt.ylabel('Log Value of Artist Played Counts')
plt.title('Artist Played Counts')
plt.savefig('figures/artist_play_count_distribution.png')


# correlation between song/composer/artist/lyricist popularities and replay chances
song_repeat_chance = songs[('target','sum')]/songs[('target','count')]
artist_repeat_chance = artists[('target','sum')]/artists[('target','count')]
composer_repeat_chance = composers[('target','sum')]/composers[('target','count')]
lyricist_repeat_chance = lyricists[('target','sum')]/lyricists[('target','count')]

plt.figure(12)
f, axes = plt.subplots(2,2, sharey = True)
axes[0,0].scatter(songs[('target','count')],song_repeat_chance, alpha = 0.3, color = 'red')
axes[0,0].set_xlabel('Number of songs played', fontsize = 7)
axes[0,0].set_ylabel('Chance of repeat listens')

axes[0,1].scatter(artists[('target','count')],artist_repeat_chance, alpha = 0.3, color = 'blue')
axes[0,1].set_xlabel('Number of artists played', fontsize = 7)

axes[1,0].scatter(composers[('target','count')],composer_repeat_chance, alpha = 0.3, color = 'orange')
axes[1,0].set_xlabel('Number of composers played', fontsize = 7)
axes[1,0].set_ylabel('Chance of repeat listens')

axes[1,1].scatter(lyricists[('target','count')],lyricist_repeat_chance, alpha = 0.3, color = 'green')
axes[1,1].set_xlabel('Number of lyricists played', fontsize = 7)

plt.savefig('figures/popularity_replay_chance.png')



# song year, language and replay chances


song_year_lang = pd.DataFrame({
                            'year': songs[('song_year','mean')],
                            'chances': song_repeat_chance,
                            'year_counts':songs[('song_year','count')],
                            'language':songs[('language','mean')],
                            'language_counts':songs[('language','count')],
                            'repeat_counts': songs[('target','sum')
                            ]})
song_groupby_year = song_year_lang.groupby('year').agg(['count','mean','sum'])
song_groupby_language = song_year_lang.groupby('language').agg(['count','mean','sum'])


plt.figure(13)
plt.ylabel('Replay Average Counts')
plt.xlabel('Language Code')
song_groupby_language[('chances', 'mean')].plot('bar',color= 'red', alpha = 0.5) # to splice secondary index using (primary index, secondary index)
plt.savefig('figures/language_replay_counts')

plt.figure(14)
plt.ylabel('Total Play Counts')
plt.xlabel('Language Code')
song_groupby_language[('chances', 'count')].plot('bar',color= 'red', alpha = 0.5)
plt.savefig('figures/language_replay_total_counts')

plt.figure(15)
plt.ylabel('Replay Average Counts')
plt.xlabel('Year')
song_groupby_year[('chances', 'mean')].plot('line',color= 'blue', alpha = 0.5)
plt.savefig('figures/year_replay_average_counts')

plt.figure(16)
plt.ylabel('Total Play Counts')
plt.xlabel('Year')
song_groupby_year[('chances', 'count')].plot('line',color= 'blue', alpha = 0.5)
plt.savefig('figures/year_total_play_counts')








### 4. feature engineering: transfrom features/ add new features

# add membership time
train_registration_time = train_merged.expiration_date-train_merged.registration_init_time
train_registration_time = train_registration_time.dt.days # convert dtype to days

test_registration_time = test_merged.expiration_date-test_merged.registration_init_time
test_registration_time = test_registration_time.dt.days

train_merged['membership_days'] = train_registration_time
test_merged['membership_days'] = test_registration_time

train_merged['registration_year'] = train_merged['registration_init_time'].dt.year
train_merged['registration_month'] = train_merged['registration_init_time'].dt.month
train_merged['registration_day'] = train_merged['registration_init_time'].dt.day

test_merged['registration_year'] = test_merged['registration_init_time'].dt.year
test_merged['registration_month'] = test_merged['registration_init_time'].dt.month
test_merged['registration_day'] = test_merged['registration_init_time'].dt.day

train_merged.drop('registration_init_time', axis =1)
test_merged.drop('registration_init_time', axis =1)


# add genre_id_count
def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') +1
    
train_merged['genre_ids'].fillna('no_genre_id', inplace = True)
test_merged['genre_ids'].fillna('no_genre_id', inplace = True)

train_merged['genre_ids_count'] = train_merged['genre_ids'].apply(genre_id_count).astype(np.int8)
test_merged['genre_ids_count'] = test_merged['genre_ids'].apply(genre_id_count).astype(np.int8)

# add lyricist_count

def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';','+'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';','+']))

train_merged['lyricist'].fillna('no_lyricist', inplace = True)
test_merged['lyricist'].fillna('no_lyricist', inplace = True)
train_merged['lyricists_count'] = train_merged['lyricist'].apply(lyricist_count).astype(np.int8)
test_merged['lyricists_count'] = test_merged['lyricist'].apply(lyricist_count).astype(np.int8)

# add composer_count

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|','/','\\',';','+'])) + 1

train_merged['composer'].fillna('no_composer', inplace = True)
test_merged['composer'].fillna('no_composer', inplace = True)
train_merged['composers_count'] = train_merged['composer'].apply(composer_count).astype(np.int8)
test_merged['composers_count'] = test_merged['composer'].apply(composer_count).astype(np.int8)

# add artist_count
def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('+') + x.count('|') + x.count('&') + x.count('feat') 

train_merged['artist_name'].fillna('no_artist', inplace = True)
test_merged['artist_name'].fillna('no_artist', inplace = True)
train_merged['artists_count'] = train_merged['artist_name'].apply(artist_count).astype(np.int8)
test_merged['artists_count'] = test_merged['artist_name'].apply(artist_count).astype(np.int8)

# if artist is the same as composer
train_merged['artist_composer'] = (train_merged['artist_name'] == train_merged['composer']).astype(np.int8)
test_merged['artist_composer'] = (test_merged['artist_name'] == test_merged['composer']).astype(np.int8)

# if artist, lyricist and composer are all the same

train_merged['artist_composer_lyricist'] = ((train_merged['artist_name'] == train_merged['composer'])&
            (train_merged['artist_name'] == train_merged['lyricist'])&
            (train_merged['composer'] == train_merged['lyricist'])).astype(np.int8)
test_merged['artist_composer_lyricist'] = ((test_merged['artist_name'] == test_merged['composer'])&
            (test_merged['artist_name'] == test_merged['lyricist'])&
            (test_merged['composer'] == test_merged['lyricist'])).astype(np.int8)

# add song_popularity
_dict_count_song_played_train = {k: v for k, v in train_merged['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test_merged['song_id'].value_counts().iteritems()}

def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0
train_merged['count_song_played'] = train_merged['song_id'].apply(count_song_played).astype(np.int64)
test_merged['count_song_played'] = test_merged['song_id'].apply(count_song_played).astype(np.int64)

# add artist_popularity
_dict_count_artist_played_train = {k:v for k, v in train_merged['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k:v for k, v in test_merged['artist_name'].value_counts().iteritems()}

def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0

train_merged['count_artist_played'] = train_merged['artist_name'].apply(count_artist_played).astype(np.int64)
test_merged['count_artist_played'] = test_merged['artist_name'].apply(count_artist_played).astype(np.int64)
# change data type
train_merged[['target','bd']] = train_merged[['target','bd']].astype(np.uint8) # bd: age, target: 1 or 0 (True or False)
train_merged.song_length = train_merged.song_length.astype(np.uint32)

import pickle as pkl
with open('train_merged.pkl', 'wb') as f:
    pkl.dump(train_merged, f)
    
with open('test_merged.pkl', 'wb') as f:
    pkl.dump(test_merged, f)








