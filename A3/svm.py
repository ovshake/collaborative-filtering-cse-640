import pandas as pd 
import numpy as np 
import json
import os 
from sklearn.model_selection import train_test_split , KFold
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score , mean_absolute_error



def convert_age_to_vector(age):

    vec = [0 , 0 , 0]
    if age == '' or age == 'nan':
        return vec
    age = int(age)
    index = -1
    if age < 1990:
        index = 0
    elif age < 2000:
        index= 1
    else:
        index = 2 
    vec[index] = 1
    return vec 
def convert_other_to_code(dic , thing , iter = False):
    vec = [0 for i in range(len(dic))]
    if iter:
        for t in thing:
            vec[dic[t]] = 1
        return vec 

     
    vec[dic[thing]] = 1 
    return vec 



data = pd.read_json('./data/ratings_correctFormat.json')
users = pd.read_csv('./data/users.csv')
items = pd.read_csv('./data/movies.csv')
old_movie_id = items['movie_id'].values
new_to_old_movie_id = {i : old_movie_id[i] for i in range(len(items))}
old_user_id = users['_id'].values
new_to_old_user_id = {i : old_user_id[i] for i in range(len(users))} 
old_to_new_user_id = {v:k for (k,v) in new_to_old_user_id.items()} 
old_to_new_movie_id  = {v:k for (k,v) in new_to_old_movie_id.items()}
rating_matrix = np.zeros((924 , 2850)) 
rating_list = [] 

for i in range(data.shape[0]):
    id = data.loc[i , '_id']
    new_id = old_to_new_user_id[id] 
    rating_dict = data.loc[i , 'rated']
    for k in rating_dict:
        if "s" in k:
            continue
        new_movie_id = old_to_new_movie_id[k] 
        rating_matrix[new_id][new_movie_id] = int(rating_dict[k][0])
        rating_list_tup = (new_id , new_movie_id ,int(rating_dict[k][0]) )
        rating_list.append(rating_list_tup)
        

genre = set() 
ratings = set()
release_year = set() 
for i in range(items.shape[0]):
    g = items.loc[i , 'genre']
    for u in g[1:-1].split(','):
        u = u.replace('"', '')
        u = u.replace(" ", '')
        genre.add(u) 
    date = str(items.loc[i,'released']).split("-")[0]
    release_year.add(date) 
    rt = items.loc[i,'rating'] 
    ratings.add(int(rt)) 
release_year.remove('nan')
rating_to_code = {int(g) : i for i,g in enumerate(ratings)}

genre_to_code = {i : g for i,g in enumerate(genre)} 
code_to_genre = {v : k for k,v in genre_to_code.items()} 


dobs = set() 
gender = set() 
jobs = set() 
languages = set() 
for i in range(users.shape[0]):
    year = str(users.loc[i , 'dob']).split('-')[-1]
    dobs.add(year) 
    gender.add(users.loc[i , 'gender']) 
    jobs.add(users.loc[i,'job'])
    l = users.loc[i , 'languages']
#     print(l)
    if type(l) != float:
        l = l[1:-1] 
        l = l.split(',')
        for v in l:
            v = v.replace('"', '')
            v = v.replace(" ", '')
            languages.add(v)
dobs.remove('nan') 
dobs.remove('')

code_to_job = {i : v for i,v in enumerate(jobs)}
job_to_code = {v : k for (k,v) in code_to_job.items()} 

code_to_lang = {i : v for i,v in enumerate(languages)}
lang_to_code = {v : k for (k,v) in code_to_lang.items()}

code_to_gender = {i : v for i,v in enumerate(gender)}
gender_to_code = {v : k for (k,v) in code_to_gender.items()}

user_vectors = {}

for i in range(users.shape[0]):
    year = str(users.loc[i , 'dob']).split('-')[-1]
    age_vector = [0 , 0 , 0] 
    age_vector = convert_age_to_vector(year)
    sex = users.loc[i , 'gender']
    work = users.loc[i,'job'] 
    sex_vector = convert_other_to_code(gender_to_code , sex)
    job_vector = convert_other_to_code(job_to_code , work)
    language_vector = [0  for i in range(len(languages))]
    l = users.loc[i , 'languages']
#     print(l)
    if type(l) != float:
        l = l[1:-1] 
        l = l.split(',')
        lang = []
        for v in l:
            v = v.replace('"', '')
            v = v.replace(" ", '')
            lang.append(v)
        language_vector = convert_other_to_code(lang_to_code, lang , iter = True) 

    user_vector = [] 
    user_vector.extend(language_vector)
    user_vector.extend(sex_vector)
    user_vector.extend(job_vector)
    user_vector.extend(age_vector)
    new_user_id = old_to_new_user_id[users.loc[i , '_id']] 
    user_vectors[new_user_id] = user_vector 



#User feature : languages , gender, job , age
print(genre_to_code)

movie_vectors = {} 
for i in range(items.shape[0]):
    g = items.loc[i , 'genre']
    movie_id = items.loc[i , 'movie_id']
    new_movie_id = old_to_new_movie_id[movie_id]
    genre_aux_array = []
    for u in g[1:-1].split(','):
        u = u.replace('"', '')
        u = u.replace(" ", '')
        genre_aux_array.append(u)
    genre_vector = convert_other_to_code(code_to_genre , genre_aux_array , iter=True)
    rt = items.loc[i,'rating'] 
    date = str(items.loc[i,'released']).split("-")[0]
    release_vector = convert_age_to_vector(date) 
    rating_vector = convert_other_to_code( rating_to_code, int(rt)) 
    movie_vector = []
    movie_vector.extend(genre_vector)
    movie_vector.extend(rating_vector) 
    movie_vector.extend(release_vector)
    movie_vectors[new_movie_id] = movie_vector



    # release_year.add(date) 
    # ratings.add(int(rt))

#Item Feature : genre, rating, release year 

total_X = [] 
total_y = [] 
for u in rating_list:
    user_id = u[0]
    mv_id = u[1] 
    rating = int(u[2])
    X = []
    # print(user_id)
    X.extend(user_vectors[user_id])
    X.extend(movie_vectors[old_to_new_movie_id[movie_id]])
    total_X.append(X) 
    total_y.append(rating)

total_X = np.asarray(total_X) 
total_y = np.asarray(total_y)
kf = KFold(n_splits=5)
for train_split , test_split in kf.split(total_X):
    X_train , X_test = total_X[train_split] , total_X[test_split] 
    y_train , y_test = total_y[train_split] , total_y[test_split] 
    clf = SVC(kernel = 'rbf' , verbose = True) 
    clf.fit(X_train , y_train) 
    y_pred = clf.predict(X_test) 
    score = mean_absolute_error(y_test , y_pred)
    print(score / 2)





# X_train , X_test, y_train , y_test = train_test_split(total_X , total_y, random_state = 42) 

#0.25 , 0.24 , 0.23 , 0.26 , 0.27
