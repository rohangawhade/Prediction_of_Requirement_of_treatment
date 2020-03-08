import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing  # label encoding
from sklearn.model_selection import train_test_split  # splitting sample into training and test data set
from sklearn.preprocessing import MinMaxScaler  # Scaling values to same decimal places
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns

# install package mysqlclient
import MySQLdb

# reading survey.csv file
train_file = pd.read_csv("survey.csv")
# train_file.shape to find the shape of the dataframe --> print(train_file.shape)
# print(train_file.shape)
train_file.describe(include='all')

# count(train_file) R
# dropping unnecessary columns such as Timestamp state and comments from dataframe only(Not from csv file)
train_file = train_file.drop(['Timestamp'], axis=1)
train_file = train_file.drop(['state'], axis=1)
train_file = train_file.drop(['comments'], axis=1)
train_file.describe(include='all')

# number of null values in dataframe
# print(train_file.isnull().sum().max())

# assign default values to each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# create lists by data type
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                  'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                  'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                  'phys_health_interview',
                  'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                  'seek_help']
floatFeatures = []

# replace NaN with default values
for f in train_file:
    if f in intFeatures:
        train_file[f] = train_file[f].fillna(defaultInt)
    elif f in stringFeatures:
        train_file[f] = train_file[f].fillna(defaultString)
    elif f in floatFeatures:
        train_file[f] = train_file[f].fillna(defaultFloat)
    else:
        print('Feature %s is not present in file', f)
# for printing top 8 rows in train_file --> print(train_file.head(8))
# print(train_file.head(8))

# Cleaning gender to keep only male, female and trans
# gender = train_file['Gender'].str.lower()
# gender
# gender = train_file['Gender'].str.lower().unique()
# gender

# Creating Gender groups
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
            "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid",
             "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman",
             "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]

# replacing the gender values with male, female and trans
for (row, col) in train_file.iterrows():
    if str.lower(col.Gender) in male_str:
        train_file['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    if str.lower(col.Gender) in female_str:
        train_file['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    if str.lower(col.Gender) in trans_str:
        train_file['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

# removing garbage gender value('A little about you', 'p') from train_file
rem_gender = ['A little about you', 'p']
train_file = train_file[~train_file['Gender'].isin(rem_gender)]

# Displays cleaned gender column --> print(train_file['Gender'].unique())
# print(train_file['Gender'].unique())

# Replace missing age value with median value of age
train_file['Age'].fillna(train_file['Age'].median)
s = pd.Series(train_file['Age'])
# replacing age less than 18(such as -29 and other under aged values from train_file) with the median value of age
s[s < 18] = train_file['Age'].median()
train_file['Age'] = s
s = pd.Series(train_file['Age'])
# replacing age greater than 120(such as 999999 and other unrealistic age values from train_file) with the median value of age
s[s > 120] = train_file['Age'].median()
train_file['Age'] = s
train_file['age_range'] = pd.cut(train_file['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"],
                                 include_lowest=True)

# There are only 0.014% of self employed(Yes) so let's change NaN to NOT self_employed
# Replace "NaN" string from defaultString
train_file['self_employed'] = train_file['self_employed'].replace([defaultString], 'No')
train_file['self_employed'].unique()

# There are only 0.20% of self work_interfere so let's change NaN to "Don't know
# Replace "NaN" string from defaultString
train_file['work_interfere'] = train_file['work_interfere'].replace([defaultString], 'Don\'t know')
train_file['work_interfere'].unique()

# Encoding data
labelDict = {}
for f in train_file:
    le = preprocessing.LabelEncoder()
    le.fit(train_file[f])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_file[f] = le.transform(train_file[f])
    # getting labels
    labelKey = 'label_' + f
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

# next two statements were used to print the key value pairs
# for key, value in labelDict.items():
#    print(key, value)

# dropping country column from train_file
train_file = train_file.drop(['Country'], axis=1)
# checking whether Country has been dropped --> print(train_file.head(4))
# print(train_file.head(4))

# checking missing data is present or not
total = train_file.isnull().sum().sort_values(ascending=False)
percent = (train_file.isnull().sum() / train_file.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'percent'])
# print(missing_data.head(15))

# graphs for age
plt.figure(figsize=(15, 10))
sns.distplot(train_file['Age'], bins=24)
plt.title("Distribution and Density by Age")
plt.xlabel('Age')
plt.show()

plt.figure(figsize=(12, 8))
labels = labelDict['label_Gender']
g = sns.countplot(x="treatment", data=train_file)
g.set_xticklabels(labels)
plt.title('Number of people getting Treatment')
plt.show()

# Scaling all age values to be between 0-1
scaler = MinMaxScaler()
# formula for scaling --> Xnew=(Xi-Xmin)/(Xmax-Xmin)
train_file['Age'] = scaler.fit_transform(train_file[['Age']])
train_file.head()

# Splitting of data set
# trying different combinations of columns to add during prediction
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
# feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'anonymity', 'work_interfere']
# feature_cols = ['Age', 'Gender', 'self_employed', 'family_history', 'remote_work', 'tech_company',
#                'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor',
#               'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'benefits',
#               'care_options', 'anonymity', 'leave', 'work_interfere']
X = train_file[feature_cols]
Y = train_file.treatment

# splitting X and Y into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# Algorithm for prediction --> Boosting Algorithm: AdaBoost
def boosting():
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)
    boost.fit(X_train, y_train)
    # predictions for testing set
    y_pred_class = boost.predict(X_test)
    # print("y_test: ")
    # Printing the type of y_test --> print(type(y_test)) o/p --> <class 'pandas.core.series.Series'>
    # print(type(y_test))
    # Printing the type of y_pred_class  --> print(type(y_pred_class)) o/p --> <class 'numpy.ndarray'>
    # print(type(y_pred_class))
    accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
    # printing accuracy out of 100
    print("accuracy = ", accuracy_score * 100)


# calling boosting to check accuracy
boosting()

# writing the predicted o/p of test file into output.csv file
clf = AdaBoostClassifier()
clf.fit(X, Y)
dfTestPredictions = clf.predict(X_test)
results = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})
results.to_csv("output.csv", index=False)
# print(results)

'''
To take a single row input and predict
value = [(0.21, 1, 0, 1, 0, 1, 1, 4)]
sample_X_test = pd.DataFrame(value)
print(sample_X_test)

clf = AdaBoostClassifier()
clf.fit(X, Y)
dfTestPredictions = clf.predict(sample_X_test)
results = pd.DataFrame({'Index': sample_X_test.index, 'Treatment': dfTestPredictions})
results.to_csv("single_output.csv", index=False)
print(results)
'''
# saving the processed data of train_file into a csv file
train_file.to_csv("processed_data.csv", index=False)



# taking input from database
connection = MySQLdb.connect(host='localhost', user='root', passwd='', db='health')
cursor = connection.cursor()
cursor.execute("SELECT * FROM survey where username='jon'")
data = cursor.fetchall()
val = []
for row in data:
    for i in range(len(row)):
        val.append(row[i])
# print(val)
# commented values of variable give output as 0.
Age = val[0]
# Age = 70
# Gender = 'M'
Gender = val[1]
Country = val[2]
state = val[3]
self_employed = val[4]
family_history = val[5]
# family_history = 'yes'
# treatment = val[6]
treatment = 'yes'
work_interfere = val[7]
# work_interfere = 'Often'
no_employees = val[8]
remote_work = val[9]
tech_company = val[10]
benefits = val[11]
# benefits = 'yes'
care_options = val[12]
# care_options = 'yes'
wellness_program = val[13]
seek_help = val[14]
anonymity = val[15]
# anonymity = 'yes'
leave = val[16]
# leave = 'Very difficult'
mental_health_consequence = val[17]
phys_health_consequence = val[18]
coworkers = val[19]
supervisor = val[20]
mental_health_interview = val[21]
phys_health_interview = val[22]
mental_vs_physical = val[23]
obs_consequence = val[24]
comments = val[25]
username = val[26]

# data processing
# Scaling Age
Age = (Age - 18) / (72 - 18)

# Replacing Gender value with male/female/trans
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
            "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid",
             "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman",
             "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]
if Gender.lower() in male_str:
    Gender = 1
elif Gender.lower() in female_str:
    Gender = 0
else:
    Gender = 2

# Labelling family_history
# Note:- Labelling can does not imply the weightage of the value.
# It is only a tag
if family_history == 0:
    family_history = 0
else:
    family_history = 1

# Labelling benefits with 0, 1 and 2
if benefits == 'Yes' or benefits == 'yes':
    benefits = 2
elif benefits == 'No' or benefits == 'no':
    benefits = 1
else:
    benefits = 0

# Labelling care_options with 0, 1 and 2
if care_options == 'Yes' or care_options == 'yes':
    care_options = 2
elif care_options == 'No' or care_options == 'no':
    care_options = 1
else:
    care_options = 0

# Labelling anonymity with 0, 1 and 2
if anonymity == 'yes' or anonymity == 'Yes':
    anonymity = 2
elif anonymity == 'no' or anonymity == 'No':
    anonymity = 1
else:
    anonymity = 0

# Labelling leave with 4, 3, 2, 1 and 0
if leave == 'Very easy':
    leave = 4
elif leave == 'Very difficult':
    leave = 3
elif leave == 'Somewhat easy':
    leave = 2
elif leave == 'Somewhat difficult':
    leave = 1
else:
    leave = 0

# Labelling work_interfere with 4, 3, 2 and 1
if work_interfere == 'Often':
    work_interfere = 2
elif work_interfere == 'Rarely':
    work_interfere = 3
elif work_interfere == 'Never':
    work_interfere = 1
else:
    work_interfere = 4

# Choosing columns to use for prediction purpose
feature_cols = [Age, Gender, family_history, benefits, care_options, anonymity, leave, work_interfere]
feature_cols_df = pd.DataFrame(
    {'Age': [Age], 'Gender': [Gender], 'family_history': [family_history], 'benefits': [benefits],
     'care_options': [care_options], 'anonymity': [anonymity], 'leave': [leave], 'work_interfere': [work_interfere]})

# writing result into a new csv file
clf = AdaBoostClassifier()
clf.fit(X, Y)
dfTestPredictions = clf.predict(feature_cols_df)
# print(type(dfTestPredictions))  --> array
res = int(dfTestPredictions)  # array to int
# print(type(res)) --> int
print("Result = ", res)
if res:
    print("Should get a treatment")
else:
    print("Treatment is not necessary")
results = pd.DataFrame({'Index': feature_cols_df.index, 'Treatment': dfTestPredictions})
results.to_csv("output.csv", index=False)
# print(results)

# Storing the result into another result_table in database
connection = MySQLdb.connect(host='localhost', user='root', passwd='', db='health')
cursor = connection.cursor()
sql = "INSERT INTO result_table(username, result) values(%s, %s)"
val = (username, res)
cursor.execute(sql, val)
connection.commit()