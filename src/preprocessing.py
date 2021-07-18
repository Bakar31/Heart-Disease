from input import *

# creating dependent and independent matrix of features
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# feature scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x = pd.DataFrame(scale.fit_transform(x))

scaled_df = pd.concat([x, y], axis = 1)

# feature engineering
scaled_df['feature1'] = ((scaled_df[0] + scaled_df[1])**2)/5
scaled_df['feature2'] = ((scaled_df[1]*2.5 + scaled_df[2])**3)/25
scaled_df['feature3'] = ((scaled_df[0]**3) + (scaled_df[1]**2))/10
scaled_df['feature4'] = ((scaled_df[3]**3 + scaled_df[4])*2)/10.5
scaled_df['feature5'] = ((scaled_df[5]*5.7 + scaled_df[6])**2)
scaled_df['feature6'] = (scaled_df[9]*1.5 + (scaled_df[11]*3.6))
scaled_df['feature7'] = ((scaled_df[2]**2 + scaled_df[10])*2)/5
scaled_df['feature8'] = ((scaled_df[6]**3 + scaled_df[7])*7)/2.5
scaled_df['feature9'] = ((scaled_df[7]*3/10) + (scaled_df[8]**2)/5.9)
scaled_df['feature10'] = ((scaled_df[12]*3/11 + scaled_df[4])**2)/10

# checking correlation
correlation = scaled_df.corr()
correlation.sort_values(["target"], ascending = False, inplace = True)
#print(correlation.target)

# feature selection
less_important = ['feature6', 9, 8, 11, 12, 1, 0, 'feature4', 3, 4, 'feature3', 5, 'feature5']
scaled_df.drop(less_important, axis = 1, inplace = True)

# create scaled independent matrix of features
scaled_x = scaled_df.iloc[:, :-1]

# create training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_x,y, test_size = 0.25, random_state = 31)