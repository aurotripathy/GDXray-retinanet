import pandas as pd 

df = pd.read_csv('annotations/train_annotations.csv', 
                 delimiter=',',
                 header=None, 
                 names=['file', 'x1', 'y1', 'x2', 'y2', 'class'])
print('Total lines', df.shape)
files = df['file'].values
print(len(set(files)))
