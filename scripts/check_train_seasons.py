import os, sys
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
import preprocessing

train_dir='data/data_2011_2023'
files=sorted([f for f in os.listdir(train_dir) if f.endswith('.csv')])
print(f"files_count={len(files)}")
for f in files:
    print(f)

try:
    df=preprocessing.load_multiple_seasons(train_dir)
    print(f"\nrows={len(df)}")
    if 'Season' in df.columns:
        seasons=sorted(df['Season'].dropna().unique().tolist())
        print(f"unique_seasons_count={len(seasons)}")
        print(seasons)
    else:
        print('No Season column')
except Exception as e:
    print('ERROR', e)
