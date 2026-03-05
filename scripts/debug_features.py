import importlib.util
import pandas as pd

# load preprocessing module
spec = importlib.util.spec_from_file_location("preproc", "src/preprocessing.py")
preproc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preproc)

# load feature_engineering module
spec2 = importlib.util.spec_from_file_location("fe", "src/feature_engineering.py")
fe = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(fe)

# run
if __name__ == '__main__':
    df = preproc.load_data('data/epl.csv')
    features = fe.calculate_team_stats(df)
    print('FEATURES HEAD:\n', features.head(20).to_string())
    print('\nDTYPES:\n', features.dtypes)
    print('\nDESCRIBE:\n', features.describe())
    n = len(features)
    for c in features.columns:
        zeros = int((features[c] == 0).sum())
        print(f'col {c} zeros: {zeros} / {n}')
