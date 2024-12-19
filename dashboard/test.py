from pathlib import Path
import pandas as pd

dataframes: dict[str, pd.DataFrame] = {}
for file_path in Path("../Data/").glob("*.csv"):
    dataframes[file_path.with_suffix("").name] = pd.read_csv(file_path)

for name, dataframe in dataframes.items():
    print(f"{name}: {dataframe.describe(include="all")}")