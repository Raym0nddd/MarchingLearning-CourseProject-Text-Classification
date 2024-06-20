import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split


# Read the parquet files
all = pd.read_parquet('0000.parquet')

# Split the data into train, validation, and test sets
train, val_test = train_test_split(all, test_size=0.3, random_state=42)
val, test = train_test_split(val_test, test_size=0.33, random_state=42)

# Write the data to txt files
train.to_csv('train.txt', header=False, index=False, columns=['text', 'label'], sep='\t')
val.to_csv('dev.txt', header=False, index=False, columns=['text', 'label'], sep='\t')
test.to_csv('test.txt', header=False, index=False, columns=['text', 'label'], sep='\t')