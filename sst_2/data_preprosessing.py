import pandas as pd

# Read the parquet files
test = pd.read_parquet('test-00000-of-00001.parquet')
train = pd.read_parquet('train-00000-of-00001.parquet')
validation = pd.read_parquet('validation-00000-of-00001.parquet')

# Print the data
print("testing data: {}".format(test.shape))
print(test)
print("training data: {}".format(train.shape))
print(train)
print("validation data: {}".format(validation.shape))
print(validation)

# Write the 'text' and 'label' columns of the test dataset to 'test.txt'
test[['sentence', 'label']].to_csv('test.txt', index=False, sep='\t', header=False)

# Write the 'text' and 'label' columns of the valida dataset to 'valida.txt'
validation[['sentence', 'label']].to_csv('valida.txt', index=False, sep='\t', header=False)

# Write the 'text' and 'label' columns of the remaining train dataset to 'train.txt'
train[['sentence', 'label']].to_csv('train.txt', index=False, sep='\t', header=False)