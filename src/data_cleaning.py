import pandas as pd 

#load the data file form processing
data_file = pd.read_csv('./data/book1.csv')

#step1 handle the missing value
data_file.fillna('', inplace=True)

#step2 standardize and normalize data
data_file['character_name'] = data_file['character_name'].str.lower().str.strip()
data_file['traints'] = data_file['traits'].str.lower().str.strip()

#step3 normalize the supernatural powers if available

data_file['traits'] = data_file['traits'].replace({
    'has superpowers' : 'superpower',
    'flight' : 'can fly'
})

#step4 remove the duplicate value
data_file.drop_duplicates(inplace=True)

#step5 convert the comma-separate traints to list (if applicable)
data_file['traits'] = data_file['traits'].apply(lambda x: x.split(',') if isinstance(x, str) else x)

#save clean data to the file
data_file.to_csv('./data/cleaned_data.csv', index=False )

print("Data cleaing are done and store to the clean data file succesfully ")