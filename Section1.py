
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
#%%

untokened_pandas = pd.read_csv("./data/stackexchange_812k.csv")
untokened_pandas

#%%

def remove_chars(regexp, df):
    regx=re.compile(regexp)
    df['text'] = df['text'].apply(lambda x: regx.sub("",x))
    return df

untokened_pandas = remove_chars("<*>",untokened_pandas)
untokened_pandas = remove_chars("\$*\$",untokened_pandas)
untokened_pandas = remove_chars("http\S+",untokened_pandas)
untokened_pandas = remove_chars("[0-9]",untokened_pandas)
untokened_pandas = remove_chars("\n",untokened_pandas)
untokened_pandas = remove_chars("@#$%^&*()_+<>;\'\"",untokened_pandas)
untokened_pandas = untokened_pandas[~untokened_pandas['text'].isna()]

#%%
tokened_pandas = untokened_pandas
tokened_pandas['text'] = tokened_pandas['text'].apply(lambda x: word_tokenize(x))

#%%

untokened_pandas.to_csv('random_file.csv')

#%%


