import pandas as pd
import codecs
from io import StringIO
import sys
contents = codecs.open(sys.argv[1], encoding='utf-8').read()

contents = contents.replace('&', ',')
contents = contents.replace('\\','')
contents = contents.replace('textbf{','')
contents = contents.replace('}','')
contents = contents.replace('textpm',',')

data = StringIO(contents)
data = pd.read_csv(data, names=['dataset', 'spnll', 'spnll_std','rpll', 'rpll_std','spntime','spntime_std','rptime','rptime_std'])
print('mean')
print(data['spnll'].mean())
print(data['spnll_std'].mean())
print(data['rpll'].mean())
print(data['rpll_std'].mean())
print(data['spntime'].mean())
print(data['spntime_std'].mean())
print(data['rptime'].mean())
print(data['rptime_std'].mean())

print('std')
print(data['spnll'].var())
print(data['spnll_std'].var())
print(data['rpll'].var())
print(data['rpll_std'].var())
print(data['spntime'].var())
print(data['spntime_std'].var())
print(data['rptime'].var())
print(data['rptime_std'].var())

