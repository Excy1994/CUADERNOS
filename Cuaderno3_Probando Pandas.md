### Probando la hoja de trucos de Pandas
La biblioteca Pandas se basa en NumPy y proporciona estructuras de datos y herramientas de análisis de datos fáciles de usar para el lenguaje de programación Python.

### Pandas


```python
import numpy as np
import pandas as pd
```


```python
 s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
print(s)
```

    a    3
    b   -5
    c    7
    d    4
    dtype: int64
    

### DataFrame


```python
data = {'Country': ['Belgium', 'India', 'Brazil'],
 'Capital': ['Brussels', 'New Delhi', 'Brasília'],
 'Population': [11190846, 1303171035, 207847528]}

```


```python
df = pd.DataFrame(data,
 columns=['Country', 'Capital', 'Population'])
print(df)
```

       Country    Capital  Population
    0  Belgium   Brussels    11190846
    1    India  New Delhi  1303171035
    2   Brazil   Brasília   207847528
    

### I/O

### Read and Write to CSV


```python
pd.read_csv('file.csv', header=None, nrows=5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Capital</td>
      <td>Pais</td>
      <td>Continente</td>
    </tr>
    <tr>
      <th>1</th>
      <td>San Salvador</td>
      <td>El Salvador</td>
      <td>América</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nicaragua</td>
      <td>Managua</td>
      <td>América</td>
    </tr>
    <tr>
      <th>3</th>
      <td>España</td>
      <td>Madrid</td>
      <td>Europa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Japón</td>
      <td>Tokio</td>
      <td>Asia</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('myDataFrame.csv')
```

### Read and Write to Excel


```python
pd.read_excel('file.xlsx')
#pd.to_excel('dir/myDataFrame.xlsx', sheet_name='Sheet1')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Capital</td>
      <td>pais</td>
      <td>Continente</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>san salvador</td>
      <td>El salvador</td>
      <td>America</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nicaragua</td>
      <td>Managua</td>
      <td>America</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>España</td>
      <td>Madrid</td>
      <td>Europa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Japón</td>
      <td>Tokio</td>
      <td>Asia</td>
    </tr>
  </tbody>
</table>
</div>



#### Read multiple sheets from the same file


```python
#Leer varias hojas del mismo archivo
xlsx = pd.ExcelFile('file.xlsx')
print(xlsx)
#df = pd.read_excel(xlsx, 'Sheet1')
```

    <pandas.io.excel._base.ExcelFile object at 0x000002810C545460>
    

### Asking For Help


```python
 help(pd.Series.loc)
```

    Help on property:
    
        Access a group of rows and columns by label(s) or a boolean array.
        
        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.
        
        Allowed inputs are:
        
        - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``['a', 'b', 'c']``.
        - A slice object with labels, e.g. ``'a':'f'``.
        
          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included
        
        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - An alignable boolean Series. The index of the key will be aligned before
          masking.
        - An alignable Index. The Index of the returned selection will be the input.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)
        
        See more at :ref:`Selection by Label <indexing.label>`.
        
        Raises
        ------
        KeyError
            If any items are not found.
        IndexingError
            If an indexed key is passed and its index is unalignable to the frame index.
        
        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
            Series/DataFrame.
        Series.loc : Access group of values using labels.
        
        Examples
        --------
        **Getting values**
        
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=['cobra', 'viper', 'sidewinder'],
        ...      columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        
        Single label. Note this returns the row as a Series.
        
        >>> df.loc['viper']
        max_speed    4
        shield       5
        Name: viper, dtype: int64
        
        List of labels. Note using ``[[]]`` returns a DataFrame.
        
        >>> df.loc[['viper', 'sidewinder']]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8
        
        Single label for row and column
        
        >>> df.loc['cobra', 'shield']
        2
        
        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.
        
        >>> df.loc['cobra':'viper', 'max_speed']
        cobra    1
        viper    4
        Name: max_speed, dtype: int64
        
        Boolean list with the same length as the row axis
        
        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8
        
        Alignable boolean Series:
        
        >>> df.loc[pd.Series([False, True, False],
        ...        index=['viper', 'sidewinder', 'cobra'])]
                    max_speed  shield
        sidewinder          7       8
        
        Index (same behavior as ``df.reindex``)
        
        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]
               max_speed  shield
        foo
        cobra          1       2
        viper          4       5
        
        Conditional that returns a boolean Series
        
        >>> df.loc[df['shield'] > 6]
                    max_speed  shield
        sidewinder          7       8
        
        Conditional that returns a boolean Series with column labels specified
        
        >>> df.loc[df['shield'] > 6, ['max_speed']]
                    max_speed
        sidewinder          7
        
        Callable that returns a boolean Series
        
        >>> df.loc[lambda df: df['shield'] == 8]
                    max_speed  shield
        sidewinder          7       8
        
        **Setting values**
        
        Set value for all items matching the list of labels
        
        >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50
        
        Set value for an entire row
        
        >>> df.loc['cobra'] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50
        
        Set value for an entire column
        
        >>> df.loc[:, 'max_speed'] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50
        
        Set value for rows matching callable condition
        
        >>> df.loc[df['shield'] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0
        
        **Getting values on a DataFrame with an index that has integer labels**
        
        Another example using integers for the index
        
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=[7, 8, 9], columns=['max_speed', 'shield'])
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8
        
        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.
        
        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8
        
        **Getting values with a MultiIndex**
        
        A number of examples using a DataFrame with a MultiIndex
        
        >>> tuples = [
        ...    ('cobra', 'mark i'), ('cobra', 'mark ii'),
        ...    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
        ...    ('viper', 'mark ii'), ('viper', 'mark iii')
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20],
        ...         [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)
        >>> df
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36
        
        Single label. Note this returns a DataFrame with a single index.
        
        >>> df.loc['cobra']
                 max_speed  shield
        mark i          12       2
        mark ii          0       4
        
        Single index tuple. Note this returns a Series.
        
        >>> df.loc[('cobra', 'mark ii')]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64
        
        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.
        
        >>> df.loc['cobra', 'mark i']
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64
        
        Single tuple. Note using ``[[]]`` returns a DataFrame.
        
        >>> df.loc[[('cobra', 'mark ii')]]
                       max_speed  shield
        cobra mark ii          0       4
        
        Single tuple for the index with a single label for the column
        
        >>> df.loc[('cobra', 'mark i'), 'shield']
        2
        
        Slice from index tuple to single label
        
        >>> df.loc[('cobra', 'mark i'):'viper']
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36
        
        Slice from index tuple to index tuple
        
        >>> df.loc[('cobra', 'mark i'):('viper', 'mark ii')]
                            max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1
    
    

### Selection


### Getting 


```python
s['b']
```




    -5




```python
 df[1:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>



### Selecting, Boolean Indexing & Setting 

#### By Position


```python
#df.iloc([0],[0]) 
df.iloc[0,0]
```




    'Belgium'




```python
#df.iat([0],[0])
df.iat[0,0]
```




    'Belgium'



#### By Label


```python
 #df.loc([0], ['Country'])
df.loc[0, 'Country']
```




    'Belgium'




```python
#df.at([0], ['Country'])
df.at[0, 'Country'] 
```




    'Belgium'



#### By Label/Position


```python
#df.ix[2]
df.loc[2]
```




    Country          Brazil
    Capital        Brasília
    Population    207847528
    Name: 2, dtype: object




```python
#df.ix[:,'Capital'] 
df.loc[:,'Capital'] 
```




    0     Brussels
    1    New Delhi
    2     Brasília
    Name: Capital, dtype: object




```python
#df.ix[1,'Capital']
df.loc[1,'Capital'] 
```




    'New Delhi'



#### Boolean Indexing


```python
 s[~(s > 1)] 
```




    b   -5
    dtype: int64




```python
s[(s < -1) | (s > 2)] 
```




    a    3
    b   -5
    c    7
    d    4
    dtype: int64




```python
df[df['Population']>1200000000]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
  </tbody>
</table>
</div>



#### Setting


```python
s['a'] = 6
```

### Read and Write to SQL Query or Database Table


```python
 from sqlalchemy import create_engine
#engine = create_engine('sqlite:///:memory:')
#pd.read_sql("SELECT * FROM my_table;", engine)
#pd.read_sql_table('my_table', engine)
#pd.read_sql_query("SELECT * FROM my_table;", engine)
# pd.to_sql('myDf', engine)
```

### Dropping


```python
s.drop(['a', 'c']) 
```




    b   -5
    d    4
    dtype: int64




```python
df.drop('Country', axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>



### Sort & Rank


```python
df.sort_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by='Country')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rank()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



### Retrieving Series/DataFrame Information

#### Basic Information


```python
df.shape
```




    (3, 3)




```python
df.index
```




    RangeIndex(start=0, stop=3, step=1)




```python
df.columns
```




    Index(['Country', 'Capital', 'Population'], dtype='object')




```python
df.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   Country     3 non-null      object
     1   Capital     3 non-null      object
     2   Population  3 non-null      int64 
    dtypes: int64(1), object(2)
    memory usage: 200.0+ bytes
    


```python
df.count() 
```




    Country       3
    Capital       3
    Population    3
    dtype: int64



#### Summary


```python
df.sum()
```




    Country              BelgiumIndiaBrazil
    Capital       BrusselsNew DelhiBrasília
    Population                   1522209409
    dtype: object




```python
df.cumsum() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BelgiumIndia</td>
      <td>BrusselsNew Delhi</td>
      <td>1314361881</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BelgiumIndiaBrazil</td>
      <td>BrusselsNew DelhiBrasília</td>
      <td>1522209409</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.min()/df.max()
df.min()
```




    Country        Belgium
    Capital       Brasília
    Population    11190846
    dtype: object




```python
df.max()
```




    Country            India
    Capital        New Delhi
    Population    1303171035
    dtype: object




```python
#df.idxmin()/df.idxmax() 
```


```python
df.describe() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.074031e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.961346e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.119085e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.095192e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.078475e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.555093e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.303171e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean()
```




    Population    5.074031e+08
    dtype: float64




```python
df.median()
```




    Population    207847528.0
    dtype: float64



### Applying Functions


```python
 f = lambda x: x*2
print(f)
```

    <function <lambda> at 0x000002810CD9F0D0>
    


```python
df.apply(f)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BrazilBrazil</td>
      <td>BrasíliaBrasília</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.applymap(f) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BrazilBrazil</td>
      <td>BrasíliaBrasília</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>



### Data Alignment

#### Internal Data Alignment

Los valores NA se introducen en los índices que no se superponen.


```python
 s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
print(s3)
```

    a    7
    c   -2
    d    3
    dtype: int64
    


```python
s + s3
```




    a    13.0
    b     NaN
    c     5.0
    d     7.0
    dtype: float64



#### Arithmetic Operations with Fill Methods

También puede realizar la alineación de datos internos usted mismo con
la ayuda de los métodos de relleno


```python
s.add(s3, fill_value=0)
```




    a    13.0
    b    -5.0
    c     5.0
    d     7.0
    dtype: float64




```python
s.sub(s3, fill_value=2)
```




    a   -1.0
    b   -7.0
    c    9.0
    d    1.0
    dtype: float64




```python
s.div(s3, fill_value=4)
```




    a    0.857143
    b   -1.250000
    c   -3.500000
    d    1.333333
    dtype: float64




```python
s.mul(s3, fill_value=3)
```




    a    42.0
    b   -15.0
    c   -14.0
    d    12.0
    dtype: float64



### - Hoja de trucos de Pandas #2

### Cambiar la forma de los datos


```python
import pandas as pd
###Declaracion de df
df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],
                  'population': [1864, 22000, 80000]},
                  index=['panda', 'polar', 'koala'])
print(df)
```

             species  population
    panda       bear        1864
    polar       bear       22000
    koala  marsupial       80000
    


```python
data = {'Date': ['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type': ['a','b','c','a','a','c'],
        'Value': [11.432,13.031,20.784,99.906,1.303,20.784]}
print(data)
```

    {'Date': ['2016-03-01', '2016-03-02', '2016-03-01', '2016-03-03', '2016-03-02', '2016-03-03'], 'Type': ['a', 'b', 'c', 'a', 'a', 'c'], 'Value': [11.432, 13.031, 20.784, 99.906, 1.303, 20.784]}
    


```python
df2 = pd.DataFrame(data,
 columns=['Date', 'Type', 'Value'])

print(df2)
```

             Date Type   Value
    0  2016-03-01    a  11.432
    1  2016-03-02    b  13.031
    2  2016-03-01    c  20.784
    3  2016-03-03    a  99.906
    4  2016-03-02    a   1.303
    5  2016-03-03    c  20.784
    


```python
df3= df2.pivot(index='Date',
              columns='Type',
              values='Value')
print(df3)
```

    Type             a       b       c
    Date                              
    2016-03-01  11.432     NaN  20.784
    2016-03-02   1.303  13.031     NaN
    2016-03-03  99.906     NaN  20.784
    

### Pivot table
#### Tabla dinámica


```python
df4= pd.pivot_table(df2,
                   values='Value',
                   index='Date',
                   columns='Type')
print(df4)
```

    Type             a       b       c
    Date                              
    2016-03-01  11.432     NaN  20.784
    2016-03-02   1.303  13.031     NaN
    2016-03-03  99.906     NaN  20.784
    

### Stack/ Unstack
#### Apilar/desapilar


```python
  #stacked = df5/stack()
#staked.unstaked
#print(stacked)
```

### Melt


```python
pd.melt(df2,
        id_vars=["Date"],
        value_vars=["Type","Value"],
        value_name="Observations")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>variable</th>
      <th>Observations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-01</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>Type</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>Type</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-03</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>Type</td>
      <td>c</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-03-01</td>
      <td>Value</td>
      <td>11.432</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016-03-02</td>
      <td>Value</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016-03-01</td>
      <td>Value</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016-03-03</td>
      <td>Value</td>
      <td>99.906</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2016-03-02</td>
      <td>Value</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016-03-03</td>
      <td>Value</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



### Iteration


```python
df.iteritems()
```




    <generator object DataFrame.iteritems at 0x000002810CDD0CF0>




```python
df.iterrows()
```




    <generator object DataFrame.iterrows at 0x000002810CDD0EB0>



### Advanced Indexing

#### Selecting
##### Seleccionar


```python
df3.loc[:,(df3>1).any()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
      <td>13.031</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.loc[:,(df3>1).all()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.loc[:,df3.isnull().any()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>13.031</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.loc[:,df3.notnull().all()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
    </tr>
  </tbody>
</table>
</div>



#### indexing with isin


```python
#df[(df.Country.isin(df2.Type))]
```


```python
#df3.filter(items="a","b")
df3.filter(items="a, b")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.select(lambda x: not x%5)

```

#### Where


```python
s.where(s > 0)
```




    a    6.0
    b    NaN
    c    7.0
    d    4.0
    dtype: float64



#### Query


```python
#df6.query('second > first')
```

### Setting/Resetting index
#### Índice de configuración / restablecimiento


```python
#df.set_index('Country')
```


```python
df4 = df.reset_index()
print(df4)
```

       index    species  population
    0  panda       bear        1864
    1  polar       bear       22000
    2  koala  marsupial       80000
    


```python
df = df.rename(index=str,
               columns={"Country":"cntry",
                        "Capital":"cptl",
                        "Population":"ppltn"})
print(df)
```

             species  population
    panda       bear        1864
    polar       bear       22000
    koala  marsupial       80000
    

### Reindexing


```python
s2 = s.reindex(['a','c','d','e','b'])
print(s2)
```

    a    6.0
    c    7.0
    d    4.0
    e    NaN
    b   -5.0
    dtype: float64
    

#### Forward Filling


```python
#df.reindex(range(4),
          # method='fill')
```

#### Backward Filling


```python
#s3 = s.reindex(range (5),
          # method='bfill')
#print(s3)
```

### Multilndexing


```python
arrays = [np.array([1,2,3]),
          np.array([5,4,3])]
df5= pd.DataFrame(np.random.rand(3, 2), index=arrays)
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples,
                                  names=['first', 'second'])
df6= pd.DataFrame(np.random.rand(3, 2), index=index)
df2.set_index(["Date", "Type"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Value</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <th>a</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <th>b</th>
      <td>13.031</td>
    </tr>
    <tr>
      <th>2016-03-01</th>
      <th>c</th>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <th>a</th>
      <td>99.906</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <th>a</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <th>c</th>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



### Duplicate Data


```python
s3.unique()
```




    array([ 7, -2,  3], dtype=int64)




```python
df2.duplicated('Type')
```




    0    False
    1    False
    2    False
    3     True
    4     True
    5     True
    dtype: bool




```python
df2.drop_duplicates('Type', keep='last')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Type</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>b</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>a</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index.duplicated()
```




    array([False, False, False])



### Grouping Data

#### Aggregation


```python
df2.groupby(by=['Date','Type']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Value</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2016-03-01</th>
      <th>a</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>c</th>
      <td>20.784</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2016-03-02</th>
      <th>a</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>b</th>
      <td>13.031</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2016-03-03</th>
      <th>a</th>
      <td>99.906</td>
    </tr>
    <tr>
      <th>c</th>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4.groupby(level=0).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df4.groupby(level=0).agg({'a':lambda x:sum(x)/len(x),
                        #  'b': np.sum})
```

#### Transformation


```python
customSum = lambda x: (x+x%2)
df4.groupby(level=0).transform(customSum)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80000</td>
    </tr>
  </tbody>
</table>
</div>



### Missing Data


```python
df.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>panda</th>
      <td>bear</td>
      <td>1864</td>
    </tr>
    <tr>
      <th>polar</th>
      <td>bear</td>
      <td>22000</td>
    </tr>
    <tr>
      <th>koala</th>
      <td>marsupial</td>
      <td>80000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.fillna(df3.mean())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
      <td>13.031</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
      <td>13.031</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
      <td>13.031</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.replace("a","f")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Type</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-01</td>
      <td>f</td>
      <td>11.432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>b</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-03</td>
      <td>f</td>
      <td>99.906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>f</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



### Combining Data


```python
data1 = pd.DataFrame({'X1': ['a','b','c'], 'X2': [11.432,1.303, 99.906]}); data1
print(data1)
data2 = pd.DataFrame({'X1': ['a','b','d'], 'X3': [20.78,"NaN", 20.784]}); data2
print(data2)
```

      X1      X2
    0  a  11.432
    1  b   1.303
    2  c  99.906
      X1      X3
    0  a   20.78
    1  b     NaN
    2  d  20.784
    

### Merge


```python
pd.merge(data1,
         data2,
        how='left',
        on='X1')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>99.906</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(data1,
         data2,
        how='right',
        on='X1')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(data1,
         data2,
        how='inner',
        on='X1')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(data1,
         data2,
        how='outer',
        on='X1')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>99.906</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



### Join


```python
#data1.join(data2, how='right')
```

### Concatenate

#### Vertical


```python
s.append(s2)
```




    a    6.0
    b   -5.0
    c    7.0
    d    4.0
    a    6.0
    c    7.0
    d    4.0
    e    NaN
    b   -5.0
    dtype: float64



#### Horizontal/Vertical


```python
pd.concat([s,s2],axis=1, keys=['One','Two'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>One</th>
      <th>Two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-5.0</td>
      <td>-5.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>e</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([data1, data2], axis=1, join='inner')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X1</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>a</td>
      <td>20.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>99.906</td>
      <td>d</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



### Dates


```python
#df2['Date']= pd.to_datetime(df2['Date'])
#df2['Date']= pd.date_range('2000-1-1',
                          #  periods=6,
                          #  freq='M')
#dates = [datetime(2012,5,1), datetime(2012,5,2)]
#index = pd.DatetimeIndex(dates)
#index = pd.date_range(datetime(2012,2,1), end, freq='BM')
```

### Visualization


```python
import matplotlib.pyplot as plt
s.plot()
print(s)
```

    Matplotlib is building the font cache; this may take a moment.
    

    a    6
    b   -5
    c    7
    d    4
    dtype: int64
    


    
![png](output_151_2.png)
    



```python
df2.plot()
plt.show()
```


    
![png](output_152_0.png)
    



```python

```
