import numpy as np
import pandas as pd

# NumPy Basics
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", np.mean(arr))
print("Square:", arr ** 2)

# Array operations
matrix = np.arange(1, 10).reshape(3, 3)
print("Matrix:\n", matrix)
print("First row:", matrix[0])
print("Column 2:", matrix[:, 1])

# Statistics
data = np.random.normal(50, 10, 100)
print("Stats - Mean:", np.mean(data), "Std:", np.std(data))

# Pandas DataFrame
student_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Math': [85, 92, 78, 95],
    'Science': [88, 87, 82, 91]
}
df = pd.DataFrame(student_data)
print("DataFrame:\n", df)

# DataFrame operations
print("Mean scores:", df[['Math', 'Science']].mean())
print("High performers:", df[df['Math'] > 85])

# GroupBy
df['Department'] = ['Eng', 'Eng', 'Sci', 'Sci']
grouped = df.groupby('Department')[['Math', 'Science']].mean()
print("Department averages:\n", grouped)
