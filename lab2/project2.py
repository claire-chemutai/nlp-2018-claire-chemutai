
# coding: utf-8

# In[1]:


def min_edit(source, target):
    n=len(source)
    m=len(target)
    D=[[0 for x in range(m+1)] for x in range(n+1)] 
    for i in range(n+1):
        for j in range(m+1):
            if i==0:
                D[i][j]=j
            elif j==0:
                D[i][j]=i
            elif source[i-1]==target[j-1]:
                D[i][j]=D[i-1][j-1]
            else:   
                D[i][j]=1+min(D[i-1][j],
                            D[i-1][j-1],
                            D[i][j-1])
    return D[n][m]
source=input("Enter the source:")
target=input("Enter the target:")
min_edit(source, target)
    

    

