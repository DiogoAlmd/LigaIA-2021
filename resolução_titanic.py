#!/usr/bin/env python
# coding: utf-8

# ## Desafio Titanic, resolução

# In[2]:


from IPython.display import Image
Image(url = "https://miro.medium.com/max/2560/0*1pQ4LZENtCzGE6wK.jpg")


# ### Coletando dados:

# In[5]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

titanic = pd.read_csv('titanic.data')
titanic.head(80)


# ## Dicionário:

# In[6]:


# PassengerId: Número de identificação do passageiro
# Survived: Informa se o passageiro sobreviveu ao desastre
# 0 = Não
# 1 = Sim
# Pclass: Classe do bilhete
# 1 = 1ª Classe
# 2 = 2ª Classe
# 3 = 3ª Classe
# Name: Nome do passageiro
# Sex: Sexo do passageiro
# Age: Idade do passageiro
# SibSp: Quantidade de cônjuges e irmãos a bordo
# Parch: Quantidade de pais e filhos a bordo
# Ticket: Número da passagem
# Fare: Preço da Passagem
# Cabin: Número da cabine do passageiro
# Embarked: Porto no qual o passageiro embarcou
# C = Cherbourg
# Q = Queenstown
# S = Southampton


# In[7]:


titanic.head()


# In[8]:


titanic.shape


# In[9]:


titanic.info()


# In[10]:


titanic.isnull().sum()


# In[11]:


#tratamento para dados ausentes ou faltantes:

titanic.isna().sum()


# In[12]:


titanic.fillna({'Age':titanic.Age.mean()}, inplace=True)
titanic.isna().sum()


# ## Gráficos para características:

# In[13]:


# Afim de analisar os dados, serão criados gráficos de barra para isso.

def grafico_barras(caracteristica):
    sobreviveu = titanic[titanic['Survived']==1][caracteristica].value_counts()
    morreu = titanic[titanic['Survived']==0][caracteristica].value_counts()
    df = pd.DataFrame([sobreviveu, morreu])
    df.index = ['Sobreviveu','Morreu']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[14]:


grafico_barras('Sex')


# In[ ]:


# O gráfico mostra que mais mulheres sobreviveram em comparação aos homens.


# In[15]:


grafico_barras('Pclass')


# In[ ]:


# Este gráfico confirma que pessoas de primeira classe tiveram mais chances de sobreviver que pessoas de terceira classe.


# In[16]:


grafico_barras('SibSp')


# In[ ]:


# Já este permite confirmar que pessoas acompanhadas de mais de 2 irmãs/irmãos ou esposa tem mais chance de sobreviver
# e também que uma pessoa que nãp está acompanhada de irmãs/irmãos ou esposa tem mais chances de morrer.


# In[17]:


grafico_barras('Parch')


# In[ ]:


# O gráfico confirma que uma pessoa a bordo com mais de 2 parentes ou crianças tem mais chances de sobreviver, enquanto
# quem estiver sozinho tem mais chances de morrer.


# In[18]:


grafico_barras('Embarked')


# In[ ]:


# O gráfico confirma que pessoas que embarcaram em "C" tem um pouco mais de chances de sobreviver, enquanto
# quem emabrcou em "Q" e "S" tem mais chances de morrer.


# In[21]:


grafico_barras("Age")


# In[28]:


sns.pairplot(titanic, hue="Survived", height=2.2)
plt.show()


# In[67]:


g = sns.FacetGrid(titanic, col="Survived")
g.map(plt.hist, 'Age', bins=20)


# In[68]:


g = sns.FacetGrid(titanic, col="Survived", row="Pclass", height=2.2)
g.map(plt.hist, 'Age', bins=20)
g.add_legend()


# ## Nome (Título)

# In[62]:


titanic_teste_dados = [titanic] # combining train and test dataset

for dataset in titanic_teste_dados:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[63]:


titanic['Title'].value_counts()


# #### Substituindo:

# In[66]:


#Mr : 0
#Miss : 1
#Mrs: 2
#Outros: 3


# In[64]:


titulo = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in titanic_teste_dados:
    dataset['Title'] = dataset['Title'].map(titulo)


# In[65]:


titanic.head()


# In[69]:


grafico_barras('Title')


# In[ ]:


# Aqui é possível analisar que "Mr" tiveram uma alta mortalidade, enquanto "Miss" e "Mrs" tiveram alta chance de sobreviver.


# ## Predição

# In[264]:


# Criando coluna predicao e deixando todos com o valor "0".

titanic['predicao'] = '0'
titanic


# In[265]:


# Definindo pelas caracteristicas, quais pessoas terão chance de sobreviver e morrer:

# Pessoas em famílias com mais de duas pessoas tiveram maiores chances de sobreviver
titanic['predicao'][(titanic['Parch'] >= 2) & (titanic['SibSp'] >= 2)] = '1'

# "Crianças e mulheres primeiro"
titanic['predicao'][(titanic['Sex'] == 'female')] = '1'
titanic['predicao'][(titanic['Age'] < 10)] == '1'

# Pessoas da primeira classe tiveram mais chances de sobreviver
titanic['predicao'][(titanic['Pclass'] == 1)] == '1'
titanic['predicao'][(titanic['Pclass'] >= 2)] == '0'

titanic['predicao'] = pd.to_numeric(titanic['predicao'])
titanic


# In[279]:


porcentagem = ((titanic['Survived'] == titanic['predicao']).mean()) * 100
print("A porcetagem de acertos foi: {:.2f}%".format(porcentagem))


# ### Muito obrigado por mais um desafio, professor! Primeiramente gostaria de pedir desculpas pelo baixo resultado... Sou calouro e estou aprendendo, por isso pesquisei bastante sobre o desafio e tentei deixa-lo o mais bem explicado o possível, e principalmente o mais longe de um simples "ctrl+c, ctrl+v" buscando assim um grande aprendizado. 
# 
# ### Diogo Pereira Almeida
