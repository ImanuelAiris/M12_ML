import pandas as pd
import numpy as np
from sklearn import linear_model

# masukkan data dari database yang berisikan data-data gula, susu, dan coklat berbentuk array ke dataframe
# contoh ada di : https://www.statology.org/add-numpy-array-to-pandas-dataframe/
#
class mlfeedwater:
    reg = linear_model.LinearRegression()
    def inisiasikoef(self):
        df = pd.DataFrame({'konduktivitas': [248, 873, 1360, 1942],
                        'tekanan': [344, 866, 918, 819],
                        'temperatur': [343, 469, -35, 233],
                        'feedwater': [1, 1, 0, 0]})
        self.reg.fit(df[['konduktivitas', 'tekanan', 'temperatur']],df.price)

    
    def hitungml(self, konduktivitas, tekanan, temperatur):
        return self.reg.predict([[konduktivitas, tekanan, temperatur]])
    
class mlsteam:
    reg = linear_model.LinearRegression()
    def inisiasikoef(self):
        df = pd.DataFrame({'ph': [2, 3, 11, 8],
                        'oksigen': [60, 17, 58, 3],
                        'uap': [1335, 790, 1421, 926],
                        'steam': [1, 0, 1, 0]})
        self.reg.fit(df[['ph', 'oksigen', 'uap']],df.price)

    
    def hitungml(self, ph, oksigen, uap):
        return self.reg.predict([[ph, oksigen, uap]])
    
class mlfurnance:
    reg = linear_model.LinearRegression()
    def inisiasikoef(self):
        df = pd.DataFrame({'gas': [130, 245, 908, 582],
                        'suara': [98, 99, 75, 93],
                        'kelembaban': [67, 27, 92, 32],
                        'furnance': [300, 826, 1480, 638]})
        self.reg.fit(df[['gas', 'suara', 'kelembaban']],df.price)

    
    def hitungml(self, gas, suara, kelembaban):
        return self.reg.predict([[gas, suara, kelembaban]])