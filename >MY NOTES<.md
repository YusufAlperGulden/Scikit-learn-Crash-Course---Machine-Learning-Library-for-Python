%pip install scikit-learn scikit-lego (ilk önce gerekli olan kütüphaneleri indirmelisin.)


import numpy as np #You need to import all of these.
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification (BUNLARIN HEPSİNİ İMPORT ET.)

X, y = make_classification(n_samples=2000)

Bu "n_samples" değerini değiştirerek grafiğin üzerindeki nokta sayının değiştirmiş oldum.

 "class_sep=1.75" grafikteki noktalar arasındaki ayrımı (seperation) kontrol ediyor.


 plt.figure(figsize=(16, 4)) Şunu fark ettim: Burada bulunan sayıları değiştirerek grafiğin boyutunu değiştirebiliyorum. Sanırım 16=width ve 4=height oluyor.


 clf1 = LogisticRegression().fit(X, y) ----LogisticRegression modeli


 clf2 = KNeighborsClassifier(n_neighbors=10).fit(X, y) -----K-neighbor modeli


clf3 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2)] ----VotingClassifer'da sol tarafa clf1'i koyup sağ tarafa clf2'yi yerleştiriyoruz.


weights=[999999, 2]) ---Hangisinin daha ağır basacağını gösterir. clf1 mi yoksa clf2 mi daha ağır basacak? Buradaki sayıları değiştirebilirsin.


from sklearn.datasets import make_blobs


X, y = make_blobs(100000, centers=[(0, 0), (1.5, 1.5)], cluster_std=[1, 0.5]) #cluster_std dağılma miktarını gösteriyor galiba?


Orada bulunan 100000 değeri yerine daha küçük bir sayı yazarsam nokta sayısı azalıyormuş. Birkaç defa test yaptım.





 
                           
