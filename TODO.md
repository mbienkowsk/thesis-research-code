
* może jakaś dynamiczna zmiana współczynnika k?

* rastrigin i przesunięty rastrigin - tu widać pomału, że sensowne są tylko k=3/4
  * rastrigin - C*pc i pc lepsze niż pozostałe metody i miejscami szybciej zbieżne niż klasyczny, ciekawe, co stanie się dla 50 wymiarów

  * przesunięty rastrigin - tu sporo zależy od wartości k, trzeba zobaczyć jak dla 50 wymiarów
    * dla k=4 ciekawa szybsza zbieżność przez pierwsze 20000 wywołań, później się gubią
    * dla k=3 zaskakująco dobrze wypadają, szybsza zbieżność o +-20%, ale gorszy punkt na koniec

* niemożność znalezienia bracketu do golden searcha dla niektórych funkcji przybliżając się do optimuk

---

* porównanie miary przesunięcia do $\sigma p_c$
* 1000 wymiarów f. pokrzywiona
  * ( BFGS )/DFP/LBFGS
