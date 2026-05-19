# zzsn-projekt


## TODO
* Przy interwencji - dodawać rekonstrukcję tylko tych wymiarów w któ©ych wbyłą interwencja
* Analiza który patch aktywuje dany feature
  * wizualizacja heatmap
* Uśrednianie featerów po patchach - zły pomysł
* Losowanie featureow i timestepow
* Odwrotna numeracja - duży timestep - najbliżej szumu
* Pomysł z różnymi grupami promptów - dobry?
* score oceniający jak ciekawy jest feature - w papierze sauron
  * coś jak tf-idf
* Może jakaś metryka oceniająca jak zlokalizowane se patche
* Może analizować featurey nie uśrednione a dla kilku wylosowanych patchy
* Tuning hiperparametrów
  * możemy np wziąc samą stratę rekonstrukcji
  * może mierzyć sparsity na zbiorze walidacyjnym
* Spróbować expansion factor 16 jak w saeuronie
* Wyłączyć filtra NSFW

* Zebrać większy zbiór
  * aktualnie 1000 promptów
  * wiele obrazków per prompt?
  * da się to puścić równolegle na wiele GPU?
* Zaimplementować interwencję w generacje obrazków
  * dyfuzja spięta z wytrenowanym modelem
  * hook na przepuszczenie aktywacji przez model, zmianę ustalonych wymiarów i przepuszczenie z powrotem
* Jak oceniać jakość modelu
  * żeby robić sweep(?)
* Obsłużyć sweep hiperparametrów z wandb

Do dalszych eksperymentów

* Naiwny wariant
  * D niezależnych predyktorów
  * zobaczyć czy na Athenie to się zmieści do pamięci
  * jak nie to spróbować ładować i odkładać na dysk
  * a może chociaż z małym D jak się nie uda dla rzadkiej dużego (rzadkiego słownika)
* Wariant z wspólnym predyktorem i hypernetwork
  * w prostym wariancie predyktor dostaje na wejściu indeks wymiaru i dokłada jego embedding
  * zamiast tego, oddzielna sieć na podstawie indeksu wyznacza wagi predyktora (per indeks)

## Resources:

### [ Athena ](./docs/antena.md)

### References
- https://github.com/cywinski/SAeUron
- https://github.com/ashleve/lightning-hydra-template/

### Papers
- [Schmidhuber](https://arxiv.org/pdf/2501.18052)
- [SAeUron](https://arxiv.org/pdf/2501.18052)

