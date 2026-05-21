# zzsn-projekt

* Mikołaj Garbowski
* Maksym Bieńkowski


## TODO
* Identyfikacja interesujących wymiarów słownika
  * score oceniający jak ciekawy jest feature w papierze sauron
  * coś jak tf-idf
* Tuning hiperparametrów
  * możemy np. wziąć samą stratę rekonstrukcji
  * może mierzyć sparsity na zbiorze walidacyjnym
  * spróbować expansion factor 16 jak w SAeUron

## Pomysły na dalsze eksperymenty

* Naiwny wariant
  * D niezależnych predyktorów
  * zobaczyć czy na Athenie to się zmieści do pamięci
  * jak nie to spróbować ładować i odkładać na dysk
  * a może chociaż z małym D jak się nie uda dla rzadkiej dużego (rzadkiego słownika)
* Wariant ze wspólnym predyktorem i hypernetwork
  * w prostym wariancie predyktor dostaje na wejściu indeks wymiaru i dokłada jego embedding
  * zamiast tego, oddzielna sieć na podstawie indeksu wyznacza wagi predyktora (per indeks)

## Resources

### [ Athena ](./docs/antena.md)

### References
- https://github.com/cywinski/SAeUron
- https://github.com/ashleve/lightning-hydra-template/

### Papers
- [Schmidhuber](https://arxiv.org/pdf/2501.18052)
- [SAeUron](https://arxiv.org/pdf/2501.18052)
