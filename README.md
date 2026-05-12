# zzsn-projekt


## TODO
* Zebrać większy zbiór
  * wiele obrazków per prompt?
* Popawić skrypt do zbierania aktywacji żeby zapisywał już spłaszczony dataset
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
