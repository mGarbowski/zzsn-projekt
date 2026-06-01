# zzsn-projekt

* Mikołaj Garbowski
* Maksym Bieńkowski


## TODO
* Opracowanie wyników ewaluacji wytrenowanych modeli

## Zrealizowane
* Zbieranie aktywacji z warstw modelu StableDiffusion 1.4 dla promptów ze zbioru danych COCO.
  * Zapisane w zbiorze danych dostępnym na [HuggingFace](https://huggingface.co/datasets/mgarbowski/zzsn-activations-2_unet.up_blocks.1.attentions.2)
* Implementacja i trening modelu minimalizowania przewidywalności.
  * Wagi wytrenowanych modeli zapisane na platformie [Weights and Biases](https://wandb.ai/mikolaj-garbowski-warsaw-university-of-technology/zzsn-projekt?nw=nwusermikolajgarbowski)
* Implementacja metody interwencji w generację obrazka
  * klasa `WrappedDiffusion`
  * demonstracja API w [notatniku](./notebooks/04-intervention-demo.ipynb)
* Wizualizacja aktywacji poszczególnych wymiarów słownika w postaci heatmap
  * demonstracja API w [notatniku](./notebooks/05-saliency.ipynb)
* Analiza przestrzeni słownika pod kątem istotności wymiarów reprezentacji dla danych konceptów
  * skrypt `analyze_autoencoder`
  * przykładowy [job](https://wandb.ai/mikolaj-garbowski-warsaw-university-of-technology/zzsn-projekt/runs/t1n57bkx?nw=nwusermikolajgarbowski) z wynikami analizy
  * ocena wymiarów miarą opisaną w paperze SAeUron (koncepcyjnie podobna do TF-IDF)


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
