---
# pandoc dokumentacja-wstepna.md --citeproc -o dokumentacja-wstepna.pdf
lang: pl
bibliography: references.bib
csl: ieee.csl
link-citations: true
---


# Dokumentacja wstępna

* Mikołaj Garbowski
* Maksym Bieńkowski

## Opis zadania

Chcemy zbadać algorytm minimalizowania przewidywalności opisany w artykule [@schmidhuber] do zastosowań w zakresie mechanistycznej interpretowalności, jako alternatywę do wykorzystania modeli typu SAE [@sae].

Algorytm [@schmidhuber] służy do wyznaczania reprezentacji wektora liczb rzeczywistych przez adwersarialny trening modułów wyznaczających reprezentację oraz modułów przewidujących pojedynczy element reprezentacji na podstawie pozostałych elementów.
Takie zadanie optymalizacji prowadzi do reprezentacji, w której wszystkie elementy są wzajemnie niezależne.

Algorytm można rozumieć jako nienadzorowane uczenie się słownika reprezentacji.
Oczekujemy że w wyniku treningu otrzymamy słownik, w którym każdy element odpowiada pojedynczemu (monosemantycznemu) konceptowi, interpretowalnemu przez człowieka.

Planujemy wytrenować opisany model na aktywacjach modelu generatywnego StableDiffusion 1.4 w celu dekomopozycji aktywacji, gdzie w ogólności pojedyncze neurony są polisemantyczne, w celu:

1. Możliwości zinterpretowania tych aktywacji
2. Możliwości wykonania interwencji, w celu wzmocnienia/osłabienia wybranego konceptu w generowanej próbce (obrazie)

Jako punkt odniesienia do naszych badań wykorzystamy [@cywinski2025saeuron], gdzie autorzy zidentyfikowali interesujące warstwy w modelu StableDiffusion 1.4 za pomocą rzadkiego autokodera.

## Architektura modelu

Dla rozmiaru słownika $n$, model składa się z:

* $n$ modułów $f_i$ wyznaczających reprezentację
    * w szczególności perceptron wielowarstwowy z jednym neuronem w warstwie wyjściowej
* $n$ predyktorów $g_i$
    * w szczególności perceptron wielowarstwowy z jednym neuronem w warstwie wyjściowej
    * $g_i: \mathbb{R}^{n-1} \rightarrow\mathbb{R}$
    * wejście to wszystkie pozycje reprezentacji poza $i$
    * wyjście to predykcja pozycji $i$ reprezentacji
* opcjonalny moduł dekodera
    * wtedy model ma ogólną strukturę autokodera
    * istotny jeśli chcemy przenieść interwencję w przestrzeni słownika na przestrzeń aktywacji modelu generatywnego


## Planowane eksperymenty

* W pierwszej kolejności planujemy zbadać liniowy wariant modelu
    * wszystkie moduły $f_i$ reprezentowane przez pojedynczą macierz wag
    * wagi macierzy wyznaczają kierunki w przestrzeni aktywacji
    * nie jest wymagany moduł dekodera żeby dokonać interwencji w przestrzeni aktywacji
* W następnej kolejności planujemy zbadać modele nieliniowe (perceptrony wielowarstwowe)
    * [@schmidhuber] sugeruje użycie modeli nieliniowych
    * według [@schmidhuber] moduły mogą współdzielić początkowe warstwy
* Badany model generatywny [Stable Diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) [@Rombach_2022_CVPR]
* Zbieranie aktywacji do wytrenowania naszego modelu
    * do zbierania aktywacji badanych warstw modelu wykorzystamy prompty ze zbioru danych COCO


## Narzędzia
* Python
* Biblioteka PyTorch
* Ekosystem bibliotek HuggingFace

---
