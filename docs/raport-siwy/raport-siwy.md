---
# pandoc raport-siwy.md --citeproc -o raport-siwy.pdf
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


## Zaimplementowany wariant modelu

Implementacja predyktorów jako oddzielnej sieci neuronowej (MLP) dla każdego elementu reprezentacji prowadzi do ogromnej liczby parametrów.
Zdecydowaliśmy się na bardziej kompaktowy wariant z jednym współdzielonym predyktorem.
Współdzielony predyktor jest perceptronem wielowarstwowym.
Jego wejście stanowi $n$-wymiarowy wektor reprezentacji, w którym $i$-ty element jest wyzerowany oraz wektor zanurzenia indeksu $i$-tego elementu reprezentacji.
W ten sposób predyktor może rozróżnić który element reprezentacji ma przewidywać, a jednocześnie liczba parametrów jest liniowa względem rozmiaru słownika.

Koder i dekoder są pojedynczymi warstwami liniowymi.

Trening odbywa się w naprzemiennych fazach:

1. Trening predyktora
  * Minimalizuje błąd predykcji reprezentacji (MSE)
  * Równoważne maksymalizacji przewidywalności reprezentacji (każdego elementu na podstawie pozostałych)
2. Trening autokodera
  * Minimalizuje błąd rekonstrukcji i minimalizuje przewidywalność reprezentacji (strata predyktora z odwróconym znakiem)
  * Funkcja straty to suma ważona tych dwóch składników, waga jest hiperparametrem

Fazy zmieniają się co określoną liczbę minipakietów, liczba jest hiperparametrem.

## Zrealizowane kroki

* Zbieranie aktywacji z warstw modelu StableDiffusion 1.4 dla promptów ze zbioru danych COCO.
  * Zapisane w zbiorze danych dostępnym na [HuggingFace](https://huggingface.co/datasets/mgarbowski/zzsn-activations-2_unet.up_blocks.1.attentions.2)
* Implementacja i trening modelu minimalizowania przewidywalności.
  * Wagi wytrenowanych modeli zapisane na platformie [Weights and Biases](https://wandb.ai/mikolaj-garbowski-warsaw-university-of-technology/zzsn-projekt?nw=nwusermikolajgarbowski)
* Implementacja metody interwencji w generacje obrazka
  * Interwencja polega na przepuszczeniu aktywacji przez koder, zmianie wybranych wymiarów reprezentacji, przepuszczeniu zmienionej reprezentacji przez dekoder i wprowadzeniu zmienionych aktywacji z powrotem do modelu generatywnego.
  * Zaimplementowana metoda pozwala na wzmocnienie lub osłabienie wybranego konceptu poprzez przemnożenie wybranych wymiarów przez skalar.
  * Demonstracja metody interwencji jest przedstawiona w notatniku [04-intervention-demo](https://github.com/mGarbowski/zzsn-projekt/blob/main/notebooks/04-intervention-demo.ipynb)

## Dalsze kroki
* Lepsze dostrojenie hiperparametrów treningu modelu
* Zidentyfikowanie interesujących wymiarów reprezentacji i zbadanie ich interpretowalności
  * przez porównanie reprezentacji w autokoderze pomiędzy generacjami z różnych promptów

---
