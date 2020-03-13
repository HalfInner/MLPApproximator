# MLPApproximator
Research how number of perceptrons against of it quality. Backward error propagation. 

## Requirements [PL]
Projekt jest rozliczany na podstawie:
 * raportu wstępnego (0 - 6 pkt.)
 * rozliczenia końcowego (0 - 24 pkt.)

Próg punktowy wymagany dla zaliczenia projektu i terminy rozliczania obu jego części podane są w regulaminie przedmiotu. Projekt rozliczany jest poprzez platformę kursy.okno.pw.edu.pl (zakładki: „Raport wstępny”, „Projekt końcowy”). W raporcie wstępnym należy:
  1. Przedstawić zadaną metodykę lub algorytm Sztucznej Inteligencji (w tym podać dokładny pseudokod algorytmu i skomentować go własnymi słowami);
  2. Przedstawić koncepcję rozwiązania zadanego problemu z wykorzystaniem metodyki lub algorytmu Sztucznej Inteligencji (w tym wyrazić zadany problem w sposób wymagany przez wybraną metodykę Szt. Int. – format danych, wymagane funkcje - i zdefiniować w pseudo-kodzie interfejs pomiędzy reprezentacją problemu a metodyką). W rozliczeniu końcowym projektu należy:
  3. Przekazać raport końcowy zawierający:
    * poprawiony i uaktualniony raport wstępny (pkt. A, B),
    * opis struktury i funkcji zrealizowanego programu,
    * sposób uruchomienia programu,
    * podsumować wykonane testy i przedstawić wnioski.
  4. Przekazać program i jego źródła (kody programu) oraz przykładowy plik monitorujący wykonanie programu. Program może wykonywać się w trybie wsadowym („batch”) –  nie jest wymagany graficzny interfejs użytkownika (za wyjątkiem wizualizacji wyników w niektórych projektach typu N). W typowym rozwiązaniu program posiada możliwość zadawania argumentów w linii wywołania, wczytuje dane z przygotowanych plików i zapisuje uzyskiwane wyniki w pliku monitorującym jego wykonanie a także w oknie wykonania programu

Część N (uczenie sieci neuronowej)
Projekt N1. MLP – aproksymacja funkcji. Wykonać program do badania wpływu ilości neuronów perceptronu na jakość aproksymacji w sieciach neuronowych typu MLP uczonych algorytmem wstecznej propagacji błędu.
 1. Zaimplementować algorytm działania sieci MLP (aproksymacji nauczonej funkcji) i algorytm uczenia tej sieci.
 2. Wygenerować dane uczące i dane testujące – przyjąć 3 nieliniowe funkcje Margumentowe (M=3, 5, 7).
 3. Przewidzieć możliwość aproksymacji zbioru funkcji za pomocą sieci neuronowej MLP o M wejściach, 3 wyjściach oraz jednej warstwie ukrytej (o zmiennej liczbie neuronów N od N=M do N = 10M).
 4. Sprawdzić działanie procedury uczenia sieci i procedury aproksymacji funkcji – zmieniać liczbę neuronów w warstwie ukrytej, warunki początkowe i liczbę iteracji procesu uczenia (I od I=100 do 1000).
 5. Wynik ma postać zależności uzyskanych wyników aproksymacji trzech funkcji (średniego błędu aproksymacji każdej z funkcji w zadanym przedziale wartości argumentów) od parametrów M i N oraz liczby iteracji I

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
