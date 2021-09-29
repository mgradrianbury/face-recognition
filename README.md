# Face recognition with django

Aplikacja, której zadaniem jest rozpoznanie osoby przedstawionej na zdjęciu.

## Instalacja

### Klonowanie repozytorium

```bash
git clone https://github.com/mgradrianbury/face-recognition
cd face-recognition
```

Plik `facenet_keras.h5` jest trzymany w `Git LFS`.
W razie problemów należy pobrać go ręcznie.

### Instalacja Pythona

Aplikacja wymaga Pythona dokładnie w wersji `3.6.9` (z innymi wersjami aplikacja nie była testowana i jest wysoce
prawdopodobne, że nie zadziała). Aby sprawdzić jaka wersja jest obecnie zainstalowana w systemie, należy wpisać:

```bash
python --version
# -> Python 3.6.9
```

Jeżeli wersja jest inna niż `3.6.9` należy użyć [pipenv](https://github.com/pypa/pipenv)
do utworzenia projektu z wymaganą wersją Pythona:

```bash
sudo pip install pipenv
pipenv --python 3.6.9
pipenv shell
python --version
# -> Python 3.6.9
```

### Instalacja wymaganych bibliotek

Lista potrzebnych bibliotek znajduje się w pliku `Pipfile`. Aby je zainstalować, należy wpisać komendę:

```bash
pipenv install
```

W trakcie instalacji mogą pojawić się błędy związane z brakiem bibliotek systemowych. W takiej sytuacji brakujące
biblioteki należy zainstalować ręcznie.

### Przygotowanie aplikacji do uruchomienia

Przed uruchomieniem aplikacji należy utworzyć bazę danych wraz z tabelami. Aplikacja wykorzystuje `SQLite`, więc nie ma
potrzeby uruchamiać zewnętrznych baz. W tym celu należy uruchomić komendę:

```bash
./face_recognition/manage.py migrate
```

Kolejnym krokiem jest utworzenie konta administratora w celu uzyskania dostępu do panelu:

```bash
./face_recognition/manage.py createsuperuser
```

Po jej wpisaniu postępuj zgodnie z poleceniami wyświetlanymi w konsoli.

### Uruchomienie aplikacji

Aby uruchomić aplikację w tybie developerskim, należy wpisać:

```bash
./face_recognition/manage.py runserver
```

Aplikacja powinna być dostępna pod adresem [http://127.0.0.1:8000](http://127.0.0.1:8000). Dostęp do panelu
administratora znajduje się pod adresem [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin).
