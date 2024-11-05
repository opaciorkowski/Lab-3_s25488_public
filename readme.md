# Lab-4_s25488 - Instrukcja Użytkowania
### **Automatyczne Wdrożenie z GitHub Actions**

Repozytorium jest skonfigurowane tak, aby automatycznie uruchamiać GitHub Actions i wdrażać aplikację na Docker Hub przy pushu do gałęzi `main` z commitem zawierającym słowo "deploy" w opisie commita. 

GitHub Actions automatycznie:
- Loguje się na Docker Hub, wykorzystując dane `DOCKERHUB_USERNAME` oraz `DOCKERHUB_TOKEN`, które są przechowywane w `GitHub Secrets`.
- Buduje obraz Docker aplikacji na podstawie pliku `Dockerfile`.
- Publikuje zbudowany obraz na Docker Hub.

Aby skorzystać z tej funkcji, dodaj `DOCKERHUB_USERNAME` i `DOCKERHUB_TOKEN` w sekcji `Secrets` repozytorium na GitHub, a następnie wykonaj `commit` i `push` ze słowem "deploy" w wiadomości.

### **a) Klonowanie Repozytorium**
1. **Skopiowanie repozytorium z GitHub:**

    ```bash
    git clone https://github.com/opaciorkowski/Lab-4_s25488_public.git
    cd Lab-4_s25488_public
    ```

### **b) Uruchomienie Aplikacji Lokalnie**
1. **Instalacja wymaganych bibliotek:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Uruchomienie aplikacji Flask:**

    Skrypt automatycznie pobiera dane oraz trenuje model, jeśli plik modelu nie istnieje. Upewnij się, że masz połączenie z Internetem przy pierwszym uruchomieniu, aby mógł pobrać dane.
   
    ```bash
    python app/s25488.py
    ```

3. **Sprawdzenie działania:**

   Aplikacja działa domyślnie na porcie 5000. Możesz otworzyć przeglądarkę i przejść do `http://localhost:5000` lub użyć poniższego polecenia `curl`, aby wysłać zapytanie do endpointu `/predict`.

### **c) Uruchomienie Aplikacji z Dockerem**

1. **Budowanie obrazu Docker:**

    W katalogu głównym projektu uruchom:
    ```bash
    docker build -t opaciorkowski/lab4:latest .
    ```
2. **Uruchomienie kontenera Docker:**
    ```bash
    docker run -d -p 5000:5000 --name lab4 opaciorkowski/lab4:latest
    ```
3. **Zatrzymanie kontenera Docker:**
    ```bash
    docker stop lab4
    ```

### **d) Korzystanie z Obrazu z Docker Hub**

1. **Pobranie obrazu z Docker Huba:**

    ```bash
    docker pull opaciorkowski/lab4:latest
    ```
2. **Uruchomienie kontenera z pobranego obrazu:**
    ```bash
    docker run -d -p 5000:5000 --name lab4 opaciorkowski/lab4:latest
    ```
3. **Zatrzymanie kontenera Docker:**
    ```bash
    docker stop lab4
    ```

### **e) Użycie Endpointu /predict**

Możesz użyć `curl` w wierszu poleceń, aby wysłać żądanie do endpointu `/predict`:

#### Przykład z użyciem JSON:
1. **Wykonanie żądania z danymi w formacie JSON (przykład użycia):**
    ```bash
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "[\"male\", \"other\", \"yes\", \"no\", \"yes\", \"yes\", 6.2, 8.09, 0.2, 0.88915, 12, \"high\", \"other\"]"
    ```

#### Przykład z użyciem pliku CSV:
1. **Przygotowanie pliku CSV z odpowiednimi danymi, np. `input.csv`.**
2. **Wysłanie pliku z danymi w formacie CSV:**
    ```bash
    curl -X POST http://localhost:5000/predict -H "Content-Type: text/csv" --data-binary @input.csv
    ```

### **Przykładowa odpowiedź:**
Aplikacja zwróci przewidywaną wartość w formacie JSON:

```json
{
    "predictions":[49.920597076416016]
}
