import cv2
import mediapipe as mp

mp_reka = mp.solutions.hands
reka = mp_reka.Hands()

def rozpoznaj_gest(hand_landmarks):
    
    koniuszek_kciuka = hand_landmarks[4]  # pozycja końca kciuka



   # rysowanie punktow pomocniczych na koncac palcow+koncu dloni
    #for i in range(0, 21, 4):
    #   cv2.circle(obraz, (int(hand_landmarks[i].x * szerokosc_obrazu), int(hand_landmarks[i].y * wysokosc_obrazu)), 5, (0,0, 0), -1)

    # warunek dla gestu kamienia
    if koniuszek_kciuka.y < hand_landmarks[8].y and koniuszek_kciuka.y < hand_landmarks[12].y and koniuszek_kciuka.y < hand_landmarks[16].y and koniuszek_kciuka.y < hand_landmarks[20].y:
        return "kamien"
    
    # warunek dla gestu nożyc
    if koniuszek_kciuka.y > hand_landmarks[5].y and hand_landmarks[16].y > hand_landmarks[5].y and hand_landmarks[20].y > hand_landmarks[9].y:
        return "nozyce"
    
    # pozostałe przypadki, uznajemy za gest papieru
    return "papier"

# inicjalizacja kamery
kamera = cv2.VideoCapture(0)

while kamera.isOpened():
    sukces, obraz = kamera.read()
    if not sukces:
        break

    wysokosc_obrazu, szerokosc_obrazu, _ = obraz.shape  # Pobranie rozmiarów ramki

    # konwersja kolorów z BGR do RGB
    rgb_obraz = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)

    # znalezienie ręki na obrazie
    wyniki = reka.process(rgb_obraz)

    # jeśli ręka została znaleziona
    if wyniki.multi_hand_landmarks:
        for hand_landmarks in wyniki.multi_hand_landmarks:
            # rysowanie siatki palców na obrazie
            for i, landmark in enumerate(hand_landmarks.landmark):
                pozycja = (int(landmark.x * szerokosc_obrazu), int(landmark.y * wysokosc_obrazu))
                cv2.circle(obraz, pozycja, 5, (255, 182, 193), -1)
                cv2.putText(obraz, str(i), (pozycja[0] - 10, pozycja[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # rozpoznawanie gestu
            gesture = rozpoznaj_gest(hand_landmarks.landmark)
            cv2.putText(obraz, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 200), 2, cv2.LINE_AA)

    # Wyświetlenie obrazu z kamery
    cv2.imshow("Rozpoznawanie Gestów Ręką", obraz)

    if cv2.waitKey(1) & 0xFF == 27:  # Naciśnij 'Esc' aby wyjść z programu
        break  # Ta linia musi być wewnątrz pętli

# Zwolnienie zasobów
kamera.release()
cv2.destroyAllWindows()
