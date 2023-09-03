import cv2
import face_recognition as fr
import os
from rich import print
from rich.console import Console
console = Console()

## SISTEMA DE CADASTRO 
print("="*28)
console.print(f'Seja Bem vindo ao DevCad{os.linesep}', style="bold green")
name_cad = input(f'Digite Seu nome: ')
console.print('Ok, Agora abra o programa na barra de tarefas', style="bold red")

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rostos
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX 

    cv2.putText(
        frame,
        'Pressione Enter Para capturar seu rosto',
        (150,20),
        font, 0.6,
        (0,0,0),
        2,
        cv2.LINE_4
    )

    cv2.putText(
        frame,
        '- ESC para sair -',
        (250,450),
        font, 0.6,
        (0,0,0),
        2,
        cv2.LINE_4
    )

    cv2.imshow('Cadastro de Foto Base', frame)

    key = cv2.waitKey(1)  
    
    if key == 27:
        break
    elif key == 13:
        cv2.imwrite(f"images/{name_cad}.png", frame)


video_capture.release()
cv2.destroyAllWindows()

## SISTEMA DE LOGIN:
print(f'{os.linesep}')
print("="*28)
console.print(f'Seja Bem vindo ao DevLogin{os.linesep}', style="bold green")
name_log = input(f'Digite Seu nome Cadastrado: ')
console.print('Ok, Agora abra o programa na barra de tarefas' , style="bold red")

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX 

    cv2.putText(
        frame,
        'Reconhecimento Facial, pressione Enter para capturar seu rosto.',
        (10,20),
        font, 0.5,
        (0,0,0),
        2,
        cv2.LINE_4
    )

    cv2.putText(
        frame,
        '- ESC para sair -',
        (250,450),
        font, 0.6,
        (0,0,0),
        2,
        cv2.LINE_4
    )

    cv2.imshow('video', frame)
    key = cv2.waitKey(1)  
    if key == 27:
        break
    elif key == 13:
        cv2.imwrite(f"images/{name_log}_desconhecido.png", frame)


## ENGINE DE CAPTURA E PARAMETROS
def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos
    else:
        return False, []


def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []


    rosto_cadastrado = reconhece_face(f"images/{name_cad}.png")
    if(rosto_cadastrado[0]):
        rostos_conhecidos.append(rosto_cadastrado[1][0])
        nomes_dos_rostos.append(f"{name_cad}")

    return rostos_conhecidos, nomes_dos_rostos


## SISTEMA DE COMPARAÇÃO DE ROSTOS E RESULTADOS
desconhecido = reconhece_face(f"images/{name_log}_desconhecido.png")
if(desconhecido[0]):
    rosto_desconhecido  = desconhecido[1][0]
    rostos_conhecidos, nomes_dos_rostos = get_rostos()
    resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
    if resultados == [True]:
        console.print(f"{os.linesep}-> Usuário Identificado!", style="green bold")
    elif resultados == [False]:
        console.print(f"{os.linesep}-> Usuário Não Corresponde ao Cadastrado!", style="red bold", )

    for i in range(len(nomes_dos_rostos)):
        resultado = resultados[i]
        if(resultado):
            print("O Rosto de", nomes_dos_rostos[i], "Foi reconhecido.")
            print(f"{os.linesep}")

else:
    print('Não foi encontrado nenhum rosto')

video_capture.release()
cv2.destroyAllWindows()