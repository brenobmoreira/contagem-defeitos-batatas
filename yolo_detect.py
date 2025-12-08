import cv2
import numpy as np
from ultralytics import YOLO

# Carregar modelo YOLO
model = YOLO('best.pt')

# Carregar imagem
img_path = 'batata_ruim.jpeg'
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

# Executar detecção
results = model(img)

# Processar resultados
for r in results:
    # Imagem anotada com bounding boxes
    annotated = r.plot()
    
    # Mostrar informações das detecções
    print(f"\nDetectou {len(r.boxes)} batata(s):")
    
    for i, box in enumerate(r.boxes):
        # Extrair informações
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        
        print(f"\nBatata {i+1}:")
        print(f"  Classe: {r.names[cls]}")
        print(f"  Confiança: {conf:.2%}")
        print(f"  Coordenadas: x1={int(xyxy[0])}, y1={int(xyxy[1])}, x2={int(xyxy[2])}, y2={int(xyxy[3])}")
        
        # Extrair região da batata
        x1, y1, x2, y2 = map(int, xyxy)
        potato_roi = img[y1:y2, x1:x2]
        
        # Salvar batata recortada (opcional)
        cv2.imwrite(f'batata_{i+1}.jpg', potato_roi)

# Salvar imagem com detecções
cv2.imwrite('batata_ruim_detectada.jpg', annotated)
print("\n✓ Imagem salva como 'batata_ruim_detectada.jpg'")

# Mostrar imagem (descomentar se quiser visualizar)
cv2.imshow('Detecções', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
