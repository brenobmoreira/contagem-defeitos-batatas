import cv2
import numpy as np
from ultralytics import YOLO

# Carregar modelo YOLO
model = YOLO('best.pt')

# Listar informações do modelo
print("=" * 50)
print("INFORMAÇÕES DO MODELO")
print("=" * 50)
print(f"\nNomes das classes: {model.names}")
print(f"Número de classes: {len(model.names)}")
print(f"\nClasses detectáveis:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")
print("=" * 50)

# Carregar imagem
img_path = 'dataset_batatas\potato1.jpeg'
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

print("\n" + "=" * 50)
print("PROCESSAMENTO DE DEFEITOS")
print("=" * 50)

# ==================================================
# PRÉ-PROCESSAMENTO
# ==================================================
print("\n1. Pré-processamento...")
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
blur = cv2.GaussianBlur(img_lab, (5,5), 0)

# ==================================================
# SEGMENTAÇÃO DA BATATA DO FUNDO
# ==================================================
print("2. Segmentação...")
L, A, B = cv2.split(blur)
_, mask = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
mask_filled = mask_clean.copy()
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(mask_filled, [contours[i]], -1, 255, -1)

mask = mask_filled

# ==================================================
# MÉDIA E COVARIÂNCIA DA REGIÃO SAUDÁVEL
# ==================================================
print("3. Calculando média e covariância...")
pixels = blur[mask == 255].reshape(-1, 3)
mean = np.mean(pixels, axis=0)
cov = np.cov(pixels, rowvar=False)
inv_cov = np.linalg.inv(cov)

print(f"   Média (Lab): {mean}")

# ==================================================
# DISTÂNCIA DE MAHALANOBIS
# ==================================================
print("4. Calculando distância de Mahalanobis...")
h, w, c = blur.shape
diff = blur.reshape(-1, 3) - mean
mdist = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
mdist_img = mdist.reshape(h, w)
mdist_norm = cv2.normalize(mdist_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ==================================================
# DETECÇÃO DE DEFEITOS
# ==================================================
print("5. Detectando regiões de defeitos...")
_, exclude_mask = cv2.threshold(mdist_norm, 180, 255, cv2.THRESH_BINARY)
_, defect_mask = cv2.threshold(mdist_norm, 100, 255, cv2.THRESH_BINARY)
defect_mask = cv2.bitwise_and(defect_mask, cv2.bitwise_not(exclude_mask))

kernel_defect = np.ones((3,3), np.uint8)
defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_defect)
defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel_defect)

# Máscara verde (região intermediária)
lower = 90
upper = 180
green_mask = cv2.inRange(mdist_norm, lower, upper)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_defect)

# ==================================================
# CRIAR MÁSCARA DE DETECÇÕES YOLO
# ==================================================
print("6. Criando máscara das detecções YOLO...")
yolo_mask = np.zeros(img.shape[:2], dtype=np.uint8)
for r in results:
    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, -1)

# ==================================================
# FILTRAR DEFEITOS APENAS DENTRO DAS DETECÇÕES YOLO
# ==================================================
print("7. Filtrando defeitos dentro das batatas detectadas...")
defect_mask_filtered = cv2.bitwise_and(defect_mask, yolo_mask)
green_mask_filtered = cv2.bitwise_and(green_mask, yolo_mask)

# Contar defeitos por batata detectada
print("\nAnalisando defeitos por batata:")
total_defects = 0

for r in results:
    for idx, box in enumerate(r.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Criar máscara apenas desta batata
        single_potato_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(single_potato_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Extrair defeitos apenas dentro desta batata
        defects_in_potato = cv2.bitwise_and(defect_mask, single_potato_mask)
        
        # Contar componentes conectados (defeitos individuais)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defects_in_potato, connectivity=8)
        
        defect_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 50:  # Área mínima
                defect_count += 1
        
        total_defects += defect_count
        
        # Calcular porcentagem de área com defeito nesta batata
        potato_area_single = (x2 - x1) * (y2 - y1)
        defect_area_single = np.sum(defects_in_potato == 255)
        percent_defect_single = (defect_area_single / potato_area_single) * 100 if potato_area_single > 0 else 0
        
        print(f"\nBatata {idx+1}:")
        print(f"  Defeitos: {defect_count}")
        print(f"  Área com defeito: {percent_defect_single:.2f}%")

print(f"\n✓ Total de defeitos detectados em todas as batatas: {total_defects}")

# ==================================================
# CALCULAR PORCENTAGEM DE ÁREA COM DEFEITO
# ==================================================
defect_area = np.sum(defect_mask_filtered == 255)
potato_area = np.sum(yolo_mask == 255)
if potato_area > 0:
    percent_defect = (defect_area / potato_area) * 100
    print(f"✓ Porcentagem da área com defeito: {percent_defect:.2f}%")

# ==================================================
# VISUALIZAÇÃO FINAL
# ==================================================
print("\n8. Gerando visualização final...")
# Overlay com defeitos em vermelho
overlay = img.copy()
overlay[defect_mask_filtered == 255] = [0, 0, 255]  # Defeitos em vermelho

# Overlay com áreas verdes
overlay_green = img.copy()
overlay_green[green_mask_filtered == 255] = [0, 255, 0]  # Verde

alpha = 0.5
result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
result_green = cv2.addWeighted(img, 1 - alpha, overlay_green, alpha, 0)

# Adicionar bounding boxes do YOLO com contagem de defeitos
for r in results:
    for idx, box in enumerate(r.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(box.conf[0])
        
        # Contar defeitos nesta batata para o label
        single_potato_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(single_potato_mask, (x1, y1), (x2, y2), 255, -1)
        defects_in_potato = cv2.bitwise_and(defect_mask_filtered, single_potato_mask)
        num_labels_potato, _, stats_potato, _ = cv2.connectedComponentsWithStats(defects_in_potato, connectivity=8)
        defect_count = sum(1 for i in range(1, num_labels_potato) if stats_potato[i, cv2.CC_STAT_AREA] >= 50)
        
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f'Batata {idx+1} | Defeitos: {defect_count}', (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Salvar resultados
cv2.imwrite('resultado_defeitos.jpg', result)
cv2.imwrite('resultado_defeitos_verdes.jpg', result_green)
cv2.imwrite('mascara_defeitos.jpg', defect_mask_filtered)
cv2.imwrite('mascara_verde.jpg', green_mask_filtered)
cv2.imwrite('mapa_mahalanobis.jpg', mdist_norm)

print("\n✓ Imagens salvas:")
print("  - resultado_defeitos.jpg (defeitos em vermelho + bounding boxes)")
print("  - resultado_defeitos_verdes.jpg (áreas verdes + bounding boxes)")
print("  - mascara_defeitos.jpg (máscara binária dos defeitos)")
print("  - mascara_verde.jpg (máscara binária das áreas verdes)")
print("  - mapa_mahalanobis.jpg (mapa de calor)")

# Mostrar resultado
cv2.imshow('Resultado Final - Defeitos nas Batatas', result)
cv2.imshow('Resultado - Areas Verdes', result_green)
cv2.imshow('Máscara de Defeitos', defect_mask_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 50)
