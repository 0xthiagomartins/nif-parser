"""
Módulo para correção de perspectiva e rotação de documentos.
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np


def detect_document_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detecta os 4 cantos do documento na imagem.
    Retorna os pontos em ordem: topo-esquerda, topo-direita, baixo-direita, baixo-esquerda.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Array de 4 pontos (x, y) ou None se não detectar
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Aplicar blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilatar para conectar linhas próximas
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filtrar por área (documento deve ocupar boa parte da imagem)
    min_area = img.shape[0] * img.shape[1] * 0.15
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        return None
    
    # Pegar o maior contorno
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Aproximar contorno para obter polígono
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Se não temos 4 pontos, tentar ajustar
    if len(approx) != 4:
        # Se temos mais de 4 pontos, pegar os 4 mais distantes
        if len(approx) > 4:
            # Calcular centro
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Ordenar pontos por distância do centro
                points = approx.reshape(-1, 2)
                center = np.array([cx, cy])
                distances = np.linalg.norm(points - center, axis=1)
                
                # Pegar 4 pontos mais distantes
                indices = np.argsort(distances)[-4:]
                approx = approx[indices]
        
        # Se ainda não temos 4 pontos, criar retângulo a partir do contorno
        if len(approx) != 4:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box).reshape(-1, 1, 2)
    
    if len(approx) != 4:
        return None
    
    # Ordenar pontos: topo-esquerda, topo-direita, baixo-direita, baixo-esquerda
    points = approx.reshape(4, 2)
    
    # Soma e diferença para identificar cantos
    sum_points = points.sum(axis=1)
    diff_points = np.diff(points, axis=1)
    
    # Topo-esquerda: menor soma
    top_left = points[np.argmin(sum_points)]
    # Baixo-direita: maior soma
    bottom_right = points[np.argmax(sum_points)]
    # Topo-direita: menor diferença
    top_right = points[np.argmin(diff_points)]
    # Baixo-esquerda: maior diferença
    bottom_left = points[np.argmax(diff_points)]
    
    # Verificar se a ordenação faz sentido
    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    return ordered


def correct_perspective(img: np.ndarray, corners: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """
    Corrige a perspectiva do documento (transforma 3D em 2D).
    
    Args:
        img: Imagem em formato numpy array (RGB)
        corners: Cantos do documento (opcional, será detectado se None)
        
    Returns:
        Tupla (imagem_corrigida, sucesso)
    """
    if corners is None:
        corners = detect_document_corners(img)
    
    if corners is None:
        return img.copy(), False
    
    # Calcular dimensões do documento corrigido
    # Usar largura e altura do retângulo delimitador
    width_a = np.linalg.norm(corners[0] - corners[1])
    width_b = np.linalg.norm(corners[2] - corners[3])
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.linalg.norm(corners[0] - corners[3])
    height_b = np.linalg.norm(corners[1] - corners[2])
    max_height = max(int(height_a), int(height_b))
    
    # Pontos de destino (retângulo reto)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Calcular matriz de transformação
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # Aplicar transformação
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    
    return warped, True


def detect_rotation_angle(img: np.ndarray) -> float:
    """
    Detecta o ângulo de rotação do documento usando detecção de linhas melhorada.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Ângulo de rotação em graus (positivo = anti-horário, negativo = horário)
    """
    # Converter para grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Aplicar blur para suavizar
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas com parâmetros mais sensíveis
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    # Usar HoughLinesP (probabilístico) que é mais preciso
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                           minLineLength=100, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        # Tentar com threshold menor
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                               minLineLength=50, maxLineGap=20)
        if lines is None or len(lines) == 0:
            return 0.0
    
    # Calcular ângulos das linhas
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calcular ângulo da linha
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        else:
            angle = 90.0
        
        # Normalizar para -45 a 45 graus
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
        
        angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Calcular ângulo médio (ignorar outliers)
    angles = np.array(angles)
    
    # Remover outliers usando IQR (Interquartile Range)
    q1 = np.percentile(angles, 25)
    q3 = np.percentile(angles, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_angles = angles[(angles >= lower_bound) & (angles <= upper_bound)]
    
    if len(filtered_angles) == 0:
        # Se não sobrou nada, usar mediana
        return float(np.median(angles))
    
    # Retornar mediana dos ângulos filtrados (mais robusto que média)
    return float(np.median(filtered_angles))


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotaciona a imagem pelo ângulo especificado.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        angle: Ângulo de rotação em graus (positivo = anti-horário)
        
    Returns:
        Imagem rotacionada
    """
    if abs(angle) < 0.5:  # Ignorar rotações muito pequenas
        return img.copy()
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Obter matriz de rotação
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calcular novas dimensões
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Ajustar matriz de rotação para nova imagem
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Aplicar rotação
    rotated = cv2.warpAffine(
        img, rotation_matrix, (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # Fundo branco
    )
    
    return rotated


def auto_align_document(img: np.ndarray, max_angle: float = 15.0, min_angle: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Detecta e corrige automaticamente a rotação do documento usando método combinado.
    Versão melhorada e mais agressiva na detecção.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        max_angle: Ângulo máximo a ser corrigido (graus). Se detectado maior, não corrige.
        min_angle: Ângulo mínimo para aplicar correção (graus). Rotações menores são ignoradas.
        
    Returns:
        Tupla (imagem_alinhada, ângulo_detectado)
    """
    # Usar método combinado para maior precisão
    angle = detect_rotation_angle_combined(img)
    
    # Se o ângulo for muito grande, pode ser um falso positivo
    if abs(angle) > max_angle:
        return img.copy(), 0.0
    
    # Se o ângulo for muito pequeno, não vale a pena corrigir
    if abs(angle) < min_angle:
        return img.copy(), angle  # Retorna o ângulo mesmo se não corrigir
    
    aligned_img = rotate_image(img, angle)
    return aligned_img, angle


def detect_rotation_angle_contour(img: np.ndarray) -> float:
    """
    Detecta rotação usando contornos (método alternativo).
    Útil quando há um documento bem definido na imagem.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Ângulo de rotação em graus
    """
    # Converter para grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Aplicar blur antes de binarizar
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarizar
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Operações morfológicas para limpar
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Encontrar o maior contorno (assumindo que é o documento)
    # Filtrar por área mínima (pelo menos 10% da imagem)
    min_area = img.shape[0] * img.shape[1] * 0.1
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        return 0.0
    
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Calcular retângulo rotacionado mínimo
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Normalizar ângulo
    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90
    
    return float(angle)


def detect_rotation_by_document_contour(img: np.ndarray) -> Optional[float]:
    """
    Detecta rotação encontrando o contorno do documento na imagem.
    Método mais robusto que detecta as bordas físicas do documento.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Ângulo de rotação em graus ou None
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Aplicar blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilatar bordas para conectar linhas próximas
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filtrar por área (documento deve ocupar boa parte da imagem)
    min_area = img.shape[0] * img.shape[1] * 0.15  # Pelo menos 15% da imagem
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        return None
    
    # Pegar o maior contorno (assumindo que é o documento)
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Aproximar contorno para reduzir pontos
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Calcular retângulo rotacionado mínimo
    rect = cv2.minAreaRect(approx)
    angle = rect[2]
    
    # Normalizar ângulo
    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90
    
    return float(angle)


def detect_document_edges(img: np.ndarray) -> Optional[float]:
    """
    Detecta rotação encontrando as bordas do documento na imagem.
    Método mais robusto para documentos com fundo branco.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Ângulo de rotação em graus ou None
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Binarizar para destacar o documento
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inverter se necessário (documento branco em fundo escuro)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    # Operações morfológicas para limpar
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filtrar contornos por área (documento deve ser grande)
    min_area = img.shape[0] * img.shape[1] * 0.2  # Pelo menos 20% da imagem
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        return None
    
    # Pegar o maior contorno (assumindo que é o documento)
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Calcular retângulo rotacionado mínimo
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Normalizar ângulo
    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90
    
    return float(angle)


def detect_rotation_angle_combined(img: np.ndarray) -> float:
    """
    Detecta rotação combinando múltiplos métodos para maior precisão.
    Versão melhorada e mais agressiva.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Ângulo de rotação em graus
    """
    angles = []
    
    # Método 1: Detecção por contorno do documento (mais confiável para documentos)
    angle_contour = detect_rotation_by_document_contour(img)
    if angle_contour is not None and abs(angle_contour) > 0.05:
        angles.append(angle_contour)
    
    # Método 2: Detecção por bordas do documento
    angle_edges = detect_document_edges(img)
    if angle_edges is not None and abs(angle_edges) > 0.05:
        angles.append(angle_edges)
    
    # Método 3: Hough Lines
    angle1 = detect_rotation_angle(img)
    if abs(angle1) > 0.05:
        angles.append(angle1)
    
    # Método 4: Contornos (método antigo)
    angle2 = detect_rotation_angle_contour(img)
    if abs(angle2) > 0.05:
        angles.append(angle2)
    
    # Método 5: Detecção por linhas horizontais/verticais
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Detectar linhas
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                           minLineLength=min(img.shape[0], img.shape[1]) // 5,
                           maxLineGap=30)
    
    if lines is not None and len(lines) > 0:
        line_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 10 or abs(y2 - y1) > 10:  # Linha significativa
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Normalizar
                    if angle > 45:
                        angle = angle - 90
                    elif angle < -45:
                        angle = angle + 90
                    if abs(angle) < 15:  # Apenas rotações pequenas
                        line_angles.append(angle)
        
        if line_angles:
            # Usar mediana dos ângulos das linhas
            angles.append(np.median(line_angles))
    
    if not angles:
        return 0.0
    
    # Filtrar outliers usando IQR
    angles_array = np.array(angles)
    if len(angles_array) > 2:
        q1 = np.percentile(angles_array, 25)
        q3 = np.percentile(angles_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_angles = angles_array[(angles_array >= lower_bound) & (angles_array <= upper_bound)]
        
        if len(filtered_angles) > 0:
            angles_array = filtered_angles
    
    # Priorizar método de contorno se disponível (mais confiável)
    if angle_contour is not None:
        # Se método de contorno está próximo da mediana, usar ele
        median_angle = np.median(angles_array)
        if abs(angle_contour - median_angle) < 3.0:
            return float(angle_contour)
    
    # Se não, priorizar método de bordas se disponível
    if angle_edges is not None:
        median_angle = np.median(angles_array)
        if abs(angle_edges - median_angle) < 2.0:
            return float(angle_edges)
    
    # Usar mediana dos ângulos (mais robusto que média)
    final_angle = float(np.median(angles_array))
    
    return final_angle


def auto_correct_perspective_and_rotation(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Corrige automaticamente perspectiva (3D→2D) e rotação (2D) do documento.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Tupla (imagem_corrigida, informações_da_correção)
    """
    info = {
        'perspective_corrected': False,
        'rotation_corrected': False,
        'rotation_angle': 0.0,
        'corners_detected': False
    }
    
    result_img = img.copy()
    
    # Passo 1: Corrigir perspectiva (3D → 2D)
    corners = detect_document_corners(img)
    if corners is not None:
        info['corners_detected'] = True
        result_img, success = correct_perspective(result_img, corners)
        if success:
            info['perspective_corrected'] = True
    
    # Passo 2: Corrigir rotação (2D)
    angle = detect_rotation_angle_combined(result_img)
    if abs(angle) > 0.05:
        result_img = rotate_image(result_img, angle)
        info['rotation_corrected'] = True
        info['rotation_angle'] = angle
    
    return result_img, info


def detect_decorative_borders_rg(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detecta bordas decorativas do RG de forma mais precisa.
    
    Args:
        img: Imagem em formato numpy array (RGB)
        
    Returns:
        Tupla (top, bottom, left, right) com coordenadas de crop
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Estratégia: bordas decorativas têm padrões repetitivos e baixa variância
    # Área de conteúdo tem mais variação (texto, foto, etc.)
    
    # 1. Analisar bordas laterais (esquerda e direita)
    # Amostrar colunas das bordas e do centro
    border_sample_width = int(w * 0.15)  # Primeiros e últimos 15%
    center_start = int(w * 0.35)
    center_end = int(w * 0.65)
    
    # Calcular variância média das bordas vs centro
    left_border = gray[:, :border_sample_width]
    right_border = gray[:, -border_sample_width:]
    center_region = gray[:, center_start:center_end]
    
    left_var = np.var(left_border)
    right_var = np.var(right_border)
    center_var = np.var(center_region)
    
    # Se as bordas têm variância muito menor que o centro, são decorativas
    threshold_factor = 0.5  # Bordas têm menos de 50% da variância do centro
    
    # Encontrar onde a variância aumenta (saída da borda decorativa)
    left_crop = 0
    if left_var < center_var * threshold_factor:
        # Analisar coluna por coluna da esquerda para direita
        for col in range(border_sample_width, center_start):
            col_data = gray[:, col]
            col_var = np.var(col_data)
            # Se a variância aumenta significativamente, saímos da borda
            if col_var > left_var * 1.8:
                left_crop = max(0, col - 3)
                break
        if left_crop == 0:
            left_crop = int(w * 0.12)  # Fallback: 12%
    else:
        left_crop = int(w * 0.08)  # Borda pequena
    
    right_crop = w
    if right_var < center_var * threshold_factor:
        # Analisar coluna por coluna da direita para esquerda
        for col in range(w - border_sample_width, center_start, -1):
            col_data = gray[:, col]
            col_var = np.var(col_data)
            if col_var > right_var * 1.8:
                right_crop = min(w, col + 3)
                break
        if right_crop == w:
            right_crop = int(w * 0.88)  # Fallback: 88%
    else:
        right_crop = int(w * 0.92)  # Borda pequena
    
    # 2. Analisar bordas superior e inferior
    border_sample_height = int(h * 0.12)  # Primeiros e últimos 12%
    middle_start = int(h * 0.25)
    middle_end = int(h * 0.75)
    
    top_border = gray[:border_sample_height, :]
    bottom_border = gray[-border_sample_height:, :]
    middle_region = gray[middle_start:middle_end, :]
    
    top_var = np.var(top_border)
    bottom_var = np.var(bottom_border)
    middle_var = np.var(middle_region)
    
    top_crop = 0
    if top_var < middle_var * threshold_factor:
        # Analisar linha por linha de cima para baixo
        for row in range(border_sample_height, middle_start):
            row_data = gray[row, :]
            row_var = np.var(row_data)
            if row_var > top_var * 1.6:
                top_crop = max(0, row - 2)
                break
        if top_crop == 0:
            top_crop = int(h * 0.10)  # Fallback: 10%
    else:
        top_crop = int(h * 0.08)  # Borda pequena
    
    bottom_crop = h
    if bottom_var < middle_var * threshold_factor:
        # Analisar linha por linha de baixo para cima
        for row in range(h - border_sample_height, middle_start, -1):
            row_data = gray[row, :]
            row_var = np.var(row_data)
            if row_var > bottom_var * 1.6:
                bottom_crop = min(h, row + 2)
                break
        if bottom_crop == h:
            bottom_crop = int(h * 0.90)  # Fallback: 90%
    else:
        bottom_crop = int(h * 0.92)  # Borda pequena
    
    return top_crop, bottom_crop, left_crop, right_crop

