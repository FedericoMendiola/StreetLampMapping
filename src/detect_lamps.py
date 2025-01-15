from ultralytics import YOLO

# Caricamento del modello pre-addestrato
model = YOLO('yolov8n.pt')  # Puoi sostituire con il modello più adatto

# Applica il rilevamento a un'immagine di test
results = model('data/test_image.jpg')

# Visualizza i risultati
results.show()
