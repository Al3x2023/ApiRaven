from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pymongo
import numpy as np
import pandas as pd
from typing import Optional, List
from enum import Enum
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import threading
import time

router = APIRouter(prefix="/emergency", tags=["Emergency Services"])

# Configuración MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["alerta_raven"]
sensors_collection = db["sensor_data"]
emergencies_collection = db["emergencies"]
training_collection = db["training_data"]  # Colección para IA

# Rutas para modelos
MODEL_PATH = "event_classifier.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

# Enumeración de tipos de evento
class EventType(str, Enum):
    NORMAL = "normal"
    PHONE_DROP = "phone_drop"
    VEHICLE_ACCIDENT = "vehicle_accident"
    MANUAL_ALERT = "manual"
    OTHER_IMPACT = "other_impact"

# Modelos Pydantic
class SensorData(BaseModel):
    device_id: str
    accelerometer: dict  # {x: float, y: float, z: float}
    gyroscope: Optional[dict] = None
    magnetometer: Optional[dict] = None
    timestamp: Optional[datetime] = None
    location: Optional[dict] = None  # {lat: float, lng: float}
    event_type: Optional[EventType] = EventType.NORMAL
    user_label: Optional[str] = None  # Para etiquetado manual

class EmergencyAlert(BaseModel):
    device_id: str
    event_type: EventType
    confidence: float  # 0-1
    sensor_data: dict  # Datos brutos del sensor
    location: dict
    timestamp: datetime = datetime.utcnow()
    status: str = "pending"  # pending/verified/false_alarm
    user_label: Optional[str] = None

class TrainingData(BaseModel):
    device_id: str
    sensor_readings: dict
    event_type: EventType
    user_label: Optional[str]
    timestamp: datetime
    features: dict  # Características calculadas

class ModelTrainingRequest(BaseModel):
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: Optional[int] = None
    retrain: bool = False

# Estado global para el modelo
model = None
label_encoder = None
model_lock = threading.Lock()

def initialize_model():
    """Carga el modelo y el codificador si existen"""
    global model, label_encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            print("Modelo cargado desde disco")
        else:
            label_encoder = LabelEncoder()
            print("Ningún modelo encontrado, se creará uno nuevo")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        model = None
        label_encoder = LabelEncoder()

# Inicializar al importar
initialize_model()

def extract_features(sensor_data: dict) -> dict:
    """Extrae características de los datos del sensor"""
    acc = sensor_data.get('accelerometer', {'x': 0, 'y': 0, 'z': 0})
    gyro = sensor_data.get('gyroscope', {'x': 0, 'y': 0, 'z': 0})
    mag = sensor_data.get('magnetometer', {'x': 0, 'y': 0, 'z': 0})
    
    # Cálculo de características
    total_acc = (acc['x']**2 + acc['y']**2 + acc['z']**2)**0.5
    total_gyro = (gyro['x']**2 + gyro['y']**2 + gyro['z']**2)**0.5
    total_mag = (mag['x']**2 + mag['y']**2 + mag['z']**2)**0.5
    
    return {
        'acc_x': acc['x'],
        'acc_y': acc['y'],
        'acc_z': acc['z'],
        'gyro_x': gyro['x'],
        'gyro_y': gyro['y'],
        'gyro_z': gyro['z'],
        'mag_x': mag['x'],
        'mag_y': mag['y'],
        'mag_z': mag['z'],
        'total_acc': total_acc,
        'total_gyro': total_gyro,
        'total_mag': total_mag,
        'acc_peak': max(abs(acc['x']), abs(acc['y']), abs(acc['z'])),
        'gyro_peak': max(abs(gyro['x']), abs(gyro['y']), abs(gyro['z'])),
        'mag_peak': max(abs(mag['x']), abs(mag['y']), abs(mag['z'])),
        'acc_var': np.var([acc['x'], acc['y'], acc['z']])
    }

def predict_event_type(features: dict) -> str:
    """Predice el tipo de evento usando el modelo de Random Forest"""
    global model, label_encoder
    
    if model is None or label_encoder is None:
        # Si no hay modelo, usar detección básica
        if features['total_acc'] > 15:
            return EventType.PHONE_DROP.value
        return EventType.NORMAL.value
    
    # Convertir características a formato de entrada del modelo
    feature_names = [
        'acc_x', 'acc_y', 'acc_z', 
        'gyro_x', 'gyro_y', 'gyro_z', 
        'mag_x', 'mag_y', 'mag_z',
        'total_acc', 'total_gyro', 'total_mag',
        'acc_peak', 'gyro_peak', 'mag_peak',
        'acc_var'
    ]
    input_data = [features.get(k, 0) for k in feature_names]
    
    # Predecir
    prediction = model.predict([input_data])
    return label_encoder.inverse_transform(prediction)[0]

def train_model_async(request: ModelTrainingRequest):
    """Entrena el modelo de RandomForest de forma asíncrona con validación robusta
    
    Args:
        request: ModelTrainingRequest con parámetros de entrenamiento
        
    Returns:
        float: Precisión del modelo en el conjunto de prueba
        None: Si ocurre un error o no hay suficientes datos
    """
    global model, label_encoder
    
    try:
        # 1. Validación y preparación de parámetros
        params = {
            'n_estimators': max(10, request.n_estimators),  # Mínimo 10 estimadores
            'max_depth': None if request.max_depth is None else max(1, request.max_depth),
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1  # Para ver progreso durante entrenamiento
        }
        
        # 2. Obtener datos de entrenamiento
        data = list(training_collection.find({}))
        if len(data) < 20:  # Mínimo 20 ejemplos para entrenar
            print(f"Insuficientes datos para entrenar. Se necesitan al menos 20, hay {len(data)}")
            return None
            
        # 3. Preparación de datos
        df = pd.DataFrame(data)
        
        # Verificar que tenemos las columnas necesarias
        if 'features' not in df.columns or 'event_type' not in df.columns:
            print("Datos mal formados: faltan columnas 'features' o 'event_type'")
            return None
            
        # Extraer características y etiquetas
        try:
            features = pd.json_normalize(df['features'])
            event_types = df['event_type']
        except Exception as e:
            print(f"Error procesando datos: {str(e)}")
            return None
        
        # 4. Codificación de etiquetas
        with model_lock:
            try:
                y = label_encoder.fit_transform(event_types)
                if len(label_encoder.classes_) < 2:
                    print("Se necesitan al menos 2 clases diferentes para entrenamiento")
                    return None
            except Exception as e:
                print(f"Error codificando etiquetas: {str(e)}")
                return None
        
        # 5. División de datos
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, 
                y, 
                test_size=max(0.1, min(0.5, request.test_size)),  # Asegurar test_size entre 0.1 y 0.5
                random_state=42,
                stratify=y  # Mantener proporción de clases
            )
        except Exception as e:
            print(f"Error dividiendo datos: {str(e)}")
            return None
        
        # 6. Entrenamiento del modelo
        try:
            print(f"Iniciando entrenamiento con {len(X_train)} ejemplos...")
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            
            # 7. Evaluación
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Precisión en test: {acc:.4f}")
            
            # 8. Guardar modelo
            with model_lock:
                model = clf
                joblib.dump(model, MODEL_PATH)
                joblib.dump(label_encoder, LABEL_ENCODER_PATH)
                print(f"Modelo guardado en {MODEL_PATH}")
            
            # 9. Registrar metadatos del entrenamiento
            training_metadata = {
                "timestamp": datetime.utcnow(),
                "accuracy": acc,
                "n_samples": len(X_train),
                "params": params,
                "event_type_counts": dict(df['event_type'].value_counts())
            }
            db["training_metadata"].insert_one(training_metadata)
            
            return acc
            
        except Exception as e:
            print(f"Error durante entrenamiento: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return None
    
@router.post("/sensor-data")
async def receive_sensor_data(data: SensorData):
    """Endpoint para recibir y clasificar datos de sensores"""
    try:
        sensor_data = data.dict()
        sensor_data["timestamp"] = sensor_data.get("timestamp") or datetime.utcnow()
        
        # Extraer características
        features = extract_features({
            'accelerometer': sensor_data['accelerometer'],
            'gyroscope': sensor_data.get('gyroscope', {}),
            'magnetometer': sensor_data.get('magnetometer', {})
        })
        
        # Predecir tipo de evento
        predicted_event = predict_event_type(features)
        sensor_data["event_type"] = predicted_event
        
        # Solo almacenar datos si no son normales o están etiquetados
        if sensor_data["event_type"] != EventType.NORMAL.value or sensor_data.get("user_label"):
            # Guardar datos para entrenamiento
            training_record = {
                "device_id": sensor_data["device_id"],
                "sensor_readings": {
                    "accelerometer": sensor_data["accelerometer"],
                    "gyroscope": sensor_data.get("gyroscope"),
                    "magnetometer": sensor_data.get("magnetometer")
                },
                "event_type": sensor_data["event_type"],
                "user_label": sensor_data.get("user_label"),
                "timestamp": sensor_data["timestamp"],
                "features": features
            }
            training_collection.insert_one(training_record)
        
        # Solo generar alertas para eventos importantes
        if sensor_data["event_type"] != EventType.NORMAL.value:
            # Calcular confianza (simulada)
            confidence = min(features['total_acc'] / 20, 1.0)
            
            alert = {
                "device_id": sensor_data["device_id"],
                "event_type": sensor_data["event_type"],
                "confidence": confidence,
                "sensor_data": sensor_data,
                "location": sensor_data.get("location", {}),
                "timestamp": sensor_data["timestamp"],
                "status": "pending",
                "user_label": sensor_data.get("user_label")
            }
            emergencies_collection.insert_one(alert)
        
        # Guardar datos crudos (solo si no es normal o está etiquetado)
        if sensor_data["event_type"] != EventType.NORMAL.value or sensor_data.get("user_label"):
            result = sensors_collection.insert_one(sensor_data)
            inserted_id = str(result.inserted_id)
        else:
            inserted_id = None
        
        return {
            "status": "data_received",
            "event_detected": sensor_data["event_type"] != EventType.NORMAL.value,
            "event_type": sensor_data["event_type"],
            "confidence": confidence if sensor_data["event_type"] != EventType.NORMAL.value else None,
            "recorded_id": inserted_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/manual-alert")
async def trigger_manual_emergency(alert: EmergencyAlert):
    """Endpoint para alertas manuales con clasificación"""
    try:
        alert_data = alert.dict()
        alert_data["event_type"] = EventType.MANUAL_ALERT.value
        alert_data["status"] = "pending"
        
        # Extraer características
        features = extract_features(alert_data["sensor_data"])
        
        # Guardar como dato de entrenamiento
        training_record = {
            "device_id": alert_data["device_id"],
            "sensor_readings": alert_data["sensor_data"],
            "event_type": EventType.MANUAL_ALERT.value,
            "user_label": alert_data.get("user_label"),
            "timestamp": alert_data["timestamp"],
            "features": features
        }
        training_collection.insert_one(training_record)
        
        result = emergencies_collection.insert_one(alert_data)
        
        return {
            "status": "alert_triggered",
            "alert_id": str(result.inserted_id)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_model_endpoint(request: ModelTrainingRequest):
    """Endpoint para entrenar el modelo de clasificación"""
    try:
        # Validar parámetros antes de entrenar
        if request.max_depth is not None and request.max_depth < 1:
            raise HTTPException(
                status_code=400,
                detail="max_depth debe ser None o un entero mayor o igual a 1"
            )
        
        if request.n_estimators < 1:
            raise HTTPException(
                status_code=400,
                detail="n_estimators debe ser un entero mayor o igual a 1"
            )

        if not 0 < request.test_size < 1:
            raise HTTPException(
                status_code=400,
                detail="test_size debe estar entre 0 y 1"
            )

        # Iniciar entrenamiento en un hilo separado
        thread = threading.Thread(
            target=train_model_async, 
            args=(request,),
            daemon=True
        )
        thread.start()
        
        return {
            "status": "training_started",
            "message": "El entrenamiento del modelo se ha iniciado en segundo plano",
            "params": {
                "n_estimators": request.n_estimators,
                "max_depth": request.max_depth,
                "test_size": request.test_size
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    """Obtener información sobre el modelo actual"""
    global model
    
    if model is None:
        return {"status": "no_model", "message": "No hay modelo cargado"}
    
    return {
        "status": "model_loaded",
        "n_estimators": model.n_estimators,
        "n_features": model.n_features_in_,
        "classes": label_encoder.classes_.tolist()
    }

@router.get("/training-data")
async def get_training_data(limit: int = 1000):
    """Obtener datos para entrenamiento de IA"""
    try:
        data = list(training_collection.find(
            {},
            sort=[("timestamp", pymongo.DESCENDING)],
            limit=limit
        ))
        
        for item in data:
            item["_id"] = str(item["_id"])
            
        return {"data": data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-alert/{alert_id}")
async def verify_alert(alert_id: str, is_real: bool):
    """Marcar alerta como verificada o falsa alarma"""
    try:
        status = "verified" if is_real else "false_alarm"
        
        # Actualizar estado en emergencias
        result = emergencies_collection.update_one(
            {"_id": pymongo.ObjectId(alert_id)},
            {"$set": {"status": status}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Actualizar etiqueta en datos de entrenamiento si es necesario
        if is_real:
            alert = emergencies_collection.find_one({"_id": pymongo.ObjectId(alert_id)})
            if alert and "sensor_data" in alert and "event_type" in alert:
                # Actualizar con etiqueta de usuario
                training_collection.update_one(
                    {
                        "device_id": alert["device_id"],
                        "timestamp": alert["timestamp"],
                        "event_type": alert["event_type"]
                    },
                    {"$set": {"user_label": "verificado"}}
                )
        
        return {"status": "updated", "new_status": status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para obtener alertas activas (mantenido de la versión anterior)
@router.get("/active-alerts")
async def get_active_alerts(device_id: str, limit: int = 5):
    """Obtener alertas recientes para un dispositivo"""
    try:
        alerts = list(emergencies_collection.find(
            {"device_id": device_id},
            sort=[("timestamp", pymongo.DESCENDING)],
            limit=limit
        ))
        
        for alert in alerts:
            alert["_id"] = str(alert["_id"])
            
        return {"alerts": alerts}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Función para entrenamiento periódico automático
def periodic_training():
    """Entrenamiento automático periódico"""
    while True:
        try:
            # Verificar si hay suficientes datos nuevos
            last_trained = training_collection.find_one(
                {"training_marker": True},
                sort=[("timestamp", pymongo.DESCENDING)]
            )
            
            count = training_collection.count_documents({})
            if count < 100:  # Mínimo de datos para entrenar
                time.sleep(3600)  # Esperar 1 hora
                continue
                
            # Entrenar modelo con parámetros por defecto
            train_model_async(ModelTrainingRequest())
            
            # Esperar 6 horas antes del próximo entrenamiento
            time.sleep(6 * 3600)
            
        except Exception as e:
            print(f"Error en entrenamiento periódico: {e}")
            time.sleep(3600)

# Iniciar hilo de entrenamiento automático
training_thread = threading.Thread(target=periodic_training, daemon=True)
training_thread.start()