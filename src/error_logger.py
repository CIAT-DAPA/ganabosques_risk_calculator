# -*- coding: utf-8 -*-
"""
Sistema de logging de errores en formato CSV para el pipeline de alertas.

Genera un archivo CSV con todos los errores ocurridos durante la ejecución,
facilitando la revisión y corrección posterior.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from threading import Lock


class ErrorLogger:
    """
    Logger de errores que escribe a un archivo CSV.
    
    Formato del CSV:
    - timestamp: Fecha/hora del error
    - stage: Etapa del pipeline (direct, indirect, enterprise, save_db, etc.)
    - farm_id: ID de la finca (sit_code)
    - period: Período siendo procesado
    - error_type: Tipo de error (GEOJSON_NOT_FOUND, RASTER_ERROR, etc.)
    - error_message: Mensaje descriptivo del error
    - additional_info: Información adicional (JSON string)
    """
    
    # Tipos de error predefinidos
    ERROR_TYPES = {
        'GEOJSON_NOT_FOUND': 'GeoJSON de finca no encontrado',
        'GEOJSON_INVALID': 'GeoJSON inválido o corrupto',
        'RASTER_ERROR': 'Error leyendo raster de deforestación',
        'RASTER_NOT_FOUND': 'Raster no encontrado para el período',
        'GEOMETRY_ERROR': 'Error de geometría (transformación, intersección)',
        'DB_CONNECTION': 'Error de conexión a base de datos',
        'DB_SAVE': 'Error guardando en base de datos',
        'CSV_PARSE': 'Error parseando archivo CSV',
        'MOVEMENT_DATA': 'Error en datos de movimientos',
        'ENTERPRISE_NOT_FOUND': 'Empresa no encontrada en BD',
        'FARM_NOT_FOUND': 'Finca no encontrada en BD',
        'UNKNOWN': 'Error desconocido'
    }
    
    def __init__(
        self,
        output_dir: str,
        prefix: str = "errors",
        append_timestamp: bool = True
    ):
        """
        Inicializa el logger de errores.
        
        Args:
            output_dir: Directorio donde guardar el archivo CSV
            prefix: Prefijo del nombre del archivo
            append_timestamp: Si True, añade timestamp al nombre del archivo
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre del archivo
        if append_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.csv"
        else:
            filename = f"{prefix}.csv"
        
        self.filepath = self.output_dir / filename
        self._lock = Lock()
        self._error_count = 0
        self._errors_by_type: Dict[str, int] = {}
        self._initialized = False
        
    def _ensure_initialized(self):
        """Crea el archivo CSV con headers si no existe."""
        if self._initialized:
            return
        
        with self._lock:
            if not self._initialized:
                with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp',
                        'stage',
                        'farm_id',
                        'period',
                        'error_type',
                        'error_message',
                        'additional_info'
                    ])
                self._initialized = True
    
    def log_error(
        self,
        stage: str,
        error_type: str,
        error_message: str,
        farm_id: Optional[str] = None,
        period: Optional[str] = None,
        additional_info: Optional[Dict] = None
    ):
        """
        Registra un error en el CSV.
        
        Args:
            stage: Etapa del pipeline (direct_alert, indirect_alert, etc.)
            error_type: Tipo de error (usar constantes de ERROR_TYPES)
            error_message: Mensaje descriptivo
            farm_id: ID de la finca afectada (opcional)
            period: Período siendo procesado (opcional)
            additional_info: Información adicional como dict (opcional)
        """
        self._ensure_initialized()
        
        timestamp = datetime.now().isoformat()
        
        # Convertir additional_info a string JSON
        if additional_info:
            import json
            info_str = json.dumps(additional_info, ensure_ascii=False)
        else:
            info_str = ""
        
        with self._lock:
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    stage,
                    farm_id or "",
                    period or "",
                    error_type,
                    error_message,
                    info_str
                ])
            
            self._error_count += 1
            self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1
    
    def log_geojson_not_found(self, farm_id: str, stage: str = "direct_alert"):
        """Atajo para error de geojson no encontrado."""
        self.log_error(
            stage=stage,
            error_type='GEOJSON_NOT_FOUND',
            error_message=f"GeoJSON no encontrado para finca {farm_id}",
            farm_id=farm_id
        )
    
    def log_raster_error(
        self,
        period: str,
        error: str,
        stage: str = "direct_alert"
    ):
        """Atajo para error de raster."""
        self.log_error(
            stage=stage,
            error_type='RASTER_ERROR',
            error_message=str(error),
            period=period
        )
    
    def log_db_error(
        self,
        error: str,
        farm_id: Optional[str] = None,
        stage: str = "save_db"
    ):
        """Atajo para error de base de datos."""
        self.log_error(
            stage=stage,
            error_type='DB_SAVE',
            error_message=str(error),
            farm_id=farm_id
        )
    
    def log_geometry_error(
        self,
        farm_id: str,
        error: str,
        period: Optional[str] = None,
        stage: str = "direct_alert"
    ):
        """Atajo para error de geometría."""
        self.log_error(
            stage=stage,
            error_type='GEOMETRY_ERROR',
            error_message=str(error),
            farm_id=farm_id,
            period=period
        )
    
    def get_summary(self) -> Dict:
        """
        Obtiene resumen de errores registrados.
        
        Returns:
            Dict con estadísticas de errores
        """
        return {
            'total_errors': self._error_count,
            'errors_by_type': dict(self._errors_by_type),
            'output_file': str(self.filepath),
            'file_exists': self.filepath.exists()
        }
    
    def print_summary(self):
        """Imprime resumen de errores."""
        summary = self.get_summary()
        
        if summary['total_errors'] == 0:
            print("No se registraron errores")
            return
        
        print(f"\n⚠ RESUMEN DE ERRORES ({summary['total_errors']} total)")
        print("=" * 50)
        
        for error_type, count in summary['errors_by_type'].items():
            description = self.ERROR_TYPES.get(error_type, error_type)
            print(f"  • {error_type}: {count:,} ({description})")
        
        print(f"\n📄 Archivo de errores: {summary['output_file']}")


# Singleton global para uso en todo el pipeline
_global_error_logger: Optional[ErrorLogger] = None


def init_error_logger(output_dir: str, prefix: str = "errors") -> ErrorLogger:
    """
    Inicializa el logger global de errores.
    
    Args:
        output_dir: Directorio de salida
        prefix: Prefijo del archivo
        
    Returns:
        Instancia del ErrorLogger
    """
    global _global_error_logger
    _global_error_logger = ErrorLogger(output_dir, prefix)
    return _global_error_logger


def get_error_logger() -> Optional[ErrorLogger]:
    """Obtiene el logger global de errores."""
    return _global_error_logger


def log_error(
    stage: str,
    error_type: str,
    error_message: str,
    **kwargs
):
    """
    Función de conveniencia para registrar errores usando el logger global.
    
    Args:
        stage: Etapa del pipeline
        error_type: Tipo de error
        error_message: Mensaje de error
        **kwargs: Argumentos adicionales (farm_id, period, additional_info)
    """
    if _global_error_logger:
        _global_error_logger.log_error(stage, error_type, error_message, **kwargs)
