#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para pre-cargar geojsons sin procesar rasters.
Útil para preparar datos antes de tests comparativos.
"""

import sys
import argparse
sys.path.append('src')

import mongoengine
from data_manager import DataManager

def main():
    parser = argparse.ArgumentParser(description='Pre-cargar geojsons desde MongoDB')
    parser.add_argument('--limit', type=int, default=20,
                       help='Número de farms a cargar (default: 20)')
    args = parser.parse_args()
    
    # Conectar a MongoDB
    print("🔌 Conectando a MongoDB...")
    mongoengine.connect('ganabosques')
    print("✅ Conectado a MongoDB: ganabosques\n")
    
    # Configuración
    WORKSPACE_DIR = "D:/OneDrive - CGIAR/Proyectos/ganabosques/data/"
    GEOSERVER_URL = "http://localhost:8600/geoserver/deforestation/wcs"
    GEOSERVER_USER = "admin"
    GEOSERVER_PASS = "geoserver"
    
    print("="*70)
    print(f"📥 Pre-cargando geojsons para {args.limit} farms")
    print("="*70)
    
    # Inicializar DataManager
    dm = DataManager(WORKSPACE_DIR, GEOSERVER_URL, GEOSERVER_USER, GEOSERVER_PASS)
    
    # Cargar metadata
    print(f"\n⚠ MODO TESTING: Limitando a {args.limit} farms")
    farms_metadata = dm.load_farms_metadata(limit=args.limit)
    
    # Preparar geojsons (descarga si no existen)
    geojsons_available = dm.prepare_geojsons(farms_metadata)
    
    print("\n" + "="*70)
    print("✅ Pre-carga completada")
    print("="*70)
    print(f"  • Farms cargados: {len(farms_metadata)}")
    print(f"  • Geojsons disponibles: {geojsons_available}")
    print(f"  • Listos para procesar")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
