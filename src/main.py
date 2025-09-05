#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import subprocess
from pathlib import Path

def run_or_exit(script_path: Path, titulo: str, continue_on_error: bool) -> bool:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    print(f"▶ Ejecutando: {script_path}")
    rc = subprocess.run([sys.executable, str(script_path)], check=False).returncode
    if rc != 0:
        print(f"✖ Error ejecutando {script_path.name} (código {rc}).")
        if not continue_on_error:
            print("Pipeline detenido.")
            return False
    else:
        print(f"✔ {script_path.name} finalizado correctamente.")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: direct_risk -> movement_risk -> total_risk"
    )
    # Flags largos
    parser.add_argument("--direct-risk",   action="store_true", help="Ejecutar solo direct_risk")
    parser.add_argument("--movement-risk", action="store_true", help="Ejecutar solo movement_risk")
    parser.add_argument("--total-risk",    action="store_true", help="Ejecutar solo total_risk")
    # Alias cortos
    parser.add_argument("--direct",   action="store_true", help="Alias de --direct-risk")
    parser.add_argument("--movement", action="store_true", help="Alias de --movement-risk")
    parser.add_argument("--total",    action="store_true", help="Alias de --total-risk")
    # Opcional
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continuar con etapas siguientes aunque una falle")

    args = parser.parse_args()

    # Resolver rutas (asume que main.py está junto a los scripts .py)
    scripts_dir = Path(__file__).resolve().parent
    direct_script   = scripts_dir / "direct_risk.py"
    movement_script = scripts_dir / "movement_risk.py"
    total_script    = scripts_dir / "total_risk.py"

    # Determinar etapas a ejecutar
    any_flag = any([args.direct_risk, args.movement_risk, args.total_risk,
                    args.direct, args.movement, args.total])
    stages = []

    if not any_flag:
        # Ejecuta todo en orden
        stages = [
            ("Paso 1: Calculando el riesgo directo",       direct_script),
            ("Paso 2: Calculando el riesgo por movilización", movement_script),
            ("Paso 3: Calculando el riesgo total",         total_script),
        ]
    else:
        if args.direct_risk or args.direct:
            stages.append(("Paso 1: Calculando el riesgo directo", direct_script))
        if args.movement_risk or args.movement:
            stages.append(("Paso 2: Calculando el riesgo por movilización", movement_script))
        if args.total_risk or args.total:
            stages.append(("Paso 3: Calculando el riesgo total", total_script))

    if not stages:
        print("No se seleccionaron etapas. Usa --direct-risk/--movement-risk/--total-risk (o sus alias).")
        sys.exit(1)

    ok = True
    for title, script in stages:
        if not script.exists():
            print(f"✖ No se encontró el script: {script}")
            ok = False
            if not args.continue_on_error:
                break
            else:
                continue

        if not run_or_exit(script, title, args.continue_on_error):
            ok = False
            if not args.continue_on_error:
                break

    if ok:
        print("\n🎉 Pipeline completado con éxito.")
        sys.exit(0)
    else:
        print("\n⚠ Pipeline finalizado con errores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
