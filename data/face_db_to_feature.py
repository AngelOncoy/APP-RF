"""
rellenar_features_from_db.py
---------------------------------
• Conecta a la BD
• Para cada usuario cuya columna `features` sea NULL o vacía
  – reconstruye la imagen desde los bytes
  – extrae el vector facial (JSON)
  – actualiza la fila
"""
import tempfile
import os
import json
import mysql.connector
from app.database.mysql_connector import get_connection, close_connection
from app.services.face_recognition import extract_face_features

def regenerate_features_for_all():
    conn = get_connection()
    if not conn:
        print("❌ No se pudo conectar a la BD.")
        return

    cur = conn.cursor()

    # ── 1. Seleccionar usuarios sin features (o forzar a todos quitando la condición)
    select_sql = """
        SELECT user_id, image
        FROM users
    """
    cur.execute(select_sql)
    rows = cur.fetchall()

    total = len(rows)
    ok, fail = 0, 0

    for user_id, image_bytes in rows:
        try:
            # ── 2. Guardar imagen en archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            # ── 3. Extraer características (JSON)
            features_json = extract_face_features(tmp_path)

            # ── 4. Actualizar fila
            update_sql = "UPDATE users SET features = %s WHERE user_id = %s"
            cur.execute(update_sql, (features_json, user_id))
            conn.commit()

            ok += 1
        except Exception as e:
            print(f"[ERROR] {user_id}: {e}")
            fail += 1
        finally:
            # limpiar archivo temporal
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    cur.close()
    close_connection(conn)

    print(f"✔ Terminó: {ok} actualizados · {fail} errores · {total} totales")

if __name__ == "__main__":
    regenerate_features_for_all()
