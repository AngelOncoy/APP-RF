import json
from app.services.face_recognition import extract_face_features, cosine_similarity
from app.services.db_operations    import get_all_users_with_features

def compare_external_image(image_path, similarity_threshold=0.20):
    external_json = extract_face_features(image_path)   # str
    external_vec  = json.loads(external_json)           # list[float]

    users = get_all_users_with_features()               # cada user trae list[float]

    best_sim, best_user = -1.0, None
    for uid, name, last, email, req, vec_db in users:
        sim = cosine_similarity(external_vec, vec_db)   # listas → OK
        if sim > best_sim:
            best_sim, best_user = sim, (uid, name, last, email, req)

    if best_sim >= similarity_threshold and best_user:
        uid, name, last, email, req = best_user
        return {
            "match": True,
            "similarity": best_sim,
            "user_data": {
                "user_id": uid,
                "name": name,
                "last_name": last,
                "email": email,
                "requisitioned": req,
            },
        }
    return {"match": False, "similarity": best_sim, "user_data": None}

def compare_external_image_verbose(image_path, top_k=5):
    import json
    from operator import itemgetter
    external_vec = json.loads(extract_face_features(image_path))

    candidates = []
    for user in get_all_users_with_features():
        uid, name, last, _, _, vec_db = user
        sim = cosine_similarity(external_vec, vec_db)
        candidates.append((sim, f"{uid}  |  {name} {last}"))

    # ordenar por similitud desc.
    top = sorted(candidates, key=itemgetter(0), reverse=True)[:top_k]

    print("\n─ Top coincidencias ─")
    for rank, (sim, label) in enumerate(top, 1):
        print(f"{rank}.  {label:<30}  sim = {sim:.4f}")
