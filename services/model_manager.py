import random
from crypto_app.models import MLModel

def get_active_models():
    return MLModel.objects.filter(is_active=True)

def choose_model():
    models = list(get_active_models())

    if not models:
        return None

    # 🎯 Weighted random selection (A/B testing)
    total_weight = sum(m.traffic_percentage for m in models)

    r = random.uniform(0, total_weight)
    upto = 0

    for m in models:
        if upto + m.traffic_percentage >= r:
            return m
        upto += m.traffic_percentage

    return models[0]