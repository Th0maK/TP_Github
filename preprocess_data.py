from sklearn.model_selection import train_test_split

def preprocess_data(data, testSize):
    """Nettoie, met en forme les données et prépare les ensembles de train et 
    de test"""
    # Nous mélangeons et prenons la sid 42
    train, test = train_test_split(data, test_size = testSize, random_state=42,shuffle=True)
    return train, test


