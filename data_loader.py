import pandas
from sklearn.preprocessing import StandardScaler
 
 
def load_data(file_path, target_col="target"):
    # lecture du fichier selon son extension
    if ".csv" in file_path:
        df = pandas.read_csv(file_path)
    elif ".xlsx" in file_path:
        df = pandas.read_excel(file_path)
    else:
        raise Exception("The file path must be a csv or excel file !")
 
    # verification que la colonne target existe bien 
    if target_col not in df.columns:
        raise Exception("The target column does not exist")
 
    print(df[target_col].value_counts())
 
    features = df.drop(columns=[target_col])
    labels = df[target_col]
 
    return features, labels
 
 
def normalize(features):
    #normalisation les features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return scaler, normalized_features
 
 
def load_normalized_data(file_path, target_col="target"):
    print("Loading data phase")
    features, labels = load_data(file_path, target_col=target_col)
    scaler, normalized_features = normalize(features)
    print("Success : Loading data")
    print("-" * 20)
    return normalized_features, labels, scaler
 
 
if __name__ == "__main__":
    normalized_features, labels, scaler = load_normalized_data(file_path="bienetre.csv")
 
