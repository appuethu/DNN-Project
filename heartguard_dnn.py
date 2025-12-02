
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib

def load_data(clinical_path, lifestyle_path):
    clinical = pd.read_csv(clinical_path)
    lifestyle = pd.read_csv(lifestyle_path)
    data = pd.concat([clinical.reset_index(drop=True), lifestyle.reset_index(drop=True)], axis=1)
    return data

def build_model(n_clinical, n_lifestyle):
    # Clinical input
    clinical_input = Input(shape=(n_clinical,), name='clinical_input')
    x = Dense(64, activation='relu')(clinical_input)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)

    # Lifestyle subnetwork / embedding
    lifestyle_input = Input(shape=(n_lifestyle,), name='lifestyle_input')
    y = Dense(32, activation='relu')(lifestyle_input)
    y = Dense(16, activation='relu')(y)
    lifestyle_embedding = Dense(8, activation='relu')(y)

    merged = concatenate([x, lifestyle_embedding])
    z = Dense(128, activation='relu')(merged)
    z = Dropout(0.3)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(32, activation='relu')(z)
    out = Dense(1, activation='sigmoid', name='output')(z)

    model = Model(inputs=[clinical_input, lifestyle_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_and_train(clinical_path, lifestyle_path, model_out_dir):
    os.makedirs(model_out_dir, exist_ok=True)
    data = load_data(clinical_path, lifestyle_path)
    X_clinical = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']].values
    X_lifestyle = data[['steps','sleep_hours','smoker','alcohol_level','stress_level','bmi']].values
    y = data['target'].values

    # split
    Xc_train, Xc_test, Xl_train, Xl_test, y_train, y_test = train_test_split(
        X_clinical, X_lifestyle, y, test_size=0.2, random_state=42, stratify=y)

    # scaling clinical and lifestyle features separately
    sc_clin = StandardScaler().fit(Xc_train)
    sc_ls = StandardScaler().fit(Xl_train)
    Xc_train_s = sc_clin.transform(Xc_train)
    Xc_test_s = sc_clin.transform(Xc_test)
    Xl_train_s = sc_ls.transform(Xl_train)
    Xl_test_s = sc_ls.transform(Xl_test)

    # save scalers
    joblib.dump(sc_clin, os.path.join(model_out_dir, 'scaler_clinical.joblib'))
    joblib.dump(sc_ls, os.path.join(model_out_dir, 'scaler_lifestyle.joblib'))

    model = build_model(n_clinical=Xc_train_s.shape[1], n_lifestyle=Xl_train_s.shape[1])
    model.summary()
    checkpoint = ModelCheckpoint(os.path.join(model_out_dir, 'best_model.h5'), save_best_only=True, monitor='val_loss')
    es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    history = model.fit([Xc_train_s, Xl_train_s], y_train, validation_split=0.15, epochs=200, batch_size=16, callbacks=[checkpoint, es])

    # evaluate
    preds = model.predict([Xc_test_s, Xl_test_s]).ravel()
    auc = roc_auc_score(y_test, preds)
    y_pred = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print('AUC:', auc)
    print('Accuracy:', acc)
    print('Confusion matrix:\\n', cm)
    print('Classification report:\\n', cr)

    # save final model (best is saved by checkpoint)
    model.save(os.path.join(model_out_dir, 'final_model.h5'))
    return model_out_dir

if __name__ == '__main__':
    # quick training runner
    clinical = 'data/heart_clinical_synthetic.csv'
    lifestyle = 'data/lifestyle_synthetic.csv'
    outdir = 'model/artifacts'
    prepare_and_train(clinical, lifestyle, outdir)
