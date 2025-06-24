import pickle

try:
    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        print("✅ Model loaded successfully:", type(model))

    # Load scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        print("✅ Scaler loaded successfully:", type(scaler))

    # Load columns
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
        print("✅ Columns loaded successfully. First 5 columns:", columns[:5])

except FileNotFoundError as e:
    print("❌ File not found:", e)

except pickle.UnpicklingError as e:
    print("❌ Error while unpickling:", e)

except Exception as e:
    print("❌ Some other error occurred:", e)
