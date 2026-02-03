import os

from New_lstm2 import *
from New_logistic import *
from New_SampleConstruction import *
from flare_preprocessing import *
from utilities import *

DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
lstm_dir = "D:\\0626_temp\\LSTM_models_0530"
model_cache = {}

# Predict with the bootstrapped-models and return mean + CI

def predict_with_ci_whole24(inputs, times, lead, forecast, model_path): # for lstm
    # normalize inputs
    inputs = normalize2(inputs, inputs.copy())
    model_name = f"lead{lead}_Mplus_{forecast}_whole"
    preds = []
    inputs_tensor = torch.from_numpy(inputs).float().to(DEVICE)
    for i in range(30):
        path = os.path.join(model_path, f"{model_name}{i}.pth")
        if path not in model_cache:
            model = lstm(inputs.shape[2]).to(DEVICE)
            #time_loadmodel = time.time()
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model = torch.jit.script(model)
            model_cache[path] = model
            #time_loadmodel_finish = time.time()
            #print(f"Model loaded in {time_loadmodel_finish - time_loadmodel:.3f} sec")
        model = model_cache[path]
        with torch.no_grad():
            #time_predict = time.time()
            out = model(inputs_tensor)
            pred = 1 / (1 + np.exp(-out.cpu().numpy()))
            pred = pred.flatten()
            #time_predic_finish = time.time()
            #print(f"Predic score in {time_predic_finish - time_predict:.3f} sec")
            preds.append(pred)
    preds = np.array(preds)
    #print(f"Predic score shape: {preds.shape}")
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    ci = std 
    return times, mean, mean - ci, mean + ci