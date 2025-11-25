import os
import time
import warnings
import joblib
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from eval import calculate_metrics, calculate_mse, calculate_daily_avg_metrics
from utils.gpu_utils import get_gpu_memory_used
from models.ml_models import train_rf, train_xgb, train_lgbm, train_linear

def train_ml_model(
    config: dict,
    train_data: tuple,
    val_data: tuple,
    test_data: tuple,
    scalers: tuple
):
    """
    Train a traditional ML model and evaluate on the test set.
    
    Args:
        config: dict with model config
        train_data: (Xh_train, Xf_train, y_train, hrs_train, dates_train)
        val_data: (Xh_val, Xf_val, y_val, hrs_val, dates_val)
        test_data: (Xh_test, Xf_test, y_test, hrs_test, dates_test)
        scalers: (scaler_hist, scaler_fcst, scaler_target)
    
    Returns:
        model: trained model
        metrics: dict with evaluation metrics
    """
    Xh_train, Xf_train, y_train, _, _ = train_data
    Xh_test, Xf_test, y_test, _, dates_test = test_data
    _, _, scaler_target = scalers

    def flatten(Xh, Xf):
        """
        Simple feature flattening, maintain feature consistency between DL and ML models
        """
        # Handle NWP-only mode: if Xh has zero features, only use Xf
        h = Xh.reshape(Xh.shape[0], -1)
        if Xf is not None:
            f = Xf.reshape(Xf.shape[0], -1)
            if h.shape[1] > 0:
                return np.concatenate([h, f], axis=1)
            else:
                # NWP-only mode: only use forecast features
                return f
        # If no forecast features, must have historical features
        if h.shape[1] == 0:
            raise ValueError("Both historical and forecast features are empty. Cannot train model.")
        return h

    X_train_flat = flatten(Xh_train, Xf_train)
    X_test_flat  = flatten(Xh_test, Xf_test)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat  = y_test.reshape(y_test.shape[0], -1)

    name = config['model']

    ml_param_keys = {
        'RF':     ['n_estimators', 'max_depth', 'random_state'],
        'XGB':    ['n_estimators', 'max_depth', 'learning_rate', 'verbosity'],
        'LGBM':   ['n_estimators', 'max_depth', 'learning_rate', 'random_state'],
        'Linear': []  # Linear Regression has no hyperparameters
    }
    # Select parameters based on model complexity
    complexity = config.get('model_complexity', 'low')
    all_params = config.get('model_params', {})
    
    # Select parameters for corresponding complexity
    if complexity == 'high' and 'ml_high' in all_params:
        model_params = all_params['ml_high']
    elif complexity == 'low' and 'ml_low' in all_params:
        model_params = all_params['ml_low']
    else:
        # Fall back to use parameters directly from model_params
        model_params = all_params
    
    allowed_keys = ml_param_keys.get(name, [])
    params = {k: model_params[k] for k in allowed_keys if k in model_params}

    if 'learning_rate' in params:
        params['learning_rate'] = float(params['learning_rate'])
    if 'n_estimators' in params:
        params['n_estimators'] = int(params['n_estimators'])
    if 'max_depth' in params and params['max_depth'] is not None:
        params['max_depth'] = int(params['max_depth'])
    if 'random_state' in params:
        params['random_state'] = int(params['random_state'])
    if 'verbosity' in params:
        params['verbosity'] = int(params['verbosity'])

    if name == 'RF':
        trainer = train_rf
    elif name == 'XGB':
        trainer = train_xgb
    elif name == 'LGBM':
        trainer = train_lgbm
    elif name == 'Linear':
        trainer = train_linear
    else:
        raise ValueError(f"Unsupported ML model: {name}")

    # Clear GPU memory before training to avoid OOM
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    start_time = time.time()
    model = trainer(X_train_flat, y_train_flat, params)
    train_time = time.time() - start_time

    # Measure inference time on test set with batch prediction to avoid GPU OOM
    inference_start = time.time()
    
    # Batch prediction to handle large test sets (avoid GPU OOM)
    batch_size = 1000  # Predict 1000 samples at a time
    n_test = len(X_test_flat)
    preds_list = []
    
    # Removed verbose prediction output
    
    for i in range(0, n_test, batch_size):
        end_idx = min(i + batch_size, n_test)
        batch_preds = model.predict(X_test_flat[i:end_idx])
        preds_list.append(batch_preds)
    
    preds_flat = np.vstack(preds_list)
    inference_time = time.time() - inference_start

    # Inverse transform using scaler_target
    # Default to enabled unless explicitly disabled
    inverse_transform = config.get('inverse_transform', True)
    fh = int(config.get('future_hours', 24))  # Default to 24 if not specified
    
    if scaler_target is not None and inverse_transform:
        try:
            y_matrix = scaler_target.inverse_transform(y_test_flat).reshape(-1, fh)
            p_matrix = scaler_target.inverse_transform(preds_flat.reshape(-1, 1)).reshape(-1, fh)
            
            # 验证结果
            if np.any(np.isnan(p_matrix)) or np.any(np.isinf(p_matrix)):
                print("  [Warning] NaN or Inf detected after inverse transform, using original values")
                y_matrix = y_test_flat.reshape(-1, fh)
                p_matrix = preds_flat.reshape(-1, fh)
        except Exception as e:
            print(f"  [Warning] Inverse transform failed: {e}, using original values")
            y_matrix = y_test_flat.reshape(-1, fh)
            p_matrix = preds_flat.reshape(-1, fh)
    else:
        y_matrix = y_test_flat.reshape(-1, fh)
        p_matrix = preds_flat.reshape(-1, fh)
    
    # Clip predictions to reasonable range [0, 100] (capacity factor percentage)
    p_matrix = np.clip(p_matrix, 0, 100)

    # Calculate all evaluation metrics
    # Important: metrics calculated based on inverse-transformed values (capacity factor percentage, 0-100)
    
    # Calculate metrics using daily average method (recommended for day-ahead forecasting)
    # This calculates RMSE/MAE for each day, then averages across days
    # All metrics (MAE, RMSE, R2, NRMSE) are calculated consistently
    daily_metrics = calculate_daily_avg_metrics(y_matrix, p_matrix)
    
    # Extract metrics (all from unified calculate_daily_avg_metrics)
    mse = daily_metrics['rmse'] ** 2  # Convert RMSE back to MSE for compatibility
    rmse = daily_metrics['rmse']
    mae = daily_metrics['mae']
    nrmse = daily_metrics['nrmse']
    
    # Also extract 24h-ahead predictions for saving to CSV (for visualization)
    from eval import extract_one_hour_ahead_predictions
    final_preds_24h, final_gt_24h = extract_one_hour_ahead_predictions(p_matrix, y_matrix)

    # Get GPU memory usage
    gpu_memory_used = get_gpu_memory_used()

    save_dir = config.get('save_dir', './results')  # Default to ./results if not specified
    
    # Decide whether to save model based on config
    save_options = config.get('save_options', {})
    if save_options.get('save_model', False):
        model_dir = os.path.join(save_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))

    metrics = {
        'mse':            mse,
        'rmse':           rmse,  # Daily averaged RMSE
        'mae':            mae,   # Daily averaged MAE
        'nrmse':          nrmse,  # Daily averaged NRMSE (from unified calculation)
        'r_square':       daily_metrics['r2'],
        'r2':             daily_metrics['r2'],
        'smape':          np.nan,  # Not calculated for daily avg
        'best_epoch':     np.nan,  # ML models don't have epoch concept
        'final_lr':       np.nan,  # ML models don't have learning rate concept
        'gpu_memory_used': gpu_memory_used,
        'train_time_sec': round(train_time, 2),
        'inference_time_sec': round(inference_time, 2),
        'param_count':    X_train_flat.shape[1] * y_train_flat.shape[1] + y_train_flat.shape[1],  # Linear model: features * outputs + bias
        'samples_count':  int(len(p_matrix) * p_matrix.shape[1]),  # Number of test samples (total hours)
        'predictions':    final_preds_24h,  # 1D array of 24h-ahead predictions (for CSV)
        'y_true':         final_gt_24h,     # 1D array of corresponding ground truth (for CSV)
        'predictions_all': p_matrix,  # Full multi-step predictions (for potential detailed analysis)
        'y_true_all':     y_matrix,   # Full ground truth (for potential detailed analysis)
        'dates':          dates_test,
        'epoch_logs':     [{'epoch': 1, 'train_loss': np.nan, 'val_loss': mse}],  # ML models don't have train_loss
        'inverse_transformed': False  # Capacity Factor doesn't need inverse normalization
    }

    return model, metrics
