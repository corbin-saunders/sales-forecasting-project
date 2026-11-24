import pandas as pd
import numpy as np
import torch

from features import FEATURE_COLUMNS

LOOKUP_CAT_COLS = ['StoreID', 'StoreType', 'Assortment', 'StateHolidayType']

FIXED_CAT_COLS = {'DayOfWeek': 8, 'Month': 13, 'WeekOfYear': 54, 'DayOfMonth': 32}

FLAG_COLS = ['Promo', 'SchoolHoliday', 'IsStateHoliday', 'Promo2']

SALES_COLS = ['SalesLag7', 'SalesLag14', 'SalesLag28', 'SalesLagMean4w', 'SalesLagMean8w']
LOG_COLS = SALES_COLS + ['CompetitionDistance']
NUM_COLS = SALES_COLS + ['CompetitionMonths', 'CompetitionDistance', 'DayOfYear']
MISSING_FLAG_COLS = SALES_COLS + ['CompetitionMonths']


def prepare_inputs(train, test, device=None):

    """
    Turns train and test into tensors for the network. Everything that is learned (category lookups, means, standard deviations, target scale) comes from train only.
    Returns a dictionary of tensors and sizes.

    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_codes = []
    test_codes = []
    cat_sizes = []

    for col in LOOKUP_CAT_COLS:
        values = sorted(train[col].astype(str).unique())
        lookup = {value: i for i, value in enumerate(values)}
        unseen = len(lookup)

        train_codes.append(train[col].astype(str).map(lookup).astype(int).to_numpy())
        test_codes.append(test[col].astype(str).map(lookup).fillna(unseen).astype(int).to_numpy())
        cat_sizes.append(len(lookup) + 1)

    assert (test_codes[0] < cat_sizes[0] - 1).all(), "Every test store should be in train."

    for col, size in FIXED_CAT_COLS.items():
        train_codes.append(train[col].astype(int).to_numpy())
        test_codes.append(test[col].astype(int).to_numpy())
        cat_sizes.append(size)

    train_cat = np.stack(train_codes, axis=1)
    test_cat = np.stack(test_codes, axis=1)

    train_flags = train[FLAG_COLS].to_numpy(dtype=np.float32)
    test_flags = test[FLAG_COLS].to_numpy(dtype=np.float32)

    train_missing = train[MISSING_FLAG_COLS].isna().to_numpy(dtype=np.float32)
    test_missing = test[MISSING_FLAG_COLS].isna().to_numpy(dtype=np.float32)

    train_num = train[NUM_COLS].astype(float).copy()
    test_num = test[NUM_COLS].astype(float).copy()

    train_num[LOG_COLS] = np.log1p(train_num[LOG_COLS])
    test_num[LOG_COLS] = np.log1p(test_num[LOG_COLS])

    means = train_num.mean()
    stds = train_num.std().replace(0, 1).fillna(1)

    train_num = ((train_num - means) / stds).fillna(0)
    test_num = ((test_num - means) / stds).fillna(0)

    train_num_block = np.hstack([train_num.to_numpy(dtype=np.float32), train_flags, train_missing])
    test_num_block = np.hstack([test_num.to_numpy(dtype=np.float32), test_flags, test_missing])

    target_scale = float(train['Sales'].mean())
    train_y = (train['Sales'] / target_scale).to_numpy(dtype=np.float32)

    assert not np.isnan(train_num_block).any() and not np.isnan(test_num_block).any(), "Numeric block should not contain NaN."
    assert not np.isnan(train_y).any(), "Target should not contain NaN."
    assert (train_cat.max(axis=0) < np.array(cat_sizes)).all() and (test_cat.max(axis=0) < np.array(cat_sizes)).all(), "A code is outside its embedding size."
    assert (train_cat.min(axis=0) >= 0).all() and (test_cat.min(axis=0) >= 0).all(), "Codes should not be negative."
    assert len(train_cat) == len(train_num_block) == len(train_y) == len(train), "Train row counts should match."
    assert len(test_cat) == len(test_num_block) == len(test), "Test row counts should match."

    return {
        'train_cat': torch.tensor(train_cat, dtype=torch.long, device=device),
        'train_num': torch.tensor(train_num_block, dtype=torch.float32, device=device),
        'train_y': torch.tensor(train_y, dtype=torch.float32, device=device),
        'test_cat': torch.tensor(test_cat, dtype=torch.long, device=device),
        'test_num': torch.tensor(test_num_block, dtype=torch.float32, device=device),
        'cat_sizes': cat_sizes,
        'num_features': train_num_block.shape[1],
        'target_scale': target_scale,
    }

NN_PARAMS = {
    'store_embed_dim': 16,
    'embed_dim': 4,
    'hidden_size': 256,
    'n_layers': 2,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 0.00001,
    'epochs': 10,
    'batch_size': 2048,
    'seed': 42,
    'n_seeds': 1
}


class SalesNet(torch.nn.Module):

    """
    Embeds each categorical column, joins the embeddings to the numeric block and passes them through a small feed forward network. 
    Outputs one number per row, the sales divided by target_scale.

    """

    def __init__(self, cat_sizes, num_features, params):
        super().__init__()

        dims = [params['store_embed_dim']] + [params['embed_dim']] * (len(cat_sizes) - 1)

        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(size, dim) for size, dim in zip(cat_sizes, dims)])

        layers = []
        width = sum(dims) + num_features

        for _ in range(params['n_layers']):
            layers += [torch.nn.Linear(width, params['hidden_size']), torch.nn.ReLU(), torch.nn.Dropout(params['dropout'])]
            width = params['hidden_size']

        layers.append(torch.nn.Linear(width, 1))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, cat, num):
        embedded = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [num], dim=1)
        return self.mlp(x).squeeze(1)


def train_and_predict(inputs, params, seed):

    """
    Trains one network with L1 loss and returns its predictions for the test rows in the original sales units (not yet clipped).
    Tensors stay on the device and batches are cut straight from them, so no DataLoader is needed.

    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = inputs['train_cat'].device

    model = SalesNet(inputs['cat_sizes'], inputs['num_features'], params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    loss_fn = torch.nn.L1Loss()

    n_rows = len(inputs['train_y'])
    batch_size = params['batch_size']

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    model.train()
    for epoch in range(params['epochs']):
        order = torch.randperm(n_rows, device=device, generator=generator)

        for start in range(0, n_rows, batch_size):
            idx = order[start:start + batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(inputs['train_cat'][idx], inputs['train_num'][idx]), inputs['train_y'][idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(inputs['test_cat'], inputs['test_num'])

    return (preds * inputs['target_scale']).cpu().numpy()


def add_nn_forecast(train, test, cutoff, params=NN_PARAMS):

    """
    Same job as add_xgb_forecast: fits on train, forecasts test, returns test with a new 'nn_forecast' column.
    When params['n_seeds'] is more than 1, the predictions of several networks (seed, seed + 1, ...) are averaged.

    """

    inputs = prepare_inputs(train, test)

    all_preds = [train_and_predict(inputs, params, params['seed'] + i) for i in range(params['n_seeds'])]

    preds_clipped = np.clip(np.mean(all_preds, axis=0), 0, None)

    assert len(preds_clipped) == len(test), "The number of predictions should match the number of test samples."
    assert not np.isnan(preds_clipped).any(), "Predictions should not contain any NaN values."

    return test.assign(nn_forecast=preds_clipped)