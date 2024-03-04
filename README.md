# Solana Multivariate Time Series Analysis using LSTM
This repository contains the code for the Solana Multivariate Time Series Analysis using LSTM. The code is written in Python and uses the PyTorch library for the LSTM model.

----------------------------------------------------

## Requirements

The code is written in Python 3.9.2. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used is from Binance API. The dataset contains the following columns:

- `unix`: the unix timestamp of the observation
- `date`: the date and time of the observation
- `open`: the opening price of the asset
- `high`: the highest price of the asset
- `low`: the lowest price of the asset
- `close`: the closing price of the asset
- volume for `USDT volume`/`Solana volume`: the volume of the asset
- `trade_count`: the volume of the quote asset


## Model

The model used is a Long Short-Term Memory (LSTM) network. The model is implemented using the PyTorch library.

## Instructions

In trianing the model, API and CLI is currently on the works. Not yet implemented.

## Results

The model was trained on the dataset and the following results were obtained:

- Validation Loss: ``0.03``

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgements

- The dataset used in this project was provided by Binance API.
- The implementation of the LSTM model was inspired by the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
- The tutorial on LSTM networks by [curiousily | Venelin Valkov](https://github.com/curiousily). [youtube video](https://www.youtube.com/watch?v=ODEGJ_kh2aA)