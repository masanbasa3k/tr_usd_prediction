
# Exchange Rate Prediction with LSTM

This project utilizes a Long Short-Term Memory (LSTM) neural network to predict exchange rates. The model is trained on historical exchange rate data and can be used to make future predictions.

## Project Overview

- The project uses PyTorch for implementing the LSTM model.
- Exchange rate data is normalized using Min-Max scaling.
- The LSTM model is trained on a portion of the data and then evaluated on the remaining test data.
- Users can input the number of days they want to forecast, and the model will provide future exchange rate predictions.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

### Installation

Clone the repository:

```bash
git clone https://github.com/masanbasa3k/tr_usd_prediction.git
cd tr_usd_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the training script to train the LSTM model:

```bash
python predict_model.py
```

2. Run the prediction script to make future forecasts:

```bash
python predict_next_days.py
```

Enter the number of days you want to forecast when prompted.

## Results

The project includes visualizations of real exchange rate values, model predictions, and future forecasts.

### Training Plot

![Figure_3](https://github.com/masanbasa3k/tr_usd_prediction/assets/66223190/32a0ce2f-4052-4361-a00b-5ad52abb0390)

### Future Predictions Plot

![Figure_5](https://github.com/masanbasa3k/tr_usd_prediction/assets/66223190/9847651c-1209-48c6-8b2f-70905e385e76)
