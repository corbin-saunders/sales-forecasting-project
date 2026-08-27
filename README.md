# sales-forecasting-project

Forecasting daily sales for a chain of retail stores using classical statistics, machine learning, and deep learning — with a comparison of all three and an interactive dashboard for the best model.

## Why this matters

Retailers need to predict how much they'll sell in the coming days and weeks to make real decisions — how much stock to order, how many staff to schedule, and when to run promotions. Getting this wrong means either running out of stock (lost sales) or over-ordering (wasted inventory). This project builds and compares several forecasting approaches to predict daily sales at the store level.

## Dataset

This project uses the [Rossmann Store Sales dataset](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales) from Kaggle, containing:

- ~1,017,000 daily sales records
- 1,115 stores
- Data from January 2013 to July 2015
- Includes store-level info (store type, competition distance) and daily context (promotions, holidays)

To run this project yourself, download the dataset from Kaggle and place the CSVs in a `data/` folder (not included in this repo — see note below).

## Key findings (exploratory analysis)

- **Weekly seasonality**: Sales are highest on Mondays and dip toward the middle of the week. Most stores are closed on Sundays; the small number that stay open show unusually high sales, likely due to differences in store type or location.
- **Holiday effect**: Sales spike sharply in the days leading up to Christmas (peak seen Dec 23), consistent with last-minute holiday shopping before stores close.
- **Promotion effect**: All three of the top non-holiday sales days in the dataset coincided with company-wide promotional events — each had 100% of stores running a promotion, compared to a ~38% baseline on a typical day. This suggests promotions may be one of the strongest predictors of sales in this dataset, potentially rivaling the holiday effect.