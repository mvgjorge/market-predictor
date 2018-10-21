# market-predictor
analyse and predict opening/closing prices of market(s) based on sentiment analysis of data. By simply changing the types of input data (and target word list), you can predict values for markets ranging from crypto to stock markets

#steps to run:
everything uses relative links, so please run everything from the default directly  <br />
first create a directly called stock_rnn_data:  <br />
```
mkdir stock_rnn_data
```
<br />
Next, run stock-forecast_tweet.py: <br />
```
  python3 ./stock-forecast_tweet.py
```
<br />
it is imperative you run this first as this also scrapes the data. All other files assume you already have data in the correct format
