# How to Run This Project

Run every command in a PowerShell terminal, from the project folder.

## 1. Open the terminal

In VS Code: open the project folder, then press `` Ctrl + ` `` (or Terminal > New Terminal).

If the terminal is not already in the project folder, type:

```powershell
cd "c:\Users\saiak\OneDrive\Desktop\Sentiment analysis"
```

## 2. Stop any old copy of the app

Skip this if you have never started the app. If it says no process was found, that is fine.

```powershell
Get-Process python | Stop-Process
```

## 3. Start the web app

The model is already trained, so this is all you need.

```powershell
streamlit run app/streamlit_app.py
```

Wait for the line **"You can now view your Streamlit app in your browser"**, then open:

http://localhost:8501

- Click an example button or paste a review, then click **Analyze**.
- Keep the terminal open while you use the page.
- Press `Ctrl + C` in the terminal to stop the app.
- If port 8501 is busy, add `--server.port 8502` to the command and open that port instead.

## Optional commands

```powershell
python -m src.predict "Battery life is terrible and it stopped working after a week"   # predict in the terminal
python -m src.explain "great quality but the battery is not good"                      # LIME explanation
python -m src.binary_benchmark                                                          # positive-vs-negative test
python -m src.train                                                                     # retrain the model (about 2-3 min)
```

## First-time setup (only on a new computer)

```powershell
pip install -r requirements.txt     # install packages
python -m src.data_loader           # download and prepare the 50K reviews
python -m src.train                 # train and save the model
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Failed to load page" | The app is not running. Run step 3 and wait for the "You can now view" message. |
| `No module named 'src'` | Wrong folder. Run the `cd` command from step 1. |
| "Model not found" in the app | Run `python -m src.train`. |
| Port already in use | Run step 2, or use `--server.port 8502`. |
| A package is missing | Run `pip install -r requirements.txt`. |
